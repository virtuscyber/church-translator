#!/usr/bin/env python3
"""Dante/AES67 Network Diagnostic Tool.

Tests every layer of the multicast stack to find why Dante Controller
can't see this machine's AES67 stream.

Usage:
    python scripts/diagnose_dante.py
    python scripts/diagnose_dante.py --target 192.168.1.50   # IP of Dante Controller machine
    python scripts/diagnose_dante.py --fix                    # Attempt auto-fixes
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import struct
import subprocess
import sys
import time


PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
INFO = "ℹ️"

SAP_MULTICAST = "239.255.255.255"
SAP_PORT = 9875
RTP_MULTICAST = "239.69.0.1"
RTP_PORT = 5004


def header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check(label: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    print(f"  {icon} {label}")
    if detail:
        print(f"     {detail}")
    return ok


def warn(label: str, detail: str = ""):
    print(f"  {WARN} {label}")
    if detail:
        print(f"     {detail}")


def info(label: str, detail: str = ""):
    print(f"  {INFO} {label}")
    if detail:
        print(f"     {detail}")


# ── 1. Network Interface ──────────────────────────────────────

def diagnose_interface():
    header("1. Network Interface")

    # Get local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("239.69.0.1", 5004))
        local_ip = s.getsockname()[0]
        s.close()
    except OSError:
        local_ip = "127.0.0.1"

    is_real = local_ip not in ("127.0.0.1", "0.0.0.0")
    check("Local IP detected", is_real, local_ip)

    if local_ip.startswith("127."):
        print(f"\n  {FAIL} PROBLEM: No network interface found.")
        print("     → Make sure Ethernet is connected and has an IP address.")
        print("     → Run: ip addr show (Linux) or ipconfig (Windows)")
        return local_ip, False

    # Check if it's an Ethernet interface (not WiFi — multicast is unreliable on WiFi)
    is_ethernet = True
    if platform.system() == "Linux":
        try:
            result = subprocess.run(
                ["ip", "route", "get", "239.69.0.1"],
                capture_output=True, text=True, timeout=5
            )
            iface = ""
            for part in result.stdout.split():
                if part == "dev":
                    idx = result.stdout.split().index("dev")
                    iface = result.stdout.split()[idx + 1]
                    break
            if iface:
                is_wifi = iface.startswith("wl") or "wifi" in iface.lower()
                if is_wifi:
                    is_ethernet = False
                    warn("Using WiFi interface", f"{iface} — multicast is unreliable over WiFi!")
                    print("     → Dante/AES67 requires Ethernet. Connect via Ethernet cable.")
                else:
                    check("Using Ethernet interface", True, iface)
        except Exception:
            pass

    # Check subnet — Dante typically uses 169.254.x.x (link-local) or standard subnet
    if local_ip.startswith("169.254."):
        info("Using link-local address (169.254.x.x)", "This is normal for Dante auto-config")
    elif local_ip.startswith("192.168.") or local_ip.startswith("10.") or local_ip.startswith("172."):
        info("Using private IP range", local_ip)

    return local_ip, is_ethernet


# ── 2. Multicast Socket ──────────────────────────────────────

def diagnose_multicast_socket():
    header("2. Multicast Socket Capability")

    # Test creating RTP multicast socket
    rtp_ok = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)
        sock.sendto(b"test", (RTP_MULTICAST, RTP_PORT))
        sock.close()
        rtp_ok = True
        check("Can send RTP multicast", True, f"{RTP_MULTICAST}:{RTP_PORT}")
    except OSError as e:
        check("Can send RTP multicast", False, str(e))
        print("     → Firewall may be blocking outbound multicast")

    # Test creating SAP multicast socket
    sap_ok = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)
        sock.sendto(b"test", (SAP_MULTICAST, SAP_PORT))
        sock.close()
        sap_ok = True
        check("Can send SAP announcements", True, f"{SAP_MULTICAST}:{SAP_PORT}")
    except OSError as e:
        check("Can send SAP announcements", False, str(e))
        print("     → SAP is how Dante Controller discovers streams")

    return rtp_ok and sap_ok


# ── 3. SAP Announcement Test ──────────────────────────────────

def diagnose_sap_announcement(local_ip: str):
    header("3. SAP Announcement (Stream Discovery)")

    session_id = int(time.time())
    stream_name = "Diagnostic Test Stream"

    sdp = (
        "v=0\r\n"
        f"o=- {session_id} 1 IN IP4 {local_ip}\r\n"
        f"s={stream_name}\r\n"
        f"c=IN IP4 {RTP_MULTICAST}/32\r\n"
        "t=0 0\r\n"
        f"m=audio {RTP_PORT} RTP/AVP 97\r\n"
        f"a=rtpmap:97 L24/48000/1\r\n"
        "a=ptime:1\r\n"
        "a=recvonly\r\n"
        "a=clock-domain:PTPv2 0\r\n"
    )

    flags = (1 << 5)  # SAP v1
    sap_header = struct.pack(
        "!BBH4s",
        flags,
        0,
        session_id & 0xFFFF,
        socket.inet_aton(local_ip),
    )
    payload_type = b"application/sdp\x00"
    sap_packet = sap_header + payload_type + sdp.encode("utf-8")

    info("SAP packet size", f"{len(sap_packet)} bytes")
    info("SDP content:")
    for line in sdp.strip().split("\r\n"):
        print(f"     {line}")

    # Send 3 rapid SAP announcements
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)
        for i in range(3):
            sock.sendto(sap_packet, (SAP_MULTICAST, SAP_PORT))
            time.sleep(0.5)
        sock.close()
        check("Sent 3 SAP announcements", True)
    except OSError as e:
        check("Sent SAP announcements", False, str(e))
        return False

    # Try to receive our own SAP (loopback test)
    info("Testing SAP loopback (can we hear our own announcements?)...")
    try:
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        recv_sock.bind(("", SAP_PORT))

        # Join SAP multicast group
        mreq = struct.pack("4s4s", socket.inet_aton(SAP_MULTICAST), socket.inet_aton("0.0.0.0"))
        recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        recv_sock.settimeout(3.0)

        # Send another SAP while listening
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)
        send_sock.sendto(sap_packet, (SAP_MULTICAST, SAP_PORT))
        send_sock.close()

        try:
            data, addr = recv_sock.recvfrom(4096)
            check("SAP loopback received", True, f"From {addr[0]}, {len(data)} bytes")
        except socket.timeout:
            check("SAP loopback received", False, "Timed out — multicast may not be routing")
            print("     → Your switch may not support multicast / IGMP snooping")
            print("     → Try: connect both machines to the same unmanaged switch")

        recv_sock.close()
    except OSError as e:
        warn("Could not test SAP loopback", str(e))

    return True


# ── 4. Firewall Check ──────────────────────────────────────

def diagnose_firewall():
    header("4. Firewall")

    system = platform.system()

    if system == "Linux":
        # Check iptables / nftables
        try:
            result = subprocess.run(
                ["iptables", "-L", "-n", "--line-numbers"],
                capture_output=True, text=True, timeout=5
            )
            if "DROP" in result.stdout or "REJECT" in result.stdout:
                warn("iptables has DROP/REJECT rules", "May be blocking multicast")
                print("     → Run: sudo iptables -I INPUT -d 239.0.0.0/8 -j ACCEPT")
                print("     → Run: sudo iptables -I INPUT -d 239.255.255.255 -j ACCEPT")
            else:
                check("iptables: no multicast blocks found", True)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            info("iptables not available or not running")

        # Check UFW
        try:
            result = subprocess.run(
                ["ufw", "status"],
                capture_output=True, text=True, timeout=5
            )
            if "active" in result.stdout.lower():
                warn("UFW firewall is active", "May be blocking multicast")
                print("     → Run: sudo ufw allow proto udp to 239.0.0.0/8")
                print("     → Run: sudo ufw allow proto udp from any to any port 9875")
                print("     → Run: sudo ufw allow proto udp from any to any port 5004")
            else:
                check("UFW: inactive", True)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    elif system == "Windows":
        info("Windows detected", "Check Windows Firewall allows:")
        print("     → Outbound UDP to 239.255.255.255:9875 (SAP)")
        print("     → Outbound UDP to 239.69.0.1:5004 (RTP)")
        print("     → Run: netsh advfirewall firewall add rule name=\"AES67 SAP\" dir=out action=allow protocol=udp remoteip=239.0.0.0/8")

    elif system == "Darwin":
        info("macOS detected", "Check System Settings > Network > Firewall")
        print("     → Ensure Python is allowed through the firewall")
        print("     → Or: sudo pfctl -d  (disable temporarily for testing)")


# ── 5. Reachability Test ──────────────────────────────────────

def diagnose_reachability(target_ip: str | None):
    header("5. Network Reachability")

    if not target_ip:
        info("No --target specified", "Skipping cross-machine test")
        print("     → Re-run with: python scripts/diagnose_dante.py --target <DANTE_CONTROLLER_IP>")
        return

    # Ping test
    param = "-n" if platform.system() == "Windows" else "-c"
    try:
        result = subprocess.run(
            ["ping", param, "3", target_ip],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            check(f"Ping to {target_ip}", True)
        else:
            check(f"Ping to {target_ip}", False, "Target unreachable")
            print("     → Machines must be on the same subnet/VLAN")
    except Exception as e:
        check(f"Ping to {target_ip}", False, str(e))

    # UDP reachability hint
    info("To test multicast delivery to the Dante Controller machine:")
    print(f"     On the Dante Controller machine ({target_ip}), run:")
    print(f"       python -c \"import socket,struct; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('',9875)); s.setsockopt(socket.IPPROTO_IP,socket.IP_ADD_MEMBERSHIP,struct.pack('4s4s',socket.inet_aton('239.255.255.255'),socket.inet_aton('0.0.0.0'))); s.settimeout(60); print('Listening for SAP on 239.255.255.255:9875...'); data,addr=s.recvfrom(4096); print(f'GOT SAP from {{addr}}: {{len(data)}} bytes')\"")
    print(f"     Then on THIS machine, run:")
    print(f"       python scripts/test_aes67.py")


# ── 6. Dante-Specific Checks ──────────────────────────────────

def diagnose_dante_specific():
    header("6. Dante / AES67 Compatibility")

    info("Checklist for Dante Controller:")
    print("     1. Open Dante Controller on the other machine")
    print("     2. Go to Device > Device View for your Dante device")
    print("     3. Check: AES67 Config > AES67 Mode = ENABLED")
    print("        (This is OFF by default on most Dante devices!)")
    print("     4. Click 'Reboot' after enabling AES67")
    print("     5. After reboot, check Device > AES67 Monitor tab")
    print("        → Our stream should appear there")
    print()
    info("Common issues:")
    print("     • AES67 not enabled on Dante device (most common!)")
    print("     • Switch doesn't forward multicast (use unmanaged switch)")
    print("     • IGMP snooping enabled but no IGMP querier configured")
    print("     • Different VLANs / subnets (must be same L2 network)")
    print("     • Windows Firewall blocking on Dante Controller machine")
    print("     • Dante Controller version too old (need 4.2+ for AES67)")


# ── 7. TTL Check ──────────────────────────────────────────────

def diagnose_ttl():
    header("7. Multicast TTL")

    # Our default TTL is 32 — check that it's reasonable
    ttl = 32
    check(f"TTL = {ttl}", ttl >= 2, "Must be ≥ 2 to cross switches")
    if ttl == 1:
        print("     → TTL=1 means multicast stays on the local machine only!")
        print("     → Set output.ttl to 32 in config.yaml")


# ── 8. Send persistent test stream ────────────────────────────

def run_test_stream(local_ip: str, duration: int = 30):
    header(f"8. Sending Test Stream ({duration}s)")

    print(f"  Sending SAP + RTP test stream for {duration} seconds...")
    print(f"  Stream: 'Diagnostic Test Stream'")
    print(f"  Multicast: {RTP_MULTICAST}:{RTP_PORT}")
    print(f"  SAP: {SAP_MULTICAST}:{SAP_PORT}")
    print(f"  Origin: {local_ip}")
    print()
    print("  → Open Dante Controller on the other machine NOW")
    print("  → Check Device > AES67 Monitor")
    print("  → You should see 'Diagnostic Test Stream' appear")
    print()

    session_id = int(time.time())
    sdp = (
        "v=0\r\n"
        f"o=- {session_id} 1 IN IP4 {local_ip}\r\n"
        "s=Diagnostic Test Stream\r\n"
        f"c=IN IP4 {RTP_MULTICAST}/32\r\n"
        "t=0 0\r\n"
        f"m=audio {RTP_PORT} RTP/AVP 97\r\n"
        "a=rtpmap:97 L24/48000/1\r\n"
        "a=ptime:1\r\n"
        "a=recvonly\r\n"
        "a=clock-domain:PTPv2 0\r\n"
    )

    flags = (1 << 5)
    sap_header = struct.pack(
        "!BBH4s", flags, 0, session_id & 0xFFFF, socket.inet_aton(local_ip),
    )
    sap_packet = sap_header + b"application/sdp\x00" + sdp.encode("utf-8")

    # SAP socket
    sap_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sap_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)

    # RTP socket (send silence as 440Hz tone)
    rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    rtp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)

    import numpy as np
    rtp_seq = 0
    rtp_ts = 0
    ssrc = int(time.time()) & 0xFFFFFFFF

    start = time.time()
    last_sap = 0

    try:
        while time.time() - start < duration:
            elapsed = time.time() - start

            # Send SAP every 5 seconds
            if time.time() - last_sap >= 5:
                sap_sock.sendto(sap_packet, (SAP_MULTICAST, SAP_PORT))
                last_sap = time.time()
                remaining = duration - int(elapsed)
                print(f"\r  SAP sent ({remaining}s remaining)... ", end="", flush=True)

            # Send 1ms of 440Hz tone as RTP
            samples = 48  # 1ms at 48kHz
            t = np.arange(samples) / 48000.0 + (rtp_ts / 48000.0)
            tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int32)

            # L24 encode
            l24 = bytearray(samples * 3)
            for i, s in enumerate(tone):
                val = int(s * 256) & 0xFFFFFF
                l24[i*3] = (val >> 16) & 0xFF
                l24[i*3+1] = (val >> 8) & 0xFF
                l24[i*3+2] = val & 0xFF

            # RTP header
            byte0 = (2 << 6)
            byte1 = 97 | (0x80 if rtp_seq == 0 else 0)
            header = struct.pack("!BBHII", byte0, byte1, rtp_seq & 0xFFFF, rtp_ts & 0xFFFFFFFF, ssrc)

            rtp_sock.sendto(header + bytes(l24), (RTP_MULTICAST, RTP_PORT))
            rtp_seq += 1
            rtp_ts += samples

            time.sleep(0.001)  # 1ms pacing

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        # Send SAP delete
        del_flags = (1 << 5) | 0x04
        del_header = struct.pack(
            "!BBH4s", del_flags, 0, session_id & 0xFFFF, socket.inet_aton(local_ip),
        )
        del_packet = del_header + b"application/sdp\x00" + sdp.encode("utf-8")
        sap_sock.sendto(del_packet, (SAP_MULTICAST, SAP_PORT))
        sap_sock.close()
        rtp_sock.close()
        print("\n  Test stream stopped. SAP deletion sent.")


def main():
    parser = argparse.ArgumentParser(description="Dante/AES67 Network Diagnostic Tool")
    parser.add_argument("--target", help="IP address of the Dante Controller machine")
    parser.add_argument("--fix", action="store_true", help="Attempt auto-fixes (Linux only)")
    parser.add_argument("--stream", action="store_true", help="Send a 30-second test stream after diagnosis")
    parser.add_argument("--duration", type=int, default=30, help="Test stream duration in seconds (default: 30)")
    args = parser.parse_args()

    print("\n🔍 Dante/AES67 Network Diagnostic Tool")
    print("=" * 60)

    local_ip, is_ethernet = diagnose_interface()
    diagnose_multicast_socket()
    diagnose_sap_announcement(local_ip)
    diagnose_firewall()
    diagnose_reachability(args.target)
    diagnose_dante_specific()
    diagnose_ttl()

    if args.fix and platform.system() == "Linux":
        header("Auto-Fix (Linux)")
        print("  Applying multicast firewall rules...")
        os.system("sudo iptables -I INPUT -d 239.0.0.0/8 -j ACCEPT 2>/dev/null")
        os.system("sudo iptables -I OUTPUT -d 239.0.0.0/8 -j ACCEPT 2>/dev/null")
        check("Firewall rules applied", True)

    if args.stream:
        run_test_stream(local_ip, args.duration)
    else:
        print(f"\n{'='*60}")
        print("  To send a test stream, re-run with --stream:")
        print(f"    python scripts/diagnose_dante.py --stream --duration 60")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
