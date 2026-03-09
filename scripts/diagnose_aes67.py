#!/usr/bin/env python3
"""AES67 diagnostic tool — verifies multicast is working end-to-end.

Run this on the laptop running the Church Translator to check:
1. Which network interface multicast will use
2. Whether multicast packets actually leave the machine
3. Whether a receiver on the same network can hear them

Usage:
    python scripts/diagnose_aes67.py

This sends test packets and listens for them simultaneously.
"""

import socket
import struct
import sys
import threading
import time


MULTICAST_ADDR = "239.69.0.1"
PORT = 5004
SAP_ADDR = "239.255.255.255"
SAP_PORT = 9875
TTL = 32


def get_all_interfaces():
    """List all network interfaces with their IPs."""
    import ipaddress
    results = []
    try:
        # Get all addresses
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Try connecting to multicast to see which interface OS picks
        s.connect((MULTICAST_ADDR, PORT))
        default_ip = s.getsockname()[0]
        s.close()
        results.append(("default_route", default_ip, "← OS picks this for multicast"))
    except OSError as e:
        results.append(("default_route", "FAILED", str(e)))
    
    # Also try getting all IPs via hostname
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            label = "loopback" if ip.startswith("127.") else "interface"
            results.append((label, ip, ""))
    except Exception:
        pass
    
    return results


def test_rtp_send(interface_ip: str):
    """Send a few test RTP packets and check if they leave."""
    print(f"\n📡 Sending test RTP packets to {MULTICAST_ADDR}:{PORT}")
    print(f"   Using interface: {interface_ip}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, TTL)
    
    try:
        sock.setsockopt(
            socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
            socket.inet_aton(interface_ip),
        )
        print(f"   ✅ Bound to interface {interface_ip}")
    except OSError as e:
        print(f"   ❌ Failed to bind interface: {e}")
        sock.close()
        return False
    
    # Send 10 test packets
    for i in range(10):
        # Minimal RTP header (12 bytes) + 144 bytes L24 silence
        header = struct.pack("!BBHII", 0x80, 97, i, i * 48, 0x12345678)
        payload = b'\x00' * 144
        try:
            sock.sendto(header + payload, (MULTICAST_ADDR, PORT))
        except OSError as e:
            print(f"   ❌ Send failed on packet {i}: {e}")
            sock.close()
            return False
        time.sleep(0.001)
    
    print(f"   ✅ Sent 10 test packets successfully")
    sock.close()
    return True


def test_rtp_receive(duration=3.0):
    """Listen for RTP packets on the multicast group."""
    print(f"\n👂 Listening for RTP packets on {MULTICAST_ADDR}:{PORT} ({duration}s)...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        sock.bind(('', PORT))
    except OSError as e:
        print(f"   ❌ Cannot bind to port {PORT}: {e}")
        print(f"   (Is the translator already running? It may be using this port)")
        sock.close()
        return
    
    # Join multicast group
    mreq = struct.pack("4s4s", socket.inet_aton(MULTICAST_ADDR), socket.inet_aton("0.0.0.0"))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(0.5)
    
    packets = 0
    non_silence = 0
    start = time.monotonic()
    
    while time.monotonic() - start < duration:
        try:
            data, addr = sock.recvfrom(2048)
            packets += 1
            # Check if payload (after 12-byte RTP header) contains non-zero audio
            payload = data[12:]
            if any(b != 0 for b in payload):
                non_silence += 1
        except socket.timeout:
            continue
    
    sock.close()
    
    if packets == 0:
        print(f"   ❌ NO packets received! Multicast not reaching this interface.")
        print(f"   → Check: firewall, switch IGMP snooping, VLAN config")
    else:
        expected = int(duration * 1000)  # 1 packet per ms
        print(f"   ✅ Received {packets} packets in {duration}s (expected ~{expected})")
        print(f"   📊 Packets with audio: {non_silence}/{packets} ({100*non_silence//max(packets,1)}%)")
        if non_silence == 0:
            print(f"   ⚠️  All packets are SILENCE — translator may not be producing audio")


def test_sap_receive(duration=35.0):
    """Listen for SAP announcements."""
    print(f"\n👂 Listening for SAP announcements on {SAP_ADDR}:{SAP_PORT} ({duration}s)...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        sock.bind(('', SAP_PORT))
    except OSError as e:
        print(f"   ❌ Cannot bind to SAP port: {e}")
        sock.close()
        return
    
    mreq = struct.pack("4s4s", socket.inet_aton(SAP_ADDR), socket.inet_aton("0.0.0.0"))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(1.0)
    
    start = time.monotonic()
    found = False
    
    while time.monotonic() - start < duration:
        try:
            data, addr = sock.recvfrom(4096)
            # SAP payload contains SDP after header
            if b"Church Translation" in data:
                print(f"   ✅ SAP announcement received from {addr[0]}!")
                # Extract and show SDP
                sdp_start = data.find(b"v=0")
                if sdp_start >= 0:
                    sdp = data[sdp_start:].decode("utf-8", errors="replace")
                    print(f"   📄 SDP:\n{sdp}")
                found = True
                break
            else:
                print(f"   ℹ️  SAP packet from {addr[0]} (not ours)")
        except socket.timeout:
            continue
    
    if not found:
        print(f"   ⚠️  No SAP announcement heard in {duration}s")
        print(f"   (SAP announces every 30s, make sure translator is running)")
    
    sock.close()


def check_firewall():
    """Check Windows firewall status."""
    import subprocess
    print("\n🔥 Checking firewall...")
    try:
        result = subprocess.run(
            ["netsh", "advfirewall", "show", "allprofiles", "state"],
            capture_output=True, text=True, timeout=5,
        )
        if "ON" in result.stdout.upper():
            print("   ⚠️  Windows Firewall is ON — may block multicast")
            print("   → Try: netsh advfirewall set allprofiles state off (temporarily)")
            print("   → Or add rule: netsh advfirewall firewall add rule name=\"AES67\" dir=out action=allow protocol=UDP remoteip=239.0.0.0/8")
        else:
            print("   ✅ Firewall appears off")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("   ℹ️  Could not check firewall (not Windows or netsh unavailable)")


def main():
    print("=" * 60)
    print("🔍 AES67 Diagnostic Tool — Church Translator")
    print("=" * 60)
    
    # 1. Check interfaces
    print("\n🌐 Network Interfaces:")
    interfaces = get_all_interfaces()
    for label, ip, note in interfaces:
        print(f"   [{label}] {ip} {note}")
    
    default_ip = next((ip for label, ip, _ in interfaces if label == "default_route"), None)
    if not default_ip or default_ip == "FAILED":
        print("   ❌ Cannot determine default interface for multicast!")
        print("   → This machine may not have a valid network route to 239.x.x.x")
        return
    
    # 2. Check firewall
    check_firewall()
    
    # 3. Test sending
    test_rtp_send(default_ip)
    
    # 4. Listen for our stream (if translator is running)
    print("\n" + "=" * 60)
    print("📻 Live stream check (is translator running?)")
    print("=" * 60)
    
    # Run receiver in a thread
    recv_thread = threading.Thread(target=test_rtp_receive, args=(5.0,))
    recv_thread.start()
    recv_thread.join()
    
    # 5. Check SAP
    print("\n💡 Tip: To check SAP announcements, run with --sap flag")
    print("   (takes ~35s since SAP announces every 30s)")
    
    if "--sap" in sys.argv:
        test_sap_receive()
    
    print("\n" + "=" * 60)
    print("📋 Summary & Next Steps")
    print("=" * 60)
    print("""
If packets send OK but Dante devices get no audio:
  1. Check Windows Firewall (most common blocker!)
  2. Check if your network switch supports IGMP snooping
     → Try disabling IGMP snooping on the switch temporarily
  3. Verify laptop and Dante devices are on same VLAN/subnet
  4. Try setting the interface IP manually in config.yaml:
     output:
       interface_ip: "192.168.x.x"  # Your real Ethernet IP
""")


if __name__ == "__main__":
    main()
