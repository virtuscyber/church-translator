"""List available audio devices — run this to find your Dante device names."""

import sounddevice as sd

print("=" * 60)
print("Available Audio Devices")
print("=" * 60)

devices = sd.query_devices()
for i, dev in enumerate(devices):
    direction = []
    if dev["max_input_channels"] > 0:
        direction.append(f"IN({dev['max_input_channels']}ch)")
    if dev["max_output_channels"] > 0:
        direction.append(f"OUT({dev['max_output_channels']}ch)")
    
    marker = ""
    if i == sd.default.device[0]:
        marker += " ◀ DEFAULT INPUT"
    if i == sd.default.device[1]:
        marker += " ◀ DEFAULT OUTPUT"
    
    print(f"  [{i:2d}] {dev['name']:<45} {' + '.join(direction)}{marker}")

print()
print("Set device in config.yaml by name or index number.")
print("Example: input_device: 'Dante Via Input'")
print("Example: input_device: 3")
