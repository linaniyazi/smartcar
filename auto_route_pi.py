"""
auto_route_pi.py
Autonomous pre-programmed route controller for Raspberry Pi 4.
Sends timed commands to the Arduino over USB serial.

Route:
  1. GO straight        (2 s)
  2. SLOW down          (1.5 s)
  3. TURN_LEFT          (1.2 s  – turning while moving)
  4. GO straight        (2 s)
  5. Full CIRCLE (right)(3.5 s  – turn right while moving)
  6. GO straight        (2 s)
  7. TURN_LEFT          (1.2 s)
  8. U-shape (left half-spin)(2 s  – slow left turn = 180°)
  9. SLOW               (0.8 s)
 10. STOP
"""

import serial
import time
import sys

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
ARDUINO_PORT = '/dev/ttyUSB0'   # Change to /dev/ttyACM0 if needed
BAUD_RATE    = 115200
CONNECT_WAIT = 2.0              # Seconds to wait for Arduino to reset after connection

# ──────────────────────────────────────────────────────────────
# Arduino Bridge
# ──────────────────────────────────────────────────────────────
class ArduinoBridge:
    def __init__(self, port, baud):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(CONNECT_WAIT)          # Wait for Arduino bootloader reset
            self.ser.reset_input_buffer()
            print(f"✅ Arduino connected on {port}")
        except serial.SerialException as e:
            print(f"❌ Cannot open {port}: {e}")
            print("   Check: 'ls /dev/ttyUSB*' or 'ls /dev/ttyACM*'")
            sys.exit(1)

    def send(self, cmd: str):
        """Send a single-line command to Arduino."""
        msg = f"{cmd}\n"
        self.ser.write(msg.encode())
        print(f"  → {cmd}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

# ──────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────
def go(bridge, duration):
    """Move straight at full base speed."""
    bridge.send("GO")
    time.sleep(duration)

def slow(bridge, duration):
    """Move at reduced speed (60 % PWM)."""
    bridge.send("SLOW")
    time.sleep(duration)

def stop(bridge):
    """Full stop with motor ramp-down (Arduino handles ramp)."""
    bridge.send("STOP")
    time.sleep(0.5)   # Small pause so the ramp finishes

def turn_left_moving(bridge, duration):
    """
    Turn left WHILE driving.
    The Arduino keeps the last motor command (GO) active; we just
    change the steering angle.
    """
    bridge.send("GO")
    time.sleep(0.1)
    bridge.send("TURN_LEFT")
    time.sleep(duration)
    bridge.send("STRAIGHT")    # ← New Arduino command: reset servo to center
    time.sleep(0.2)

def turn_right_moving(bridge, duration):
    """Turn right WHILE driving (used for full circle)."""
    bridge.send("GO")
    time.sleep(0.1)
    bridge.send("TURN_RIGHT")
    time.sleep(duration)
    bridge.send("STRAIGHT")
    time.sleep(0.2)

def u_shape(bridge, duration):
    """
    U-turn (half circle to the left) at slow speed.
    Uses SLOW speed + TURN_LEFT for ~180°.
    """
    bridge.send("SLOW")
    time.sleep(0.1)
    bridge.send("TURN_LEFT")
    time.sleep(duration)
    bridge.send("STRAIGHT")
    time.sleep(0.2)

# ──────────────────────────────────────────────────────────────
# Main Route
# ──────────────────────────────────────────────────────────────
def run_route(bridge):
    print("\n🚗 Starting autonomous route...\n")

    # ── Step 1: Go straight
    print("[1/9] GO straight (2 s)")
    go(bridge, 2.0)

    # ── Step 2: Slow down
    print("[2/9] SLOW down (1.5 s)")
    slow(bridge, 1.5)

    # ── Step 3: Turn left then continue
    print("[3/9] TURN LEFT (1.2 s)")
    turn_left_moving(bridge, 1.2)

    # ── Step 4: Go straight again
    print("[4/9] GO straight (2 s)")
    go(bridge, 2.0)

    # ── Step 5: Full circle (right turn for ~360°)
    print("[5/9] Full CIRCLE right (3.5 s)")
    turn_right_moving(bridge, 3.5)

    # ── Step 6: Go straight
    print("[6/9] GO straight (2 s)")
    go(bridge, 2.0)

    # ── Step 7: Turn left
    print("[7/9] TURN LEFT (1.2 s)")
    turn_left_moving(bridge, 1.2)

    # ── Step 8: U-shape (half-spin left)
    print("[8/9] U-SHAPE – half spin left (2 s)")
    u_shape(bridge, 2.0)

    # ── Step 9: Slow then stop
    print("[9/9] SLOW then STOP")
    slow(bridge, 0.8)
    stop(bridge)

    print("\n✅ Route complete. Car stopped.\n")

# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bridge = ArduinoBridge(ARDUINO_PORT, BAUD_RATE)
    try:
        run_route(bridge)
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user.")
        stop(bridge)
    finally:
        bridge.close()
        print("Serial port closed.")
