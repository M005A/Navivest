# code.py — XIAO RP2040 + VL53L5CX (4×4 robust serial streamer)

import time
import board
import busio
import digitalio
from collections import deque

from vl53l5cx import (
    DATA_DISTANCE_MM,
    DATA_TARGET_STATUS,
    STATUS_VALID,
    RESOLUTION_4X4,
)
from vl53l5cx.cp import VL53L5CXCP

# ---------- Hardware ----------
# XIAO RP2040: D5=SCL, D4=SDA, D6=LPn
i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
lpn = digitalio.DigitalInOut(board.D6)
lpn.direction = digitalio.Direction.OUTPUT

def lpn_reset():
    lpn.value = False
    time.sleep(0.02)
    lpn.value = True
    time.sleep(0.12)

lpn_reset()

# ---------- Sensor init ----------
tof = VL53L5CXCP(i2c, lpn=lpn)
tof.init()
tof.resolution = RESOLUTION_4X4

# Conservative settings to reduce invalids
TARGET_FREQ_HZ = 10       # inter-measurement ~100 ms
TIMING_BUDGET_MS = 50     # integration time per frame

# Apply (positional args only; some forks lack these methods)
if hasattr(tof, "set_ranging_timing_budget_ms"):
    try:
        tof.set_ranging_timing_budget_ms(TIMING_BUDGET_MS)
    except Exception as e:
        print("timing budget set failed:", e)

if hasattr(tof, "set_ranging_frequency_hz"):
    try:
        tof.set_ranging_frequency_hz(TARGET_FREQ_HZ)
    except Exception as e:
        print("freq set failed:", e)
else:
    try:
        tof.ranging_freq = TARGET_FREQ_HZ
    except Exception as e:
        print("ranging_freq attr failed:", e)

# Start ranging (no kwargs for maximum compatibility)
tof.start_ranging({DATA_DISTANCE_MM, DATA_TARGET_STATUS})

print("Streaming 4x4 distance frames (robust mode)")

# ---------- Robustness helpers ----------
N = 16  # 4x4 zones
last_good = [0] * N
age = [999] * N                  # frames since last good per cell
HOLD_FRAMES = 5                  # keep last value this many frames
history = deque((), 3)        # 3-frame temporal median

def accept(status, dist):
    return (status == STATUS_VALID) and (50 <= dist <= 4000)

def median_of_history(idx):
    vals = [frame[idx] for frame in history if frame[idx] > 0]
    if not vals:
        return 0
    vals.sort()
    return vals[len(vals)//2]

# ---------- Main loop ----------
while True:
    if tof.check_data_ready():
        r = tof.get_ranging_data()
        dist = list(r.distance_mm)
        stat = list(r.target_status)

        # Validate + hold-last
        clean = [0] * N
        for i in range(N):
            d = int(dist[i])
            s = int(stat[i])
            if accept(s, d):
                clean[i] = d
                last_good[i] = d
                age[i] = 0
            else:
                age[i] += 1
                clean[i] = last_good[i] if age[i] <= HOLD_FRAMES else 0

        # Temporal median
        history.append(clean[:])
        out = [median_of_history(i) for i in range(N)]

        # Emit CSV line for your PC script
        print("D:" + ",".join(str(v) for v in out))

    time.sleep(0.001)
