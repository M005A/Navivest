import argparse, serial, numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="e.g. COM8 or /dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    return p.parse_args()

def main():
    args = parse_args()
    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"Connected to {args.port}")

    fig, ax = plt.subplots()
    data = np.zeros((8,8))
    im = ax.imshow(data, vmin=0, vmax=3000, cmap="turbo", interpolation="nearest")
    plt.colorbar(im, label="Distance (mm)")
    ax.set_title("VL53L5CX — 8×8 Live Distance Heatmap")

    line_pattern = re.compile(r"^D:([0-9,]+)$")

    def update(_):
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return [im]
        m = line_pattern.match(line)
        if m:
            vals = [int(x) for x in m.group(1).split(",")]
            if len(vals) == 64:
                arr = np.array(vals).reshape(8,8)
                im.set_data(arr)
        return [im]

    ani = animation.FuncAnimation(fig, update, interval=30, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
