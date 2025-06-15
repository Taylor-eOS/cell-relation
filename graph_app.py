import tkinter as tk
import threading
import socket
import json
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import settings

data = []
last_stage = None

def listener():
    if False: print("Listener thread starting", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 9999))
    if False: print(f"Socket family={sock.family}, bound to {sock.getsockname()}", flush=True)
    while True:
        packet, _ = sock.recvfrom(1024)
        d = json.loads(packet.decode("utf-8"))
        global last_stage
        if last_stage is None or d["stage"] != last_stage:
            data.clear()
            last_stage = d["stage"]
        data.append(d)

def redraw():
    ax.clear()
    xs = [d["episode"] for d in data]
    ys = [d["success_rate"] for d in data]
    ax.scatter(xs, ys)
    ax.set_ylim(0, 1)
    ax.axhline(y=settings.threshold, color='black', linestyle='-', linewidth=0.7)
    if len(xs) > 1:
        m_all, b_all = np.polyfit(xs, ys, 1)
        x0, x1 = min(xs), max(xs)
        ax.plot([x0, x1], [m_all*x0 + b_all, m_all*x1 + b_all], color='black', linewidth=0.5)
        xs_recent = xs[-25:]
        ys_recent = ys[-25:]
        m_recent, b_recent = np.polyfit(xs_recent, ys_recent, 1)
        ax.plot([xs_recent[0], xs_recent[-1]],
                [m_recent*xs_recent[0] + b_recent, m_recent*xs_recent[-1] + b_recent])
    ax.set_title(f"Stage {last_stage}")
    canvas.draw()
    root.after(1000, redraw)

root = tk.Tk()
width, height = 800, 600
right_margin, top_margin = 100, 50
x = root.winfo_screenwidth() - width - right_margin
y = top_margin
root.geometry(f"{width}x{height}+{x}+{y}")
fig = Figure()
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

threading.Thread(target=listener, daemon=True).start()
root.after(500, redraw)
root.mainloop()

