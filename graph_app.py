import tkinter as tk
import threading
import socket
import json
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

data = []

def listener():
    print("Listener thread starting", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 9999))
    print(f"Socket family={sock.family}, bound to {sock.getsockname()}", flush=True)
    while True:
        packet, _ = sock.recvfrom(1024)
        d = json.loads(packet.decode("utf-8"))
        print("Received:", d, flush=True)
        data.append(d)
        root.after(0, redraw)

def redraw():
    ax.clear()
    xs = [d["episode"] for d in data]
    ys = [d["success_rate"] for d in data]
    ax.plot(xs, ys)
    canvas.draw()
    root.after(1000, redraw)

root = tk.Tk()
fig = Figure()
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

threading.Thread(target=listener, daemon=True).start()
root.after(500, redraw)
root.mainloop()

