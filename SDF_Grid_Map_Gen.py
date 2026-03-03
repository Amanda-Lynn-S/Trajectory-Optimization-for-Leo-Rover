""" 
This is a grid map editor GUI for creating/saving a 2D Grid Map with its Occupancy Grid and computing/saving a 2D Signed Distance Field heatmap and its values.
In map GUI: "save" -> type "map" // "export" -> type "sdf" // put saved documents in folder containing the codes.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import json
import numpy as np
from scipy import ndimage
try:
    from PIL import Image, ImageDraw, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

class Node:
    def __init__(self, x, y, obstacle=False):
        self.x = x
        self.y = y
        self.obstacle = obstacle

class GridGUI:
    def __init__(self, width=10, height=10, cell_size=10, obstacles=None):
        self.width = width #number of cells in x
        self.height = height #number of cells in y
        self.cell_size = cell_size #pixels per cell
        self.obstacles = set(obstacles or [])
        self.root = tk.Tk()
        self.root.title("Grid Map Editor")
        # Initial State
        self.mode = tk.StringVar(value="obstacle")  #'obstacle'|'start'|'end'|'erase'
        self.grid = [[Node(x, y, obstacle=((x, y) in self.obstacles))
                      for y in range(self.height)] for x in range(self.width)] #obstacle = True if (x, y) in self.obstacles
        self.start = None
        self.end = None
        # Toolbar
        toolbar = tk.Frame(self.root, padx=8, pady=6)
        toolbar.pack(fill="x")
        for label, val, key in [
            ("Obstacle (O)", "obstacle", "O"),
            ("Start (S)",    "start",    "S"),
            ("End (E)",      "end",      "E"),
            ("Erase (R)",    "erase",    "R"),
        ]:
            tk.Radiobutton(toolbar, text=label, value=val, variable=self.mode).pack(side="left", padx=4)
        tk.Button(toolbar, text="Clear", command=self.clear_all).pack(side="left", padx=12)
        tk.Button(toolbar, text="Save…", command=self.save_all).pack(side="left", padx=4)
        tk.Button(toolbar, text="Show 0/1 Table", command=self.show_table).pack(side="left", padx=12)
        tk.Button(toolbar, text="Show SDF (nums)", command=self.show_sdf_numbers).pack(side="left", padx=12)
        tk.Button(toolbar, text="Show SDF (color)", command=self.show_sdf_colors).pack(side="left", padx=4)
        tk.Button(toolbar, text="Export SDF", command=self.export_sdf).pack(side="left", padx=4)
        # Canvas
        w = self.width * self.cell_size
        h = self.height * self.cell_size
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="white", highlightthickness=0)
        self.canvas.pack()
        # Events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.root.bind("<Key-o>", lambda e: self.mode.set("obstacle"))
        self.root.bind("<Key-O>", lambda e: self.mode.set("obstacle"))
        self.root.bind("<Key-s>", lambda e: self.mode.set("start"))
        self.root.bind("<Key-S>", lambda e: self.mode.set("start"))
        self.root.bind("<Key-e>", lambda e: self.mode.set("end"))
        self.root.bind("<Key-E>", lambda e: self.mode.set("end"))
        self.root.bind("<Key-r>", lambda e: self.mode.set("erase"))
        self.root.bind("<Key-R>", lambda e: self.mode.set("erase"))
        self.root.bind("<Key-p>", lambda e: self.save_all())
        self.root.bind("<Key-P>", lambda e: self.save_all())
        self.root.bind("<Key-c>", lambda e: self.clear_all())
        self.root.bind("<Key-C>", lambda e: self.clear_all())
        # Drawing Grid
        self.draw_grid()

    # ---------- Drawing ----------
    def draw_grid(self):
        self.canvas.delete("all")
        # Cells
        for x in range(self.width):
            for y in range(self.height):
                node = self.grid[x][y]
                color = "white"
                if node.obstacle:
                    color = "black"
                if self.start is not None and (x, y) == (self.start.x, self.start.y):
                    color = "green"
                if self.end is not None and (x, y) == (self.end.x, self.end.y):
                    color = "red"
                self.canvas.create_rectangle(
                    x * self.cell_size,
                    y * self.cell_size,
                    (x + 1) * self.cell_size,
                    (y + 1) * self.cell_size,
                    fill=color, outline=""
                )
        # Grid lines
        for gx in range(self.width + 1):
            X = gx * self.cell_size
            self.canvas.create_line(X, 0, X, self.height * self.cell_size, fill="#cccccc")
        for gy in range(self.height + 1):
            Y = gy * self.cell_size
            self.canvas.create_line(0, Y, self.width * self.cell_size, Y, fill="#cccccc")

    # ---------- Editing ----------
    def on_click(self, event):
        self._apply_action(event)

    def on_drag(self, event):
        self._apply_action(event)

    def _apply_action(self, event):
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        m = self.mode.get()
        if m == "obstacle":
            self.set_obstacle(x, y, True)
        elif m == "erase":
            self.set_obstacle(x, y, False)
        elif m == "start":
            self.set_start(x, y)
        elif m == "end":
            self.set_end(x, y)
        self.draw_grid()

    def set_obstacle(self, x, y, value=True):
        node = self.grid[x][y]
        if self.start and (x, y) == (self.start.x, self.start.y): #prevent making start/end an obstacle directly
            self.start = None
        if self.end and (x, y) == (self.end.x, self.end.y):
            self.end = None
        node.obstacle = value

    def set_start(self, x, y):
        if self.grid[x][y].obstacle:
            self.grid[x][y].obstacle = False #can't place start on obstacle
        self.start = self.grid[x][y]
        if self.end and (self.end.x, self.end.y) == (x, y):
            self.end = None #ensure start != end

    def set_end(self, x, y):
        if self.grid[x][y].obstacle:
            self.grid[x][y].obstacle = False #can't place end on obstacle
        self.end = self.grid[x][y] 
        if self.start and (self.start.x, self.start.y) == (x, y):
            self.start = None #ensure start != end

    def clear_all(self):
        for x in range(self.width):
            for y in range(self.height):
                self.grid[x][y].obstacle = False
        self.start = None
        self.end = None
        self.draw_grid()

    # ---------- Data ----------
    def to_occupancy(self):
        """Return H x W array with 1=obstacle, 0=free."""
        arr = np.zeros((self.height, self.width), dtype=np.uint8)
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x][y].obstacle:
                    arr[y, x] = 1
        return arr

    def show_table(self):
        arr = self.to_occupancy()
        win = tk.Toplevel(self.root)
        win.title("Occupancy Grid (0=free, 1=obstacle)")
        text = tk.Text(win, width=min(80, self.width * 2), height=min(40, self.height + 2), font=("Courier", 10))
        text.pack(fill="both", expand=True)
        for i in range(arr.shape[0]):
            row = " ".join(str(v) for v in arr[i])
            text.insert("end", row + "\n")
        text.config(state="disabled")

    def compute_sdf(self):
        """Compute the Signed Distance Field (SDF) of the occupancy grid."""
        occ = self.to_occupancy()
        # Distance to nearest obstacle:
        dist_to_obstacle = ndimage.distance_transform_edt(np.logical_not(occ)) #EDT takes 1-occ as input array -> free space=1 & obstacles=0 since EDT computes distance from non-zero/non-background point to nearest zero/background point
        # Distance to nearest free space:
        dist_to_free = ndimage.distance_transform_edt(occ)
        # SDF -> distances across grid space to nearest obstacle (positive outside / negative inside obstacles):
        sdf = dist_to_obstacle - dist_to_free
        return sdf

    def show_sdf_numbers(self):
        sdf = self.compute_sdf()
        win = tk.Toplevel(self.root)
        win.title("SDF (numbers)")
        text = tk.Text(win, width=min(120, self.width * 7),
                    height=min(35, self.height + 3),
                    font=("Courier", 10))
        text.pack(fill="both", expand=True)
        for row in sdf:
            text.insert("end", " ".join(f"{v:6.2f}" for v in row) + "\n")
        text.config(state="disabled")

    def _sdf_rgb(self, sdf):
        # Red (boundary/inside) -> Orange -> Yellow -> Green (far)
        a = float(np.max(np.abs(sdf))) or 1.0
        z = np.clip(sdf / a, -1.0, 1.0)
        rgb = np.zeros((*z.shape, 3), dtype=np.uint8)
        # -- Outside obstacles: red→green gradient --
        outside = z >= 0
        closeness = 1.0 - z[outside] #0=far (green), 1=near (red)
        closeness = np.clip(closeness, 0, 1)
        # Interpolate manually across 4 colors: (0.0=green, 0.33=yellow, 0.66=orange, 1.0=red)
        r = np.zeros_like(closeness)
        g = np.zeros_like(closeness)
        b = np.zeros_like(closeness)
        for i, c in enumerate(closeness):
            if c < 0.33: #green -> yellow
                t = c / 0.33
                r[i] = 0 + (255 - 0) * t
                g[i] = 128 + (255 - 128) * t
                b[i] = 0
            elif c < 0.66: #yellow -> orange
                t = (c - 0.33) / 0.33
                r[i] = 255
                g[i] = 255 - (255 - 165) * t
                b[i] = 0
            else: #orange -> red
                t = (c - 0.66) / 0.34
                r[i] = 255
                g[i] = 165 - 165 * t
                b[i] = 0
        rgb[outside] = np.stack([r, g, b], axis=-1).astype(np.uint8)
        # -- Inside obstacles: solid red --
        inside = ~outside
        rgb[inside] = (255, 0, 0)
        return rgb

    def show_sdf_colors(self):
        if not PIL_AVAILABLE:
            messagebox.showwarning("Preview", "Pillow not available (`pip install Pillow`).")
            return
        from PIL import ImageTk
        sdf = self.compute_sdf()
        rgb = self._sdf_rgb(sdf)
        img = Image.fromarray(rgb, "RGB").resize(
            (self.width * self.cell_size, self.height * self.cell_size),
            resample=Image.NEAREST
        )
        # Build vertical legend
        a = float(np.max(np.abs(sdf))) or 1.0
        H = img.height
        yy = np.linspace(+a, -a, H)[:, None] #(H,1)
        leg_rgb = self._sdf_rgb(yy / a) #(H,1,3)
        legend = Image.fromarray(leg_rgb, "RGB").resize(
            (max(20, self.cell_size), H), Image.NEAREST
        )
        # Compose final canvas
        canvas = Image.new("RGB", (img.width + legend.width + 40, H), "white")
        canvas.paste(img, (0, 0))
        canvas.paste(legend, (img.width + 20, 0))
        # Labels
        draw = ImageDraw.Draw(canvas)
        x0 = img.width + legend.width + 26
        draw.text((x0, 4),        f"+{a:.2f}", fill=(0, 0, 0))
        draw.text((x0, H // 2 - 6), "0", fill=(0, 0, 0))
        draw.text((x0, H - 16),   f"-{a:.2f}", fill=(0, 0, 0))
        # Show in new window
        win = tk.Toplevel(self.root)
        win.title("SDF (colors + legend)")
        panel = tk.Label(win)
        panel.pack(fill="both", expand=True)
        panel.img = ImageTk.PhotoImage(canvas) #keep reference
        panel.config(image=panel.img)

    # ---------- Save ----------
    def save_all(self):
        """Save occupancy.csv (occupancy grid values 0-1), meta.json (start,end,dimensions), and map.png."""
        # Ask a base filename (without extension)
        base = filedialog.asksaveasfilename(
            defaultextension="",
            filetypes=[("All Files", "*.*")],
            title="Save map (choose base name, no extension)"
        )
        if not base:
            return
        # Save occupancy.csv
        occ = self.to_occupancy()
        np.savetxt(base + "_occupancy.csv", occ, fmt="%d", delimiter=",")
        # Save meta.json (start/end + dims)
        meta = {
            "width": self.width,
            "height": self.height,
            "cell_size": self.cell_size,
            "start": None if not self.start else [int(self.start.x), int(self.start.y)],
            "end": None if not self.end else [int(self.end.x), int(self.end.y)],
        }
        with open(base + "_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        # Save map.png (GUI look)
        try:
            self.save_png(base + "_map.png")
        except Exception as e:
            messagebox.showwarning(
                "Saved with warning",
                f"Occupancy and meta saved.\nCouldn't save PNG: {e}"
            )
            return
        messagebox.showinfo("Saved", f"Saved:\n{base}_occupancy.csv\n{base}_meta.json\n{base}_map.png")

    def save_png(self, path):
        """Render a PNG by redrawing the grid into a PIL image."""
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow (PIL) not available. `pip install Pillow` to enable PNG export.")
        W = self.width * self.cell_size
        H = self.height * self.cell_size
        img = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(img)
        # Cells
        for x in range(self.width):
            for y in range(self.height):
                node = self.grid[x][y]
                color = (255, 255, 255) #white
                if node.obstacle:
                    color = (0, 0, 0) #black
                if self.start is not None and (x, y) == (self.start.x, self.start.y):
                    color = (0, 128, 0) #green
                if self.end is not None and (x, y) == (self.end.x, self.end.y):
                    color = (200, 0, 0) #red
                x0 = x * self.cell_size
                y0 = y * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                draw.rectangle([x0, y0, x1, y1], fill=color)
        # Grid lines
        for gx in range(self.width + 1):
            X = gx * self.cell_size
            draw.line([(X, 0), (X, H)], fill=(204, 204, 204))
        for gy in range(self.height + 1):
            Y = gy * self.cell_size
            draw.line([(0, Y), (W, Y)], fill=(204, 204, 204))
        img.save(path)
    
    def export_sdf(self):
        """Save SDF numbers (CSV) and heatmap PNG with legend (red→orange→yellow→green)."""
        sdf = self.compute_sdf()
        base = filedialog.asksaveasfilename(
            defaultextension="",
            filetypes=[("All Files", "*.*")],
            title="Save SDF (choose base name, no extension)"
        )
        if not base:
            return
        # Save numbers
        np.savetxt(base + "_sdf.csv", sdf, fmt="%.6f", delimiter=",")
        # Save colored PNG
        if not PIL_AVAILABLE:
            messagebox.showwarning("Save SDF", "Pillow not available (`pip install Pillow`). CSV saved; PNG skipped.")
            return
        try:
            # Main heatmap
            rgb = self._sdf_rgb(sdf)
            img = Image.fromarray(rgb, "RGB").resize(
                (self.width * self.cell_size, self.height * self.cell_size),
                resample=Image.NEAREST
            )
            # Legend: top = red (near), bottom = green (far)
            a = float(np.max(np.abs(sdf))) or 1.0
            H = img.height
            legend_w = max(20, self.cell_size)
            # Build a vertical gradient of positive SDF from near(0)→far(a)
            z = np.linspace(0.0, 1.0, H, dtype=float).reshape(H, 1)  #(H,1)
            legend_sdf = z * a                                       #(H,1) positive
            leg_rgb = self._sdf_rgb(legend_sdf)                      #(H,1,3)
            legend = Image.fromarray(leg_rgb, "RGB").resize((legend_w, H), Image.NEAREST)
            # Compose canvas 
            extra_pad = 100 
            canvas = Image.new("RGB", (img.width + legend_w + extra_pad, H), "white")
            canvas.paste(img, (0, 0))
            canvas.paste(legend, (img.width + 20, 0))
            # Labels (ASCII safe)
            draw = ImageDraw.Draw(canvas)
            x0 = img.width + legend_w + 30
            draw.text((x0, 4),              "near",   fill=(0, 0, 0))
            draw.text((x0, H // 2 - 6),     "safer",     fill=(0, 0, 0))
            draw.text((x0, H - 16),         "far",  fill=(0, 0, 0))
            canvas.save(base + "_map.png")
            messagebox.showinfo("Saved", f"Saved:\n{base}_values.csv\n{base}_map.png")
        except Exception as e:
            messagebox.showerror("Save SDF", f"Failed to save PNG: {e}\nCSV was saved.")

    # ---------- Main ----------
    def start_gui(self):
        self.draw_grid()
        self.root.mainloop()


if __name__ == "__main__":
    """ 
    Dimensions chosen for realistic testing purposes:
    - Map of 10x10 meters
    - Cell size of 0.1 meter (10 cm = 20 cm / 2 -> half size of rover footprint)
    - Cell size of 10 pixels (1 pixel = 1 cm)
    """
    # Example usage
    grid_size = 100 #cells
    cell_size = 10 #pixels 
    obstacles = [(4, 5), (5, 5), (6, 5)] #example (can modify on GUI)
    gui = GridGUI(grid_size, grid_size, cell_size, obstacles)
    gui.start_gui()
