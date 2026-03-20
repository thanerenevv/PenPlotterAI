import numpy as np


class PathGenerator:
    def __init__(
        self,
        canvas_width=180.0,
        canvas_height=180.0,
        feed_rate=2000,
        travel_rate=4000,
        pen_up_z=3.0,
        pen_down_z=0.0,
        margin=5.0,
    ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.feed_rate = feed_rate
        self.travel_rate = travel_rate
        self.pen_up_z = pen_up_z
        self.pen_down_z = pen_down_z
        self.margin = margin

    def _scale_contours(self, contours, frame_w, frame_h):
        draw_w = self.canvas_width - 2 * self.margin
        draw_h = self.canvas_height - 2 * self.margin
        scale = min(draw_w / frame_w, draw_h / frame_h)
        offset_x = self.margin + (draw_w - frame_w * scale) / 2
        offset_y = self.margin + (draw_h - frame_h * scale) / 2
        scaled = []
        for contour in contours:
            path = []
            for pt in contour:
                x, y = pt[0]
                sx = x * scale + offset_x
                sy = y * scale + offset_y
                path.append((round(sx, 3), round(sy, 3)))
            if len(path) >= 2:
                scaled.append(path)
        return scaled

    def _sort_paths_nearest(self, paths):
        if not paths:
            return paths
        sorted_paths = []
        remaining = list(paths)
        current_pos = (0.0, 0.0)
        while remaining:
            distances = [
                (i, np.hypot(p[0][0] - current_pos[0], p[0][1] - current_pos[1]))
                for i, p in enumerate(remaining)
            ]
            nearest_idx = min(distances, key=lambda d: d[1])[0]
            nearest = remaining.pop(nearest_idx)
            sorted_paths.append(nearest)
            current_pos = nearest[-1]
        return sorted_paths

    def generate(self, contours, frame_w, frame_h):
        paths = self._scale_contours(contours, frame_w, frame_h)
        paths = self._sort_paths_nearest(paths)
        lines = []
        lines.append("G21")
        lines.append("G90")
        lines.append("G92 X0 Y0 Z0")
        lines.append(f"G0 Z{self.pen_up_z} F{self.travel_rate}")
        lines.append(f"G0 X0 Y0 F{self.travel_rate}")

        for path in paths:
            x0, y0 = path[0]
            lines.append(f"G0 X{x0} Y{y0} F{self.travel_rate}")
            lines.append(f"G1 Z{self.pen_down_z} F{self.feed_rate}")
            for x, y in path[1:]:
                lines.append(f"G1 X{x} Y{y} F{self.feed_rate}")
            lines.append(f"G1 Z{self.pen_up_z} F{self.feed_rate}")

        lines.append(f"G0 Z{self.pen_up_z} F{self.travel_rate}")
        lines.append(f"G0 X0 Y0 F{self.travel_rate}")
        lines.append("M2")
        return "\n".join(lines)

    def stats(self, contours, frame_w, frame_h):
        paths = self._scale_contours(contours, frame_w, frame_h)
        total_points = sum(len(p) for p in paths)
        total_paths = len(paths)
        return {"paths": total_paths, "points": total_points}
