import argparse
import sys
import time
import cv2
from camera import Camera
from vision import VisionProcessor
from path_generator import PathGenerator
from plotter import Plotter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--port", type=str, default=None)
    p.add_argument("--canvas-width", type=float, default=180.0)
    p.add_argument("--canvas-height", type=float, default=180.0)
    p.add_argument("--feed-rate", type=int, default=2000)
    p.add_argument("--canny-low", type=int, default=30)
    p.add_argument("--canny-high", type=int, default=100)
    p.add_argument("--pen-up-z", type=float, default=3.0)
    p.add_argument("--pen-down-z", type=float, default=0.0)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", type=str, default="output.gcode")
    p.add_argument("--countdown", type=int, default=3)
    return p.parse_args()


def run_preview(camera, vision, countdown):
    print("Preview active — press SPACE to capture, Q to quit")
    frame = None
    while True:
        frame = camera.capture()
        contours, found_face = vision.process(frame)
        preview = vision.render_preview(frame, contours, found_face)
        cv2.imshow("Pen Plotter AI — Preview", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord(" "):
            if countdown > 0:
                for i in range(countdown, 0, -1):
                    f = camera.capture()
                    disp = f.copy()
                    cv2.putText(
                        disp,
                        str(i),
                        (disp.shape[1] // 2 - 30, disp.shape[0] // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.0,
                        (0, 0, 255),
                        6,
                    )
                    cv2.imshow("Pen Plotter AI — Preview", disp)
                    cv2.waitKey(1000)
                frame = camera.capture()
            cv2.destroyAllWindows()
            return frame
    return frame


def progress_bar(current, total):
    pct = current / total
    bar_len = 40
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {current}/{total}", end="", flush=True)
    if current == total:
        print()


def main():
    args = parse_args()

    vision = VisionProcessor(canny_low=args.canny_low, canny_high=args.canny_high)
    path_gen = PathGenerator(
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        feed_rate=args.feed_rate,
        pen_up_z=args.pen_up_z,
        pen_down_z=args.pen_down_z,
    )

    with Camera(index=args.camera_index) as camera:
        if args.preview:
            frame = run_preview(camera, vision, args.countdown)
        else:
            print("Capturing frame...")
            frame = camera.capture()

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    print("Processing vision...")
    contours, found_face = vision.process(frame)
    print(f"  Face detected : {found_face}")
    print(f"  Contours found: {len(contours)}")

    if not contours:
        print("No contours extracted — try adjusting --canny-low / --canny-high")
        sys.exit(1)

    stats = path_gen.stats(contours, w, h)
    print(f"  Paths: {stats['paths']}  Points: {stats['points']}")

    print("Generating G-code...")
    gcode = path_gen.generate(contours, w, h)

    with open(args.output, "w") as f:
        f.write(gcode)
    print(f"G-code saved to {args.output}")

    if args.dry_run:
        print("Dry run — skipping plotter connection")
        return

    print(f"Connecting to plotter{' on ' + args.port if args.port else ' (autodetect)'}...")
    with Plotter(port=args.port) as plotter:
        plotter.unlock()
        print("Sending G-code to plotter...")
        plotter.send_gcode(gcode, on_progress=progress_bar)
    print("Plot complete.")


if __name__ == "__main__":
    main()
