#!/usr/bin/env python3
"""
test_canny.py — Correctness + timing comparison: canny_all vs cv2.Canny.

For each image in ../pictures/images_2k and ../pictures/images_4k:
  1. Run cv2.GaussianBlur(5x5, sigma=1.4) + cv2.Canny(L2gradient=True),
     time it with time.perf_counter.
  2. Invoke ./canny_all.x <image_path> <low> <high> <threads>.
  3. Read canny_output.png (CUDA edge image) from disk.
  4. Parse canny_timing.txt to extract the CUDA total time.
  5. Compare pixel-by-pixel — both are {0, 255} binary maps.
  6. Print one row of the table and accumulate for a per-folder summary.

Usage:
    Single image (mirrors canny_all.x):
        python test_canny.py <image_path> [low] [high] [threads]

    Folder sweep:
        python test_canny.py [options]

    --2k-dir DIR      (default ../pictures/images_2k)
    --4k-dir DIR      (default ../pictures/images_4k)
    -n N              max images per folder   (0 = all,  default 50)
    --seed S          random seed             (default 42)
    --exe PATH        CUDA binary path        (default ./canny_all.x / .exe)
    --low FLOAT       low  threshold          (default  50)
    --high FLOAT      high threshold          (default 150)
    -T THREADS        OpenMP threads for CPU hysteresis (default 8)
"""

import argparse, cv2, glob, numpy as np, os, random, re, subprocess, sys, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
cv2.setNumThreads(0)


def list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
        paths.extend(glob.glob(os.path.join(folder, e.upper())))
    return sorted(set(paths))


def pick(paths, n, rng):
    if n <= 0 or n >= len(paths):
        return paths
    return sorted(rng.sample(paths, n))


def parse_timing_txt(path):
    """
    Parse the pretty table written by canny_all.cu.
    Returns dict with keys: h2d, gauss, nms, d2h, hyst, final, total, wall.
    Missing fields default to None.
    """
    keys = {
        "H2D":            "h2d",
        "Gaussian":       "gauss",
        "Sobel + NMS":    "nms",
        "D2H":            "d2h",
        "Hysteresis":     "hyst",
        "Final output":   "final",
        "TOTAL (sum)":    "total",
        "Wall (start":    "wall",   # match 'Wall (start -> end)'
    }
    out = {v: None for v in keys.values()}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.rstrip()
                for label, key in keys.items():
                    if label in line:
                        m = re.search(r"(\d+\.\d+)\s*$", line)
                        if m:
                            out[key] = float(m.group(1))
                            break
    except FileNotFoundError:
        return out
    return out


def run_cuda(exe, img_path, low, high, threads):
    """Invoke canny_all binary; returns True on success."""
    try:
        proc = subprocess.run(
            [exe, img_path, str(low), str(high), str(threads)],
            capture_output=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        sys.stderr.write(f"  runner returned {proc.returncode}\n")
        sys.stderr.write(f"  stderr: {proc.stderr.decode()[:300]}\n")
        return False
    return True


NREPS = 10  # match canny_all.cu's internal NREPS so both sides average over the same N


def process_image(p, exe, low, high, threads):
    """Run OpenCV + CUDA on one image; return (bname, w, h, match, cv_ms, cuda_ms)
    or None on failure. Prints a single result row.

    OpenCV is run NREPS times here; cv_ms is the average. CUDA total comes from
    canny_timing.txt, which is already a per-run average (canny_all.x loops
    NREPS times internally), so we only invoke the binary once."""
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    bname = os.path.basename(p)

    # 1. OpenCV reference — averaged over NREPS runs
    cv_total = 0.0
    edges_cv = None
    for _ in range(NREPS):
        t0 = time.perf_counter()
        blurred = cv2.GaussianBlur(img, (5, 5), 1.4,
                                   borderType=cv2.BORDER_REPLICATE)
        edges_cv = cv2.Canny(blurred, low, high,
                             apertureSize=3, L2gradient=True)
        t1 = time.perf_counter()
        cv_total += (t1 - t0) * 1000.0
    cv_ms = cv_total / NREPS

    # 2. CUDA runner (one call → averages internally, writes canny_output.png + canny_timing.txt)
    if not run_cuda(exe, p, low, high, threads):
        print(f"{bname:<30} {'CUDA FAILED':>11}")
        return None

    # 3. Read CUDA edge output
    edges_cuda = cv2.imread("canny_output.png", cv2.IMREAD_GRAYSCALE)
    if edges_cuda is None or edges_cuda.shape != (h, w):
        print(f"{bname:<30} {'bad PNG':>11}")
        return None

    # 4. Parse CUDA timing file
    t = parse_timing_txt("canny_timing.txt")
    cuda_ms = t["total"] if t["total"] is not None else float("nan")

    # 5. Pixel-by-pixel comparison ({0, 255} maps)
    cv_edge   = (edges_cv   == 255)
    cuda_edge = (edges_cuda == 255)
    match = (cv_edge == cuda_edge).sum() / cv_edge.size * 100.0

    spd = f"{cv_ms / cuda_ms:.2f}x" if cuda_ms > 0 else "N/A"
    print(f"{bname:<30} {f'{w}x{h}':>11} {match:>9.4f}% "
          f"{cv_ms:>12.3f} {cuda_ms:>12.3f} {spd:>10}")

    return (bname, w, h, match, cv_ms, cuda_ms)


def main():
    ap = argparse.ArgumentParser()
    # Positional args mirror canny_all.x: <image> <low> <high> <threads>.
    # If image is given → single-image mode; otherwise folder sweep.
    ap.add_argument("image",   nargs="?", default=None)
    ap.add_argument("pos_low", nargs="?", type=float, default=None)
    ap.add_argument("pos_hi",  nargs="?", type=float, default=None)
    ap.add_argument("pos_T",   nargs="?", type=int,   default=None)
    ap.add_argument("--2k-dir", default="../pictures/images_2k")
    ap.add_argument("--4k-dir", default="../pictures/images_4k")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--low",  type=float, default=50.0)
    ap.add_argument("--high", type=float, default=150.0)
    ap.add_argument("-T", "--threads", type=int, default=8)
    args = ap.parse_args()

    # Positional values override flag defaults when provided.
    low     = args.pos_low if args.pos_low is not None else args.low
    high    = args.pos_hi  if args.pos_hi  is not None else args.high
    threads = args.pos_T   if args.pos_T   is not None else args.threads

    if args.exe is None:
        if os.path.exists("./canny_all.x"):
            args.exe = "./canny_all.x"
        elif os.path.exists("./canny_all.exe"):
            args.exe = "./canny_all.exe"
        else:
            print("ERROR: canny_all.x / .exe not found. Compile first (make).")
            sys.exit(1)

    SEP = "=" * 96
    HDR = (f"{'Image':<30} {'Size':>11} {'Match%':>10} "
           f"{'OpenCV ms':>12} {'CUDA ms':>12} {'Speedup':>10}")

    # ---- Single-image mode --------------------------------------------------
    if args.image is not None:
        if not os.path.exists(args.image):
            print(f"ERROR: image not found: {args.image}")
            sys.exit(1)
        print(SEP)
        print(f"  Full Canny — canny_all vs OpenCV — single image")
        print(f"  low={low}  high={high}  threads={threads}  "
              f"(OpenCV L2gradient=True)")
        print(SEP)
        print(HDR)
        print("-" * 96)
        process_image(args.image, args.exe, low, high, threads)
        return

    # ---- Folder sweep mode --------------------------------------------------
    rng = random.Random(args.seed)
    folders = []
    for label, d in [("2K", getattr(args, "2k_dir")),
                     ("4K", getattr(args, "4k_dir"))]:
        imgs = list_images(d)
        if imgs:
            folders.append((label, d, pick(imgs, args.n, rng)))
        else:
            print(f"  [{label}] no images in {d} — skipped")

    if not folders:
        print("No images found.")
        return

    all_rows = []

    for label, folder, paths in folders:
        print(f"\n{SEP}")
        print(f"  Full Canny — canny_all vs OpenCV — {label}  "
              f"({len(paths)} images from {folder})")
        print(f"  low={low}  high={high}  threads={threads}  "
              f"(OpenCV L2gradient=True)")
        print(SEP)
        print(HDR)
        print("-" * 96)

        cv_times, cuda_times, match_pcts = [], [], []

        for p in paths:
            row = process_image(p, args.exe, low, high, threads)
            if row is None:
                continue
            _, _, _, match, cv_ms, cuda_ms = row
            cv_times.append(cv_ms)
            cuda_times.append(cuda_ms)
            match_pcts.append(match)
            all_rows.append(row)

        if cv_times:
            avg_cv   = sum(cv_times)   / len(cv_times)
            avg_cuda = sum(cuda_times) / len(cuda_times)
            avg_mt   = sum(match_pcts) / len(match_pcts)
            print("-" * 96)
            spd = f"{avg_cv / avg_cuda:.2f}x" if avg_cuda > 0 else "N/A"
            print(f"{'AVERAGE':<30} {'':>11} {avg_mt:>9.4f}% "
                  f"{avg_cv:>12.3f} {avg_cuda:>12.3f} {spd:>10}")

    with open("canny_test_results.txt", "w") as f:
        f.write("image,W,H,match_pct,opencv_ms,cuda_ms\n")
        for row in all_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\nResults written to canny_test_results.txt")


if __name__ == "__main__":
    main()
