"""Balance the RDD2022-CN train split with offline augmentation (no new deps).

Why augmentation, not duplication: D40 has 223 instances vs D00's 2869. Plain
x13 repetition shows the identical pixels 13 times (memorization risk). This
script generates transformed *variants* instead — new pixel data every time,
with mathematically exact label transforms — so the model never sees the same
image twice.

Scope: TRAIN ONLY. Val/test are never read for output, never written.
Deterministic: random.Random(0) drives all choices.

Transforms (all exact on axis-aligned boxes, 512x512):
  geometric pool: hflip, vflip, rot90, rot180, rot270, transpose, hflip+rot90
  photometric pool: none, brightness 0.8/1.2, contrast 0.8/1.2 (PIL ImageEnhance)
Variant k of an image = geo[k % 7] + photo[(k // 7) % 5]; 35 unique combos.

Quota: TARGET instances per class (default: majority count, ~full equalization).
Images are assigned to their rarest class (rarest-first) so one image fills one
class quota; co-occurring classes ride along and are reported, not corrected.

Outputs:
  data/rdd2022-cn-balanced/images/train/aug_<stem>_<k>.jpg + labels/train/*.txt
  data/train-balanced.txt  (originals + augmented, absolute paths)
  stdout: before/after instance histogram + per-class variants added.

Usage:
    .venv/bin/python scripts/balance_by_augmentation.py [--target 2900] [--seed 0]
"""

import argparse
import collections
import pathlib
import random

from PIL import Image, ImageEnhance

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_IMG = ROOT / "data" / "rdd2022-cn" / "images" / "train"
SRC_LBL = ROOT / "data" / "rdd2022-cn" / "labels" / "train"
DST = ROOT / "data" / "rdd2022-cn-balanced"
N_CLASSES = 5


def read_label(path):
    """[(cls, xc, yc, w, h)] normalised."""
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            c, *b = line.split()
            out.append((int(c), *map(float, b)))
    return out


def write_label(path, boxes):
    path.write_text("\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                              for c, x, y, w, h in boxes) + "\n")


def geo_hflip(b):
    return [(c, 1 - x, y, w, h) for c, x, y, w, h in b]


def geo_vflip(b):
    return [(c, x, 1 - y, w, h) for c, x, y, w, h in b]


def geo_rot90(b):
    """90 deg clockwise: (x, y, w, h) -> (1-y, x, h, w)."""
    return [(c, 1 - y, x, h, w) for c, x, y, w, h in b]


def geo_rot180(b):
    return [(c, 1 - x, 1 - y, w, h) for c, x, y, w, h in b]


def geo_rot270(b):
    return [(c, y, 1 - x, h, w) for c, x, y, w, h in b]


def geo_transpose(b):
    return [(c, y, x, h, w) for c, x, y, w, h in b]


GEOS = [
    ("hflip", lambda im, b: (im.transpose(Image.FLIP_LEFT_RIGHT), geo_hflip(b))),
    ("vflip", lambda im, b: (im.transpose(Image.FLIP_TOP_BOTTOM), geo_vflip(b))),
    ("rot90", lambda im, b: (im.transpose(Image.ROTATE_270), geo_rot90(b))),
    ("rot180", lambda im, b: (im.transpose(Image.ROTATE_180), geo_rot180(b))),
    ("rot270", lambda im, b: (im.transpose(Image.ROTATE_90), geo_rot270(b))),
    ("transpose", lambda im, b: (im.transpose(Image.TRANSPOSE), geo_transpose(b))),
    ("hflip_rot90", lambda im, b: (
        im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270),
        geo_rot90(geo_hflip(b)))),
]

PHOTOS = [
    ("", lambda im: im),
    ("b08", lambda im: ImageEnhance.Brightness(im).enhance(0.8)),
    ("b12", lambda im: ImageEnhance.Brightness(im).enhance(1.2)),
    ("c08", lambda im: ImageEnhance.Contrast(im).enhance(0.8)),
    ("c12", lambda im: ImageEnhance.Contrast(im).enhance(1.2)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=None,
                    help="instances per class; default = majority count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include", default="",
                    help="only use train images whose stem contains this "
                         "substring, e.g. China_Drone_ for drone-only")
    ap.add_argument("--out-subdir", default="rdd2022-cn-balanced",
                    help="data/<out-subdir>/ receives augmented images+labels")
    ap.add_argument("--list-name", default="train-balanced.txt",
                    help="data/<list-name> train list that is written")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    items = []  # (stem, boxes)
    counts = collections.Counter()
    for lp in sorted(SRC_LBL.glob("*.txt")):
        if args.include and args.include not in lp.stem:
            continue
        boxes = read_label(lp)
        if not boxes:
            continue
        items.append((lp.stem, boxes))
        for c, *_ in boxes:
            counts[c] += 1
    assert items, f"no train images match include={args.include!r}"

    target = args.target or max(counts.values())
    print(f"before: {dict(sorted(counts.items()))} target={target}", flush=True)

    # Assign each image to its rarest class (rarest-first processing).
    by_class = collections.defaultdict(list)
    for stem, boxes in items:
        rarest = min({c for c, *_ in boxes}, key=lambda c: counts[c])
        by_class[rarest].append((stem, boxes))

    out_img = DST.parent / args.out_subdir / "images" / "train"
    out_lbl = DST.parent / args.out_subdir / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    new_files = []
    for cls in sorted(counts, key=lambda c: counts[c]):
        need = target - counts[cls]
        if need <= 0:
            continue
        pool = by_class[cls][:]
        rng.shuffle(pool)
        if not pool:
            print(f"  class {cls}: no images, skipped", flush=True)
            continue
        per_img, extra = divmod(need, len(pool))
        made = 0
        for j, (stem, boxes) in enumerate(pool):
            n = per_img + (1 if j < extra else 0)
            im = Image.open(SRC_IMG / f"{stem}.jpg").convert("RGB")
            for k in range(n):
                gname, gfn = GEOS[k % len(GEOS)]
                pname, pfn = PHOTOS[(k // len(GEOS)) % len(PHOTOS)]
                im2, boxes2 = gfn(im, boxes)
                im2 = pfn(im2)
                tag = f"{gname}{'+' + pname if pname else ''}"
                name = f"aug_c{cls}_{stem}_{tag}"
                im2.save(out_img / f"{name}.jpg", quality=95)
                write_label(out_lbl / f"{name}.txt", boxes2)
                new_files.append(name)
                made += 1
        print(f"  class {cls}: +{made} variants from {len(pool)} images", flush=True)

    # Final histogram over originals + augmented.
    final = collections.Counter(counts)
    for name in new_files:
        for line in (out_lbl / f"{name}.txt").read_text().splitlines():
            if line.strip():
                final[int(line.split()[0])] += 1
    print(f"after: {dict(sorted(final.items()))}", flush=True)
    print(f"wrote {len(new_files)} augmented images", flush=True)

    # Train list: originals (absolute) + augmented (absolute).
    with (ROOT / "data" / args.list_name).open("w") as fh:
        for stem, _ in sorted(items):
            fh.write(str((SRC_IMG / f"{stem}.jpg").resolve()) + "\n")
        for name in sorted(new_files):
            fh.write(str((out_img / f"{name}.jpg").resolve()) + "\n")
    print(f"wrote data/{args.list_name} "
          f"({len(items)} orig + {len(new_files)} aug)", flush=True)


if __name__ == "__main__":
    main()
