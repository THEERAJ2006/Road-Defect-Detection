"""Extract the Chinese portions of RDD2022 and convert them to YOLO detect format.

Source: RDD2022_released_through_CRDDC2022.zip (the CRDDC'2022 release of RDD2022).
Only the China_Drone and China_MotorBike subsets are used, per Sun et al. (2024),
Electronics 13(12) 2413, which trains on a Chinese subset of RDD2022.

Two facts about the release drive the filtering:
  - only the train/ split ships annotations; China_MotorBike/test/ has 500 images
    with no labels at all, so it is unusable and is skipped
  - the Chinese labels use six names, but only five are RDD damage codes. "Block
    crack" appears 3 times across 2401 images and is not a code; images whose only
    label is "Block crack" are dropped rather than silently relabelled.

VOC XML -> YOLO detect ("cls xc yc w h", normalised 0-1). Split 7:2:1 as the paper
states, stratified so every split contains every class.
"""

import collections
import io
import pathlib
import random
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile

SRC = pathlib.Path("RDD2022_released_through_CRDDC2022.zip")
OUT = pathlib.Path("data/rdd2022-cn")
STAGE = pathlib.Path("/tmp/rdd2022_cn_stage")

# RDD damage codes, in the order used throughout the RDD literature.
CLASSES = ["D00", "D10", "D20", "D40", "Repair"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
DROPPED_NAMES = {"Block crack"}
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 0


def load_subset(outer, inner):
    with outer.open(inner) as fh:
        return zipfile.ZipFile(io.BytesIO(fh.read()))


def parse_objects(xml_bytes):
    """Return (width, height, [(class_name, xmin, ymin, xmax, ymax)]) for one image."""
    root = ET.fromstring(xml_bytes)
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        box = obj.find("bndbox")
        b = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        objs.append((name, *b))
    return w, h, objs


def to_yolo_lines(w, h, objs):
    """Convert one image's objects to YOLO lines, or None if it must be dropped."""
    lines, dropped = [], False
    for name, xmin, ymin, xmax, ymax in objs:
        if name in DROPPED_NAMES or name not in CLASS_ID:
            dropped = True
            continue
        xmin, ymin = max(0.0, xmin), max(0.0, ymin)
        xmax, ymax = min(float(w), xmax), min(float(h), ymax)
        if xmax <= xmin or ymax <= ymin:  # degenerate box: VOC XML sometimes has these
            dropped = True
            continue
        xc, yc = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        lines.append(f"{CLASS_ID[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    # An image is dropped entirely if anything about it was unlabellable, so the
    # label file always describes the whole image.
    return None if dropped or not lines else lines


def main():
    rng = random.Random(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    outer = zipfile.ZipFile(SRC)
    records = []          # (subset, stem, image_bytes, yolo_lines)
    stats = collections.Counter()

    for inner in ("RDD2022/China_Drone.zip", "RDD2022/China_MotorBike.zip"):
        zi = load_subset(outer, inner)
        subset = inner.split("/")[-1].removesuffix(".zip")
        anns = {n.rsplit("/", 1)[-1].removesuffix(".xml"): n for n in zi.namelist() if n.endswith(".xml")}
        imgs = {
            n.rsplit("/", 1)[-1].rsplit(".", 1)[0]: n
            for n in zi.namelist()
            if n.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        labelled = sorted(set(anns) & set(imgs))
        stats[f"{subset}/images_in_zip"] = len(imgs)
        stats[f"{subset}/labelled"] = len(labelled)
        print(f"{subset}: {len(imgs)} images, {len(anns)} xml, {len(labelled)} usable pairs")

        for stem in labelled:
            try:
                w, h, objs = parse_objects(zi.read(anns[stem]))
                lines = to_yolo_lines(w, h, objs)
            except (ET.ParseError, ValueError, AttributeError) as exc:
                stats[f"{subset}/unparseable"] += 1
                print(f"  skip {stem}: {exc}")
                continue
            if lines is None:
                stats[f"{subset}/dropped_label"] += 1
                continue
            stats[f"{subset}/kept"] += 1
            records.append((subset, stem, zi.read(imgs[stem]), lines))

    print(f"\nkept {len(records)} labelled images across {len(CLASSES)} classes")

    # Stratify by class presence so every split sees every class. Each class queue is
    # shuffled and cut 7:2:1; an image lands in the first split whose queue claims it,
    # and anything no queue claims goes to train.
    by_class = collections.defaultdict(list)
    for i, rec in enumerate(records):
        for cls in {int(line.split()[0]) for line in rec[3]}:
            by_class[cls].append(i)
    order = sorted(by_class, key=lambda c: len(by_class[c]))
    buckets = {"train": [], "val": [], "test": []}
    seen = set()
    for cls in order:
        queue = [i for i in by_class[cls] if i not in seen]
        rng.shuffle(queue)
        n = len(queue)
        n_tr = round(n * SPLIT["train"])
        n_va = round(n * SPLIT["val"])
        buckets["train"] += queue[:n_tr]
        buckets["val"] += queue[n_tr:n_tr + n_va]
        buckets["test"] += queue[n_tr + n_va:]
        seen.update(queue)
    leftovers = [i for i in range(len(records)) if i not in seen]
    rng.shuffle(leftovers)
    buckets["train"] += leftovers

    for split in SPLIT:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = collections.Counter()
    for split, idxs in buckets.items():
        for i in idxs:
            subset, stem, img, lines = records[i]
            (OUT / "images" / split / f"{stem}.jpg").write_bytes(img)
            (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            written[split] += 1
    print("\nsplit (7:2:1):")
    for s in ("train", "val", "test"):
        print(f"  {s:<6} {written[s]:>5} images")

    # verification pass: every image has a label, every box is inside the image
    print("\nverifying written data...")
    problems = 0
    per_split_classes = {}
    for split in SPLIT:
        imgs = sorted((OUT / "images" / split).glob("*.jpg"))
        labs = sorted((OUT / "labels" / split).glob("*.txt"))
        if len(imgs) != len(labs):
            print(f"  {split}: MISMATCH {len(imgs)} images vs {len(labs)} labels")
            problems += 1
        seen_cls = set()
        for lab in labs:
            for line in lab.read_text().split("\n"):
                if not line.strip():
                    continue
                c, xc, yc, w, h = line.split()
                seen_cls.add(int(c))
                xc, yc, w, h = map(float, (xc, yc, w, h))
                # Tolerance is a thousandth of a pixel at 512x512 (1.95e-3). The
                # %.6f formatting rounds, so an exact-in-[0,1] test would false-alarm
                # on ~1000 boxes that are in fact correct to within 5e-7.
                tol = 1e-5
                if not (
                    xc - w / 2 >= -tol
                    and xc + w / 2 <= 1 + tol
                    and yc - h / 2 >= -tol
                    and yc + h / 2 <= 1 + tol
                ):
                    problems += 1
        per_split_classes[split] = seen_cls
        print(f"  {split}: {len(imgs)} images, classes present {sorted(seen_cls)}")
    missing = [s for s, c in per_split_classes.items() if c != set(range(len(CLASSES)))]
    if missing:
        print(f"  WARNING: splits missing some classes: {missing}")

    yaml_lines = [
        "# Chinese subset of RDD2022 (China_Drone + China_MotorBike), prepared for",
        "# Sun et al. (2024) Electronics 13(12) 2413. VOC XML converted to YOLO detect",
        "# format; split 7:2:1 as the paper states; see scripts/prepare_rdd2022_cn.py.",
        "#",
        "# Class ids follow the RDD damage-code convention. 'Block crack' (3 objects",
        "# in China_Drone) is not an RDD code and the images carrying only that label",
        "# were dropped rather than relabelled.",
        # No `path:` key on purpose. A relative path resolves against ultralytics'
        # datasets_dir, and an absolute one would break portability; omitting it makes
        # ultralytics resolve the dataset root to this yaml's own directory (data/).
        "",
        "train: rdd2022-cn/images/train",
        "val: rdd2022-cn/images/val",
        "test: rdd2022-cn/images/test",
        "",
        "names:",
    ]
    for name in CLASSES:
        yaml_lines.append(f"  - {name}")
    (pathlib.Path("data/rdd2022-cn.yaml")).write_text("\n".join(yaml_lines) + "\n")
    print("\nwrote data/rdd2022-cn.yaml")
    print("problems:", problems)


if __name__ == "__main__":
    main()