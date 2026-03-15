"""
Prepare pest detection data: convert YOLO annotations to TFRecord.

Expected input layout:
  data/raw/pest_detection/
    images/
      *.jpg|png|jpeg|webp
    annotations/
      <image_stem>.txt         # YOLO format: class x_center y_center width height (normalized)
      classes.txt              # optional, one class name per line

Output:
  data/processed/detection_tfrecords/
    - train.tfrecord
    - val.tfrecord
    - test.tfrecord
    - label_map.txt
    - dataset_info.json

What this script does:
- Reads YOLO annotations (normalized)
- Resizes images to 320x320 and encodes as JPEG
- Splits into train/val/test (70/15/15, reproducible)
- Writes TFRecords compatible with common TF object detection pipelines

Usage:
  python scripts/prepare_detection_data.py
  python scripts/prepare_detection_data.py --image-size 320 --min-box-area 0.0001
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SplitCounts:
    train: int
    val: int
    test: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--image-size", type=int, default=320, help="Resize to square size (default: 320).")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument(
        "--min-box-area",
        type=float,
        default=0.0,
        help="Filter boxes with normalized area smaller than this threshold.",
    )
    p.add_argument(
        "--skip-unlabeled",
        action="store_true",
        help="If set, skip images without a matching annotation txt file.",
    )
    return p.parse_args()


def _split_indices(n: int, *, train: float, val: float, test: float) -> Tuple[slice, slice, slice]:
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_test = max(0, n - n_train - n_val)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_train + n_val + n_test)


def _read_label_map(annotations_dir: Path) -> List[str]:
    classes_path = annotations_dir / "classes.txt"
    if not classes_path.exists():
        # Default single-class detector.
        return ["pest"]
    lines = [ln.strip() for ln in classes_path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    return lines or ["pest"]


def _parse_yolo_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Returns list of (class_id, xc, yc, w, h) all normalized [0,1].
    """
    out = []
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return out
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        out.append((cls, xc, yc, w, h))
    return out


def _yolo_to_corners(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    xmin = xc - w / 2.0
    ymin = yc - h / 2.0
    xmax = xc + w / 2.0
    ymax = yc + h / 2.0
    # Clip to [0,1]
    xmin = min(max(xmin, 0.0), 1.0)
    ymin = min(max(ymin, 0.0), 1.0)
    xmax = min(max(xmax, 0.0), 1.0)
    ymax = min(max(ymax, 0.0), 1.0)
    return xmin, ymin, xmax, ymax


def _bytes_feature(v: bytes):
    import tensorflow as tf

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _bytes_list_feature(vs: Sequence[bytes]):
    import tensorflow as tf

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(vs)))


def _int64_feature(v: int):
    import tensorflow as tf

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def _int64_list_feature(vs: Sequence[int]):
    import tensorflow as tf

    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(map(int, vs))))


def _float_list_feature(vs: Sequence[float]):
    import tensorflow as tf

    return tf.train.Feature(float_list=tf.train.FloatList(value=list(map(float, vs))))


def _encode_resized_jpeg(image_path: Path, *, size: int) -> Tuple[bytes, int, int]:
    """
    Load image, resize to size x size, encode as JPEG.
    Returns (jpeg_bytes, height, width).
    """
    import tensorflow as tf

    raw = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [size, size], method="bilinear")
    img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)
    encoded = tf.io.encode_jpeg(img, quality=95).numpy()
    return encoded, size, size


def _serialize_detection_example(
    *,
    encoded_jpeg: bytes,
    filename: str,
    height: int,
    width: int,
    xmins: Sequence[float],
    xmaxs: Sequence[float],
    ymins: Sequence[float],
    ymaxs: Sequence[float],
    classes_text: Sequence[str],
    classes: Sequence[int],
) -> bytes:
    import tensorflow as tf

    features = {
        "image/encoded": _bytes_feature(encoded_jpeg),
        "image/format": _bytes_feature(b"jpeg"),
        "image/filename": _bytes_feature(filename.encode("utf-8")),
        "image/source_id": _bytes_feature(filename.encode("utf-8")),
        "image/height": _int64_feature(height),
        "image/width": _int64_feature(width),
        "image/object/bbox/xmin": _float_list_feature(xmins),
        "image/object/bbox/xmax": _float_list_feature(xmaxs),
        "image/object/bbox/ymin": _float_list_feature(ymins),
        "image/object/bbox/ymax": _float_list_feature(ymaxs),
        "image/object/class/text": _bytes_list_feature([t.encode("utf-8") for t in classes_text]),
        "image/object/class/label": _int64_list_feature(classes),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=features))
    return ex.SerializeToString()


def main() -> int:
    args = parse_args()
    root = _project_root()

    in_images = root / "data" / "raw" / "pest_detection" / "images"
    in_ann = root / "data" / "raw" / "pest_detection" / "annotations"
    out_root = root / "data" / "processed" / "detection_tfrecords"
    out_root.mkdir(parents=True, exist_ok=True)

    if not in_images.exists():
        raise FileNotFoundError(f"Input images folder not found: {in_images}")
    if not in_ann.exists():
        raise FileNotFoundError(f"Input annotations folder not found: {in_ann}")

    label_names = _read_label_map(in_ann)
    (out_root / "label_map.txt").write_text("\n".join(label_names) + "\n", encoding="utf-8")

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in in_images.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    images.sort()
    if not images:
        raise RuntimeError(f"No images found under: {in_images}")

    # Filter images depending on annotation availability.
    labeled_pairs: List[Tuple[Path, Optional[Path]]] = []
    for img in images:
        ann_path = in_ann / f"{img.stem}.txt"
        if ann_path.exists():
            labeled_pairs.append((img, ann_path))
        else:
            if args.skip_unlabeled:
                continue
            labeled_pairs.append((img, None))

    rng = random.Random(args.seed)
    rng.shuffle(labeled_pairs)

    s_train, s_val, s_test = _split_indices(
        len(labeled_pairs), train=args.train_frac, val=args.val_frac, test=args.test_frac
    )
    splits = {
        "train": labeled_pairs[s_train],
        "val": labeled_pairs[s_val],
        "test": labeled_pairs[s_test],
    }

    import tensorflow as tf

    def write_split(name: str, pairs: Sequence[Tuple[Path, Optional[Path]]]) -> int:
        out_path = out_root / f"{name}.tfrecord"
        n_written = 0
        with tf.io.TFRecordWriter(str(out_path)) as w:
            for img_path, ann_path in pairs:
                if ann_path is None:
                    # No labels; skip because detection training needs boxes.
                    continue

                yolo_boxes = _parse_yolo_file(ann_path)
                if not yolo_boxes:
                    continue

                xmins: List[float] = []
                xmaxs: List[float] = []
                ymins: List[float] = []
                ymaxs: List[float] = []
                classes: List[int] = []
                classes_text: List[str] = []

                for cls_id, xc, yc, bw, bh in yolo_boxes:
                    area = float(bw * bh)
                    if area < args.min_box_area:
                        continue

                    xmin, ymin, xmax, ymax = _yolo_to_corners(xc, yc, bw, bh)
                    if xmax <= xmin or ymax <= ymin:
                        continue

                    # TF Object Detection API typically expects 1-based class labels.
                    # If your YOLO classes are already 1-based, set classes.txt and adjust here.
                    tf_label = int(cls_id) + 1
                    name_txt = label_names[cls_id] if 0 <= cls_id < len(label_names) else "pest"

                    xmins.append(xmin)
                    xmaxs.append(xmax)
                    ymins.append(ymin)
                    ymaxs.append(ymax)
                    classes.append(tf_label)
                    classes_text.append(name_txt)

                if not classes:
                    continue

                encoded, height, width = _encode_resized_jpeg(img_path, size=args.image_size)
                ex = _serialize_detection_example(
                    encoded_jpeg=encoded,
                    filename=img_path.name,
                    height=height,
                    width=width,
                    xmins=xmins,
                    xmaxs=xmaxs,
                    ymins=ymins,
                    ymaxs=ymaxs,
                    classes_text=classes_text,
                    classes=classes,
                )
                w.write(ex)
                n_written += 1
        return n_written

    counts_written = {k: write_split(k, v) for k, v in splits.items()}

    per_split_counts = {k: len(v) for k, v in splits.items()}
    info = {
        "input_images_dir": str(in_images),
        "input_annotations_dir": str(in_ann),
        "output_dir": str(out_root),
        "image_size": args.image_size,
        "splits": {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac},
        "label_map": label_names,
        "pairs_total": len(labeled_pairs),
        "pairs_per_split": per_split_counts,
        "records_written": counts_written,
        "notes": "Only images with annotation txt files are written to TFRecords.",
    }
    (out_root / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"[detection] Images discovered: {len(images)}")
    print(f"[detection] Pairs used (after skip rules): {len(labeled_pairs)}")
    print(f"[detection] TFRecords written under: {out_root}")
    print(f"[detection] Records written: {counts_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

