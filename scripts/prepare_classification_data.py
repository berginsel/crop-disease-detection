"""
Prepare PlantVillage leaf disease classification data for training.

Input (expected):
  data/raw/leaf_disease_classification/images/<class_name>/*.(jpg|png|jpeg|webp)

Output:
  data/processed/classification_tfds/
    - train.tfrecord
    - val.tfrecord
    - test.tfrecord
    - labels.txt
    - dataset_info.json

What this script does:
- Resizes images to 224x224
- Normalizes pixel values to [0, 1] (float32)
- Applies data augmentation to the TRAIN split (configurable)
- Splits by class into train/val/test = 70/15/15 (reproducible)
- Writes TFRecords for efficient training with tf.data

Usage:
  python scripts/prepare_classification_data.py
  python scripts/prepare_classification_data.py --augment-copies 1
  python scripts/prepare_classification_data.py --limit-per-class 500
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


def _list_classes(images_root: Path) -> List[str]:
    classes = [p.name for p in images_root.iterdir() if p.is_dir()]
    classes.sort()
    return classes


def _gather_by_class(images_root: Path, classes: Sequence[str]) -> Dict[str, List[Path]]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out: Dict[str, List[Path]] = {}
    for c in classes:
        files = [p for p in (images_root / c).rglob("*") if p.is_file() and p.suffix.lower() in exts]
        files.sort()
        out[c] = files
    return out


def _split_indices(n: int, *, train: float, val: float, test: float) -> Tuple[slice, slice, slice]:
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    # Ensure all items accounted for (avoid rounding loss).
    n_test = max(0, n - n_train - n_val)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_train + n_val + n_test)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1337, help="Random seed for deterministic split/shuffle.")
    p.add_argument("--image-size", type=int, default=224, help="Resize to square size (default: 224).")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument(
        "--augment-copies",
        type=int,
        default=1,
        help="Number of augmented copies to write per TRAIN image. 0 disables augmentation. Default: 1.",
    )
    p.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap per class before splitting (useful for quick experiments).",
    )
    return p.parse_args()


def _bytes_feature(v: bytes):
    import tensorflow as tf

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int64_feature(v: int):
    import tensorflow as tf

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def _float_list_feature(vs: Sequence[float]):
    import tensorflow as tf

    return tf.train.Feature(float_list=tf.train.FloatList(value=list(map(float, vs))))


def _serialize_example(*, image_floats: List[float], label: int, height: int, width: int, channels: int) -> bytes:
    import tensorflow as tf

    features = {
        "image/height": _int64_feature(height),
        "image/width": _int64_feature(width),
        "image/channels": _int64_feature(channels),
        "image/label": _int64_feature(label),
        # Store normalized float32 values directly (HWC flattened).
        "image/float": tf.train.Feature(float_list=tf.train.FloatList(value=image_floats)),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=features))
    return ex.SerializeToString()


def _augment_image(img):
    """
    Lightweight augmentations that work without extra deps.
    Expects float32 [0,1], returns float32 [0,1].
    """
    import tensorflow as tf

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.90, upper=1.10)
    # Small random crop/zoom then resize back.
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    crop_frac = tf.random.uniform([], 0.85, 1.0)
    crop_h = tf.cast(tf.cast(h, tf.float32) * crop_frac, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * crop_frac, tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3])
    img = tf.image.resize(img, [h, w], method="bilinear")
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def _load_preprocess(path: Path, *, size: int):
    import tensorflow as tf

    data = tf.io.read_file(str(path))
    # decode_image supports jpeg/png/webp, sets shape dynamically
    img = tf.image.decode_image(data, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [size, size], method="bilinear")
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def _write_tfrecord(
    out_path: Path,
    items: Sequence[Tuple[Path, int]],
    *,
    size: int,
    augment_copies: int,
    seed: int,
) -> int:
    import tensorflow as tf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    # Ensure deterministic augmentation sequence.
    tf.random.set_seed(seed)

    with tf.io.TFRecordWriter(str(out_path)) as w:
        for img_path, label in items:
            img = _load_preprocess(img_path, size=size)  # float32 [0,1]

            def emit(img_tensor):
                nonlocal count
                flat = tf.reshape(img_tensor, [-1]).numpy().astype("float32").tolist()
                ex = _serialize_example(image_floats=flat, label=label, height=size, width=size, channels=3)
                w.write(ex)
                count += 1

            # Original
            emit(img)

            # Augmented copies (only if requested)
            for _ in range(max(0, augment_copies)):
                emit(_augment_image(img))

    return count


def main() -> int:
    args = parse_args()
    root = _project_root()

    in_root = root / "data" / "raw" / "leaf_disease_classification" / "images"
    out_root = root / "data" / "processed" / "classification_tfds"
    out_root.mkdir(parents=True, exist_ok=True)

    if not in_root.exists():
        raise FileNotFoundError(f"Input folder not found: {in_root}")

    classes = _list_classes(in_root)
    if not classes:
        raise RuntimeError(f"No class subfolders found under: {in_root}")

    (out_root / "labels.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    by_class = _gather_by_class(in_root, classes)
    rng = random.Random(args.seed)

    train_items: List[Tuple[Path, int]] = []
    val_items: List[Tuple[Path, int]] = []
    test_items: List[Tuple[Path, int]] = []

    per_class_split: Dict[str, SplitCounts] = {}

    for label_idx, cls in enumerate(classes):
        items = list(by_class[cls])
        if args.limit_per_class is not None:
            items = items[: args.limit_per_class]
        rng.shuffle(items)

        s_train, s_val, s_test = _split_indices(
            len(items), train=args.train_frac, val=args.val_frac, test=args.test_frac
        )
        train_files = items[s_train]
        val_files = items[s_val]
        test_files = items[s_test]

        train_items.extend((p, label_idx) for p in train_files)
        val_items.extend((p, label_idx) for p in val_files)
        test_items.extend((p, label_idx) for p in test_files)

        per_class_split[cls] = SplitCounts(train=len(train_files), val=len(val_files), test=len(test_files))

    # Shuffle globally so batches are mixed across classes.
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    print(f"[classification] Classes: {len(classes)}")
    print(f"[classification] Train/Val/Test items (pre-augmentation): {len(train_items)}/{len(val_items)}/{len(test_items)}")

    train_written = _write_tfrecord(
        out_root / "train.tfrecord",
        train_items,
        size=args.image_size,
        augment_copies=args.augment_copies,
        seed=args.seed,
    )
    val_written = _write_tfrecord(
        out_root / "val.tfrecord",
        val_items,
        size=args.image_size,
        augment_copies=0,
        seed=args.seed,
    )
    test_written = _write_tfrecord(
        out_root / "test.tfrecord",
        test_items,
        size=args.image_size,
        augment_copies=0,
        seed=args.seed,
    )

    info = {
        "input_dir": str(in_root),
        "output_dir": str(out_root),
        "image_size": args.image_size,
        "normalize": "[0,1] float32",
        "splits": {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac},
        "augment_copies_per_train_image": args.augment_copies,
        "counts_written": {"train": train_written, "val": val_written, "test": test_written},
        "per_class_split": {k: asdict(v) for k, v in per_class_split.items()},
        "labels": classes,
    }
    (out_root / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"[classification] Wrote TFRecords under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

