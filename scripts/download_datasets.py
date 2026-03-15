"""
Download datasets for the offline crop disease detection project.

This script downloads:
1) PlantVillage (leaf disease classification) via TensorFlow Datasets (TFDS)
2) A small open insect/pest detection dataset (default: Yellow Sticky Traps Dataset from GitHub)

Outputs:
- data/raw/leaf_disease_classification/images/<class_name>/*.jpg|png
- data/raw/pest_detection/images/*.jpg|png
- data/raw/pest_detection/annotations/*.txt   (YOLO-format bboxes when available)

Usage:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --max-pest-images 2000
  python scripts/download_datasets.py --skip-pests

Notes:
- Some large pest datasets (e.g., IP102, LeAF Pest Detection) may require gated access or
  manual download. This script uses a freely accessible GitHub dataset by default.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Tuple


def _project_root() -> Path:
    # scripts/ -> project root
    return Path(__file__).resolve().parents[1]


def _ensure_empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_files(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def download_plantvillage_tfds(out_images_dir: Path, *, limit_per_class: Optional[int], seed: int) -> None:
    """
    Export TFDS 'plant_village' images into an ImageFolder-style directory structure.

    This keeps downstream scripts simple and avoids coupling preprocessing to TFDS APIs.
    """
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies. Install TensorFlow and TFDS:\n"
            "  pip install tensorflow tensorflow-datasets"
        ) from e

    out_images_dir.mkdir(parents=True, exist_ok=True)

    # TFDS naming has historically been 'plant_village'. If it changes, we fail loudly.
    builder = tfds.builder("plant_village")
    builder.download_and_prepare()

    # Prefer a deterministic export order for reproducibility.
    rng = random.Random(seed)

    label_names = builder.info.features["label"].names
    per_class_counts = {name: 0 for name in label_names}

    def export_split(split_name: str) -> None:
        nonlocal per_class_counts
        ds = tfds.load("plant_village", split=split_name, shuffle_files=False)

        # Shuffling per example ensures we get a representative subset when limit_per_class is set.
        ds = ds.shuffle(50_000, seed=seed, reshuffle_each_iteration=False)

        for ex in tfds.as_numpy(ds):
            label_idx = int(ex["label"])
            label_name = label_names[label_idx]

            if limit_per_class is not None and per_class_counts[label_name] >= limit_per_class:
                continue

            img = ex["image"]  # HWC uint8
            # Preserve original encoding by re-encoding to JPEG (stable, smaller).
            img_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
            encoded = tf.io.encode_jpeg(img_tensor, quality=95)

            class_dir = out_images_dir / label_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Use a simple deterministic-ish filename.
            idx = per_class_counts[label_name]
            out_path = class_dir / f"{split_name.lower()}_{idx:06d}.jpg"
            tf.io.write_file(str(out_path), encoded)

            per_class_counts[label_name] += 1

    # Many TFDS datasets use "train" only; if additional splits exist, export them too.
    splits = list(builder.info.splits.keys())
    if not splits:
        splits = ["train"]

    for split in splits:
        export_split(split)


def _download_url_to_file(url: str, dst: Path) -> None:
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def download_yellow_sticky_traps_dataset(
    out_images_dir: Path,
    out_annotations_dir: Path,
    *,
    max_images: Optional[int],
    seed: int,
) -> None:
    """
    Download and extract the Yellow Sticky Traps dataset (GitHub).

    Repo: https://github.com/md-121/yellow-sticky-traps-dataset
    License: CC0-1.0 (per repository)

    The repo layout may evolve; this function tries to locate common image/label folders.
    """
    # Use the default branch zip archive (works without git).
    url = "https://github.com/md-121/yellow-sticky-traps-dataset/archive/refs/heads/master.zip"

    rng = random.Random(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        zip_path = tmpdir_p / "sticky_traps.zip"
        _download_url_to_file(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir_p)

        # Find extracted repo root.
        extracted_roots = [p for p in tmpdir_p.iterdir() if p.is_dir() and "yellow-sticky-traps-dataset" in p.name]
        if not extracted_roots:
            raise RuntimeError("Could not locate extracted sticky-traps dataset directory.")
        repo_root = extracted_roots[0]

        # Heuristics to find images and labels.
        # Common patterns in CV repos: images/, imgs/, data/images; labels/, annotations/
        candidate_image_dirs = [
            repo_root / "images",
            repo_root / "imgs",
            repo_root / "data" / "images",
            repo_root / "dataset" / "images",
        ]
        candidate_label_dirs = [
            repo_root / "labels",
            repo_root / "annotations",
            repo_root / "data" / "labels",
            repo_root / "dataset" / "labels",
        ]

        image_dir = next((d for d in candidate_image_dirs if d.exists()), None)
        label_dir = next((d for d in candidate_label_dirs if d.exists()), None)

        if image_dir is None:
            # Fall back: search for common image extensions anywhere.
            all_imgs = list(_iter_files(repo_root, (".jpg", ".jpeg", ".png", ".webp")))
            if not all_imgs:
                raise RuntimeError("No images found in the sticky-traps dataset archive.")
            # Choose their parent as "image_dir" to keep relative structure minimal.
            image_dir = repo_root
        else:
            all_imgs = list(_iter_files(image_dir, (".jpg", ".jpeg", ".png", ".webp")))

        # If labels are not found, we still export images (detection prep can treat as unlabeled / skip).
        all_lbls = []
        if label_dir is not None:
            all_lbls = list(_iter_files(label_dir, (".txt",)))

        # Subsample if requested.
        rng.shuffle(all_imgs)
        if max_images is not None:
            all_imgs = all_imgs[: max_images]

        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_annotations_dir.mkdir(parents=True, exist_ok=True)

        # Copy images and matching YOLO label files when present.
        # Match by stem (filename without extension).
        lbl_by_stem = {p.stem: p for p in all_lbls}
        copied = 0
        for img in all_imgs:
            dst_img = out_images_dir / img.name
            shutil.copy2(img, dst_img)

            lbl = lbl_by_stem.get(img.stem)
            if lbl is not None and lbl.exists():
                shutil.copy2(lbl, out_annotations_dir / lbl.name)

            copied += 1

        # Write a minimal classes file for downstream conversion if the dataset doesn't provide one.
        classes_txt = out_annotations_dir / "classes.txt"
        if not classes_txt.exists():
            classes_txt.write_text("pest\n", encoding="utf-8")

        print(f"[pests] Exported {copied} images to: {out_images_dir}")
        if any(out_annotations_dir.glob("*.txt")):
            print(f"[pests] Exported annotations to: {out_annotations_dir}")
        else:
            print("[pests] No annotation txt files found; detection TFRecord creation may skip all images.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1337, help="Random seed for deterministic sampling.")

    p.add_argument(
        "--plantvillage-limit-per-class",
        type=int,
        default=None,
        help="Optional: cap examples per class to keep the export small (e.g., 500).",
    )

    p.add_argument("--skip-plantvillage", action="store_true", help="Skip PlantVillage download/export.")
    p.add_argument("--skip-pests", action="store_true", help="Skip pest dataset download/export.")
    p.add_argument(
        "--max-pest-images",
        type=int,
        default=3000,
        help="Limit pest images copied into raw folder (keeps dataset small).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = _project_root()

    plant_out = root / "data" / "raw" / "leaf_disease_classification" / "images"
    pests_images_out = root / "data" / "raw" / "pest_detection" / "images"
    pests_ann_out = root / "data" / "raw" / "pest_detection" / "annotations"

    plant_out.mkdir(parents=True, exist_ok=True)
    pests_images_out.mkdir(parents=True, exist_ok=True)
    pests_ann_out.mkdir(parents=True, exist_ok=True)

    if not args.skip_plantvillage:
        print("[plantvillage] Downloading/exporting via TFDS (this may take a while)...")
        download_plantvillage_tfds(
            plant_out,
            limit_per_class=args.plantvillage_limit_per_class,
            seed=args.seed,
        )
        print(f"[plantvillage] Done. Images saved under: {plant_out}")
    else:
        print("[plantvillage] Skipped.")

    if not args.skip_pests:
        print("[pests] Downloading/exporting a small open detection dataset...")
        download_yellow_sticky_traps_dataset(
            pests_images_out,
            pests_ann_out,
            max_images=args.max_pest_images,
            seed=args.seed,
        )
    else:
        print("[pests] Skipped.")

    print("All downloads completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

