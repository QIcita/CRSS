#!/usr/bin/env python3
"""Audit a symlinked NREDD TLP spatial-group split before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import tifffile


SPLITS = ("train", "validation", "test")
FEATURE_LAYOUT = {
    "raw_data": ".swc",
    "image": ".tif",
    "gt": ".swc",
    "label": ".txt",
    "SP_feature": "",
    "point_surface_mc_v1": ".npz",
}
ORIGIN_PATTERN = re.compile(r"^x(-?\d+)_y(-?\d+)_z(-?\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_swc_xyz(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#", ndmin=2, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 5 or len(data) == 0:
        raise ValueError(f"Invalid SWC: {path}")
    return data[:, 2:5]


def parse_origin(stem: str) -> tuple[int, int, int]:
    match = ORIGIN_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Cannot parse origin: {stem}")
    return tuple(int(value) for value in match.groups())


def positive_overlap(left, right) -> bool:
    return all(
        max(left["origin"][axis], right["origin"][axis])
        < min(left["upper"][axis], right["upper"][axis])
        for axis in range(3)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("data/link_pred_data/spatial_grouped_v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split_root = args.split_root.resolve()
    manifest_path = split_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))

    errors = []
    warnings = []
    rows = []
    names_by_split = {}
    for split in SPLITS:
        expected_names = set(manifest["splits"][split]["filenames"])
        observed_names = {
            path.name for path in (split_root / split / "raw_data").glob("*.swc")
        }
        names_by_split[split] = observed_names
        if observed_names != expected_names:
            errors.append(
                f"{split} raw_data manifest mismatch: expected={len(expected_names)} "
                f"observed={len(observed_names)}"
            )
        for filename in sorted(expected_names):
            stem = Path(filename).stem
            feature_paths = {
                feature: split_root / split / feature / f"{stem}{suffix}"
                for feature, suffix in FEATURE_LAYOUT.items()
            }
            for feature, path in feature_paths.items():
                if not path.is_symlink():
                    errors.append(f"Not a symlink: {path}")
                elif not path.exists():
                    errors.append(f"Broken symlink: {path}")
            if errors:
                continue
            swc_xyz = load_swc_xyz(feature_paths["raw_data"])
            label = np.loadtxt(feature_paths["label"], ndmin=2)
            if label.shape != (len(swc_xyz), len(swc_xyz)):
                errors.append(
                    f"Label/SWC shape mismatch for {split}/{stem}: "
                    f"{label.shape} vs {len(swc_xyz)}"
                )
            with np.load(feature_paths["point_surface_mc_v1"], allow_pickle=False) as cache:
                cached_xyz = cache["node_xyz"].astype(np.float32)
                point_features = cache["point_features"]
                point_mask = cache["point_mask"]
            if cached_xyz.shape != swc_xyz.shape or not np.allclose(
                cached_xyz, swc_xyz, atol=1e-3
            ):
                errors.append(f"MC cache/SWC coordinate mismatch: {split}/{stem}")
            if point_features.shape[:2] != point_mask.shape or (
                point_features.shape[0] != len(swc_xyz)
                or point_features.shape[-1] != 26
            ):
                errors.append(f"MC cache shape mismatch: {split}/{stem}")
            spe_files = list(feature_paths["SP_feature"].glob("*_*.tif"))
            expected_spe_names = {f"{index}_.tif" for index in range(len(swc_xyz))}
            if {path.name for path in spe_files} != expected_spe_names:
                errors.append(f"SPE feature index mismatch: {split}/{stem}")
            with tifffile.TiffFile(feature_paths["image"]) as handle:
                shape_zyx = tuple(int(value) for value in handle.series[0].shape)
            if len(shape_zyx) != 3:
                errors.append(f"Non-3-D image: {split}/{stem}: {shape_zyx}")
            shape_xyz = tuple(reversed(shape_zyx))
            origin = parse_origin(stem)
            rows.append(
                {
                    "split": split,
                    "filename": filename,
                    "origin": origin,
                    "upper": tuple(a + b for a, b in zip(origin, shape_xyz)),
                    "nodes": len(swc_xyz),
                    "source_raw_data": str(feature_paths["raw_data"].resolve()),
                    "source_image": str(feature_paths["image"].resolve()),
                    "source_mc": str(feature_paths["point_surface_mc_v1"].resolve()),
                }
            )

    for left_index in range(len(rows)):
        for right_index in range(left_index + 1, len(rows)):
            left = rows[left_index]
            right = rows[right_index]
            if left["split"] != right["split"] and positive_overlap(left, right):
                errors.append(
                    f"Cross-split AABB overlap: {left['split']}/{left['filename']} "
                    f"vs {right['split']}/{right['filename']}"
                )
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            duplicate = sorted(names_by_split[left] & names_by_split[right])
            if duplicate:
                errors.append(f"Duplicate filenames in {left}/{right}: {duplicate}")

    result = {
        "schema_version": 1,
        "protocol": manifest["protocol"],
        "status": {
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "checks": {
            "expected_blocks": sum(
                manifest["splits"][split]["blocks"] for split in SPLITS
            ),
            "audited_blocks": len(rows),
            "cross_split_filename_duplicates": 0
            if not any("Duplicate filenames" in error for error in errors)
            else None,
            "cross_split_positive_volume_aabb_overlaps": 0
            if not any("Cross-split AABB" in error for error in errors)
            else None,
            "all_inputs_are_resolving_symlinks": not any(
                "symlink" in error for error in errors
            ),
            "all_label_shapes_match_swc": not any(
                "Label/SWC" in error for error in errors
            ),
            "all_spe_indices_match_swc": not any(
                "SPE feature" in error for error in errors
            ),
            "all_mc_cache_shapes_and_coordinates_match_swc": not any(
                "MC cache" in error for error in errors
            ),
        },
        "splits": {
            split: {
                "blocks": sum(row["split"] == split for row in rows),
                "nodes": sum(row["nodes"] for row in rows if row["split"] == split),
            }
            for split in SPLITS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
