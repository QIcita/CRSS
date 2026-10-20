#!/usr/bin/env python3
"""Audit whether NREDD TLP spatial overlap originates in official archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zlib

import numpy as np
import tifffile


ORIGIN_PATTERN = re.compile(r"^x(-?\d+)_y(-?\d+)_z(-?\d+)$")
SOURCE_SPLITS = ("train", "test")


def command_result(command, cwd):
    result = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def parse_origin(stem):
    match = ORIGIN_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Cannot parse origin: {stem}")
    return tuple(int(value) for value in match.groups())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crc32(path):
    value = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return f"{value & 0xFFFFFFFF:08X}"


def parse_7z_listing(archive):
    result = subprocess.run(
        ["7z", "l", "-slt", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    )
    records = []
    current = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            current[key] = value
    if current:
        records.append(current)
    return {record["Path"]: record for record in records if "Path" in record}


def collect_blocks(data_root):
    blocks = []
    for split in SOURCE_SPLITS:
        for path in sorted((data_root / split / "image").glob("*.tif")):
            with tifffile.TiffFile(path) as handle:
                shape_zyx = tuple(int(value) for value in handle.series[0].shape)
            shape_xyz = tuple(reversed(shape_zyx))
            origin = parse_origin(path.stem)
            blocks.append(
                {
                    "split": split,
                    "stem": path.stem,
                    "path": path,
                    "origin": origin,
                    "shape_xyz": shape_xyz,
                    "upper": tuple(a + b for a, b in zip(origin, shape_xyz)),
                }
            )
    return blocks


def overlap_box(left, right):
    lower = tuple(max(a, b) for a, b in zip(left["origin"], right["origin"]))
    upper = tuple(min(a, b) for a, b in zip(left["upper"], right["upper"]))
    return lower, upper


def has_positive_volume(lower, upper):
    return all(a < b for a, b in zip(lower, upper))


def xyz_box_to_zyx_slices(block, lower, upper):
    return tuple(
        slice(lower[axis] - block["origin"][axis], upper[axis] - block["origin"][axis])
        for axis in (2, 1, 0)
    )


def coordinate_key(value):
    return tuple(int(round(float(item) * 1000.0)) for item in value)


def missing_edge_signatures(project_root, data_root, split):
    sys.path.insert(0, str(project_root))
    from dataset.utils import get_edge_for_coonect

    signatures = set()
    nodes = set()
    edge_count = 0
    for swc_path in sorted((data_root / split / "raw_data").glob("*.swc")):
        swc = np.loadtxt(swc_path, comments="#", ndmin=2).astype(np.float32)
        label_path = data_root / split / "label" / f"{swc_path.stem}.txt"
        label = np.loadtxt(label_path, ndmin=2) > 0.5
        observed = get_edge_for_coonect(swc) > 0
        missing = np.triu(label & ~observed, k=1)
        origin = np.asarray(parse_origin(swc_path.stem), dtype=np.float64)
        global_xyz = swc[:, 2:5].astype(np.float64) + origin
        node_keys = [coordinate_key(value) for value in global_xyz]
        nodes.update(node_keys)
        for left, right in np.argwhere(missing):
            signatures.add(tuple(sorted((node_keys[int(left)], node_keys[int(right)]))))
            edge_count += 1
    return signatures, nodes, edge_count


def hash_duplicates(data_root, feature, suffix):
    hashes = {}
    per_split = {}
    for split in SOURCE_SPLITS:
        paths = sorted((data_root / split / feature).glob(f"*{suffix}"))
        per_split[split] = {path.name: sha256(path) for path in paths}
        for name, digest in per_split[split].items():
            hashes.setdefault(digest, []).append((split, name))
    cross = []
    for digest, members in hashes.items():
        member_splits = {split for split, _ in members}
        if len(member_splits) > 1:
            cross.append({"sha256": digest, "members": members})
    return per_split, cross


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    data_root = project_root / "data/link_pred_data"

    git_head = command_result(["git", "rev-parse", "HEAD"], project_root)
    archive_diff = command_result(
        [
            "git",
            "diff",
            "--exit-code",
            "--",
            "data/link_pred_data/train.7z",
            "data/link_pred_data/test.7z",
        ],
        project_root,
    )
    lfs_fsck = command_result(["git", "lfs", "fsck"], project_root)
    lfs_files = command_result(
        [
            "git",
            "lfs",
            "ls-files",
            "-l",
            "data/link_pred_data/train.7z",
            "data/link_pred_data/test.7z",
        ],
        project_root,
    )

    archive_records = {}
    mc_entries = {}
    for split in SOURCE_SPLITS:
        archive = data_root / f"{split}.7z"
        archive_records[split] = parse_7z_listing(archive)
        mc_entries[split] = sorted(
            path for path in archive_records[split] if "point_surface_mc_v1" in path
        )

    blocks = collect_blocks(data_root)
    train_blocks = [block for block in blocks if block["split"] == "train"]
    test_blocks = [block for block in blocks if block["split"] == "test"]
    overlap_pairs = []
    image_cache = {}
    covered_test_stems = set()
    coverage_masks = {}
    for test_block in test_blocks:
        for train_block in train_blocks:
            lower, upper = overlap_box(train_block, test_block)
            if not has_positive_volume(lower, upper):
                continue
            for block in (train_block, test_block):
                if block["stem"] not in image_cache:
                    image_cache[block["stem"]] = tifffile.imread(block["path"])
            train_values = image_cache[train_block["stem"]][
                xyz_box_to_zyx_slices(train_block, lower, upper)
            ]
            test_values = image_cache[test_block["stem"]][
                xyz_box_to_zyx_slices(test_block, lower, upper)
            ]
            identical = bool(np.array_equal(train_values, test_values))
            if train_values.size and np.std(train_values) and np.std(test_values):
                correlation = float(
                    np.corrcoef(train_values.ravel(), test_values.ravel())[0, 1]
                )
            else:
                correlation = 1.0 if identical else None
            difference = np.abs(
                train_values.astype(np.int64) - test_values.astype(np.int64)
            )
            overlap_pairs.append(
                {
                    "train": train_block["path"].name,
                    "test": test_block["path"].name,
                    "lower_xyz": lower,
                    "upper_xyz": upper,
                    "voxel_count": int(train_values.size),
                    "identical": identical,
                    "correlation": correlation,
                    "maximum_absolute_difference": int(difference.max(initial=0)),
                }
            )
            covered_test_stems.add(test_block["stem"])
            mask = coverage_masks.setdefault(
                test_block["stem"], np.zeros(test_block["shape_xyz"][::-1], dtype=bool)
            )
            mask[xyz_box_to_zyx_slices(test_block, lower, upper)] = True

    coverage = {
        f"{stem}.tif": float(mask.mean())
        for stem, mask in sorted(coverage_masks.items())
    }
    overlapping_paths = {
        (split, name)
        for row in overlap_pairs
        for split, name in (("train", row["train"]), ("test", row["test"]))
    }
    crc_checks = []
    for split, name in sorted(overlapping_paths):
        extracted = data_root / split / "image" / name
        archive_path = f"{split}/image/{name}"
        record = archive_records[split].get(archive_path)
        crc_checks.append(
            {
                "split": split,
                "filename": name,
                "archive_path": archive_path,
                "archive_size": int(record["Size"]) if record and record.get("Size") else None,
                "extracted_size": extracted.stat().st_size,
                "archive_crc32": record.get("CRC") if record else None,
                "extracted_crc32": crc32(extracted),
                "matches": bool(
                    record
                    and int(record["Size"]) == extracted.stat().st_size
                    and record.get("CRC") == crc32(extracted)
                ),
            }
        )

    duplicate_hashes = {}
    for feature, suffix in (
        ("raw_data", ".swc"),
        ("image", ".tif"),
        ("gt", ".swc"),
        ("label", ".txt"),
    ):
        _, duplicates = hash_duplicates(data_root, feature, suffix)
        duplicate_hashes[feature] = duplicates

    train_edges, train_nodes, train_edge_count = missing_edge_signatures(
        project_root, data_root, "train"
    )
    test_edges, test_nodes, test_edge_count = missing_edge_signatures(
        project_root, data_root, "test"
    )
    checks = {
        "official_archives_match_git_checkout": archive_diff["returncode"] == 0,
        "git_lfs_fsck_passed": lfs_fsck["returncode"] == 0,
        "mc_cache_absent_from_official_archives": not any(mc_entries.values()),
        "all_overlapping_extracted_tiffs_match_archive_crc_and_size": all(
            row["matches"] for row in crc_checks
        ),
        "all_overlap_voxels_identical": all(
            row["identical"] for row in overlap_pairs
        ),
        "cross_split_complete_file_hash_duplicates_are_zero": all(
            not rows for rows in duplicate_hashes.values()
        ),
        "cross_split_global_missing_edge_duplicates_are_zero": not (
            train_edges & test_edges
        ),
    }
    result = {
        "schema_version": 1,
        "audit": "NREDD official TLP spatial leakage provenance",
        "status": {"passed": all(checks.values())},
        "source": {
            "repository": "https://github.com/mohen-star/NREDD.git",
            "git_head": git_head,
            "archive_diff": archive_diff,
            "git_lfs_fsck": lfs_fsck,
            "git_lfs_files": lfs_files,
        },
        "checks": checks,
        "official_split_counts": {
            "train": len(train_blocks),
            "test": len(test_blocks),
        },
        "spatial_overlap": {
            "train_test_pairs": len(overlap_pairs),
            "unique_train_blocks": len({row["train"] for row in overlap_pairs}),
            "unique_test_blocks": len({row["test"] for row in overlap_pairs}),
            "test_block_coverage_fraction": coverage,
            "pairs": overlap_pairs,
        },
        "archive_provenance": {
            "mc_entries": mc_entries,
            "overlapping_tiff_crc_checks": crc_checks,
        },
        "direct_duplicate_audit": {
            feature: {
                "cross_split_duplicate_hash_groups": len(rows),
                "groups": rows,
            }
            for feature, rows in duplicate_hashes.items()
        },
        "global_graph_audit": {
            "train_missing_edges": train_edge_count,
            "test_missing_edges": test_edge_count,
            "cross_split_duplicate_missing_edges": len(train_edges & test_edges),
            "cross_split_duplicate_node_coordinates": len(train_nodes & test_nodes),
            "coordinate_quantization": "0.001 voxel after adding filename XYZ origin",
        },
        "interpretation": (
            "The spatial overlap is present in the official train/test archives. "
            "The MC cache is absent from those archives and therefore cannot have "
            "created the original split overlap. This is partial raw-image leakage, "
            "not evidence of complete sample or missing-edge label duplication."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": result["status"],
                "checks": checks,
                "spatial_overlap": {
                    key: value
                    for key, value in result["spatial_overlap"].items()
                    if key != "pairs"
                },
                "global_graph_audit": result["global_graph_audit"],
            },
            indent=2,
        )
    )
    if not result["status"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
