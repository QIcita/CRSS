#!/usr/bin/env python3
"""Build a leakage-resistant NREDD TLP split from spatial block groups.

The split assignment uses only block origins, fixed image extents, and raw SWC
node counts. Labels, model predictions, and historical test scores are never
loaded while optimizing the assignment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import tifffile


SPLITS = ("train", "validation", "test")
SOURCE_SPLITS = ("train", "test")
FEATURE_LAYOUT = {
    "raw_data": ".swc",
    "image": ".tif",
    "gt": ".swc",
    "label": ".txt",
    "SP_feature": "",
    "point_surface_mc_v1": ".npz",
}
ORIGIN_PATTERN = re.compile(r"^x(-?\d+)_y(-?\d+)_z(-?\d+)$")


@dataclass(frozen=True)
class Block:
    name: str
    stem: str
    source_split: str
    origin: tuple[int, int, int]
    shape_xyz: tuple[int, int, int]
    node_count: int

    @property
    def upper(self) -> tuple[int, int, int]:
        return tuple(a + b for a, b in zip(self.origin, self.shape_xyz))


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_origin(stem: str) -> tuple[int, int, int]:
    match = ORIGIN_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Cannot parse block origin: {stem}")
    return tuple(int(value) for value in match.groups())


def count_swc_nodes(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    if count == 0:
        raise ValueError(f"Empty SWC: {path}")
    return count


def read_shape_xyz(path: Path) -> tuple[int, int, int]:
    with tifffile.TiffFile(path) as handle:
        shape_zyx = tuple(int(value) for value in handle.series[0].shape)
    if len(shape_zyx) != 3:
        raise ValueError(f"Expected 3-D TIFF, got {shape_zyx}: {path}")
    return tuple(reversed(shape_zyx))


def collect_blocks(data_root: Path) -> list[Block]:
    blocks = []
    names = set()
    for source_split in SOURCE_SPLITS:
        split_root = data_root / source_split
        for swc_path in sorted((split_root / "raw_data").glob("*.swc")):
            if swc_path.name in names:
                raise ValueError(f"Duplicate block filename: {swc_path.name}")
            names.add(swc_path.name)
            stem = swc_path.stem
            image_path = split_root / "image" / f"{stem}.tif"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            shape_xyz = read_shape_xyz(image_path)
            for feature, suffix in FEATURE_LAYOUT.items():
                expected = split_root / feature / f"{stem}{suffix}"
                if not expected.exists():
                    raise FileNotFoundError(expected)
            blocks.append(
                Block(
                    name=swc_path.name,
                    stem=stem,
                    source_split=source_split,
                    origin=parse_origin(stem),
                    shape_xyz=shape_xyz,
                    node_count=count_swc_nodes(swc_path),
                )
            )
    return sorted(blocks, key=lambda block: block.name)


def positive_volume_overlap(left: Block, right: Block) -> bool:
    return all(
        max(left.origin[axis], right.origin[axis])
        < min(left.upper[axis], right.upper[axis])
        for axis in range(3)
    )


def build_groups(blocks: list[Block]):
    union_find = UnionFind(len(blocks))
    overlap_pairs = []
    for left in range(len(blocks)):
        for right in range(left + 1, len(blocks)):
            if positive_volume_overlap(blocks[left], blocks[right]):
                union_find.union(left, right)
                overlap_pairs.append((left, right))
    members_by_root: dict[int, list[int]] = {}
    for index in range(len(blocks)):
        members_by_root.setdefault(union_find.find(index), []).append(index)
    groups = []
    for members in members_by_root.values():
        members = sorted(members, key=lambda index: blocks[index].name)
        groups.append(
            {
                "id": f"group_{min(blocks[index].stem for index in members)}",
                "members": members,
                "block_count": len(members),
                "node_count": sum(blocks[index].node_count for index in members),
            }
        )
    groups.sort(key=lambda row: row["id"])
    return groups, overlap_pairs


def solve_assignment(groups, ratios):
    group_count = len(groups)
    total_blocks = sum(group["block_count"] for group in groups)
    total_nodes = sum(group["node_count"] for group in groups)
    target_blocks = np.asarray(ratios, dtype=np.float64) * total_blocks
    target_nodes = np.asarray(ratios, dtype=np.float64) * total_nodes

    # x[g, s], then positive/negative deviations for block and node totals.
    x_count = group_count * len(SPLITS)
    dev_offset = x_count
    variable_count = x_count + 12
    integrality = np.zeros(variable_count, dtype=np.int32)
    integrality[:x_count] = 1
    lower = np.zeros(variable_count, dtype=np.float64)
    upper = np.full(variable_count, np.inf, dtype=np.float64)
    upper[:x_count] = 1.0

    cost = np.zeros(variable_count, dtype=np.float64)
    for split_index in range(len(SPLITS)):
        cost[dev_offset + split_index * 2 : dev_offset + split_index * 2 + 2] = (
            1.0 / max(target_blocks[split_index], 1.0)
        )
        node_dev = dev_offset + 6 + split_index * 2
        cost[node_dev : node_dev + 2] = 1.0 / max(target_nodes[split_index], 1.0)
    # Deterministic, negligible tie-break among equal primary objectives.
    for group_index in range(group_count):
        for split_index in range(len(SPLITS)):
            cost[group_index * 3 + split_index] = (
                1e-10 * (group_index * 3 + split_index + 1)
            )

    equality_rows = []
    equality_values = []
    for group_index in range(group_count):
        row = np.zeros(variable_count)
        row[group_index * 3 : group_index * 3 + 3] = 1.0
        equality_rows.append(row)
        equality_values.append(1.0)
    for split_index in range(len(SPLITS)):
        block_row = np.zeros(variable_count)
        node_row = np.zeros(variable_count)
        for group_index, group in enumerate(groups):
            block_row[group_index * 3 + split_index] = group["block_count"]
            node_row[group_index * 3 + split_index] = group["node_count"]
        block_dev = dev_offset + split_index * 2
        node_dev = dev_offset + 6 + split_index * 2
        block_row[block_dev] = -1.0
        block_row[block_dev + 1] = 1.0
        node_row[node_dev] = -1.0
        node_row[node_dev + 1] = 1.0
        equality_rows.extend((block_row, node_row))
        equality_values.extend((target_blocks[split_index], target_nodes[split_index]))

    # The three largest spatial regions must land in distinct splits. We
    # enumerate the six mappings and choose the best label-free balance.
    largest = sorted(
        range(group_count),
        key=lambda index: (
            -groups[index]["block_count"],
            -groups[index]["node_count"],
            groups[index]["id"],
        ),
    )[:3]
    candidates = []
    for permutation in itertools.permutations(range(3)):
        local_lower = lower.copy()
        local_upper = upper.copy()
        for group_index, assigned_split in zip(largest, permutation):
            for split_index in range(3):
                value = 1.0 if split_index == assigned_split else 0.0
                local_lower[group_index * 3 + split_index] = value
                local_upper[group_index * 3 + split_index] = value
        result = milp(
            c=cost,
            integrality=integrality,
            bounds=Bounds(local_lower, local_upper),
            constraints=LinearConstraint(
                np.asarray(equality_rows),
                np.asarray(equality_values),
                np.asarray(equality_values),
            ),
            options={"time_limit": 60.0, "mip_rel_gap": 0.0},
        )
        if not result.success or result.x is None:
            continue
        assignment = tuple(
            int(np.argmax(result.x[group_index * 3 : group_index * 3 + 3]))
            for group_index in range(group_count)
        )
        block_totals = np.asarray(
            [
                sum(
                    group["block_count"]
                    for group, split_index in zip(groups, assignment)
                    if split_index == selected
                )
                for selected in range(3)
            ],
            dtype=np.float64,
        )
        node_totals = np.asarray(
            [
                sum(
                    group["node_count"]
                    for group, split_index in zip(groups, assignment)
                    if split_index == selected
                )
                for selected in range(3)
            ],
            dtype=np.float64,
        )
        primary = float(
            (np.abs(block_totals - target_blocks) / target_blocks).sum()
        ) + float((np.abs(node_totals - target_nodes) / target_nodes).sum())
        candidates.append((round(primary, 12), assignment, block_totals, node_totals))
    if not candidates:
        raise RuntimeError("No feasible spatial-group assignment found")
    _, assignment, block_totals, node_totals = min(
        candidates, key=lambda row: (row[0], row[1])
    )
    return {
        "assignment": assignment,
        "target_blocks": target_blocks,
        "target_nodes": target_nodes,
        "block_totals": block_totals,
        "node_totals": node_totals,
        "largest_group_indices": largest,
    }


def source_path(data_root: Path, block: Block, feature: str) -> Path:
    suffix = FEATURE_LAYOUT[feature]
    return data_root / block.source_split / feature / f"{block.stem}{suffix}"


def create_link_tree(output_root: Path, data_root: Path, blocks, split_members):
    staging = output_root.with_name(output_root.name + ".staging")
    if output_root.exists() or staging.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing split tree: {output_root} or {staging}"
        )
    for split in SPLITS:
        for feature in FEATURE_LAYOUT:
            (staging / split / feature).mkdir(parents=True, exist_ok=False)
        for block_index in split_members[split]:
            block = blocks[block_index]
            for feature, suffix in FEATURE_LAYOUT.items():
                source = source_path(data_root, block, feature).resolve(strict=True)
                destination = staging / split / feature / f"{block.stem}{suffix}"
                os.symlink(source, destination, target_is_directory=source.is_dir())
    return staging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/link_pred_data")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/link_pred_data/spatial_grouped_v1"),
    )
    parser.add_argument(
        "--ratios", type=float, nargs=3, default=(0.60, 0.20, 0.20)
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.ratios) or not np.isclose(
        sum(args.ratios), 1.0
    ):
        raise ValueError("Split ratios must be positive and sum to one")

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    project_root = Path(__file__).resolve().parents[2]
    blocks = collect_blocks(data_root)
    groups, overlap_pairs = build_groups(blocks)
    solution = solve_assignment(groups, np.asarray(args.ratios))
    split_members = {split: [] for split in SPLITS}
    split_groups = {split: [] for split in SPLITS}
    for group_index, split_index in enumerate(solution["assignment"]):
        split = SPLITS[split_index]
        split_members[split].extend(groups[group_index]["members"])
        split_groups[split].append(group_index)
    for split in SPLITS:
        split_members[split].sort(key=lambda index: blocks[index].name)

    cross_split_overlaps = []
    block_split = {
        block_index: split
        for split, members in split_members.items()
        for block_index in members
    }
    for left, right in overlap_pairs:
        if block_split[left] != block_split[right]:
            cross_split_overlaps.append((blocks[left].name, blocks[right].name))
    if cross_split_overlaps:
        raise RuntimeError(f"Cross-split spatial overlap: {cross_split_overlaps}")

    manifest = {
        "schema_version": 1,
        "protocol": "NREDD_TLP_spatial_grouped_v1",
        "source": {
            "project_root": str(project_root),
            "data_root": str(data_root),
            "git_commit": git_commit(project_root),
            "official_source_splits": list(SOURCE_SPLITS),
        },
        "assignment_policy": {
            "inputs_used": [
                "filename-derived XYZ block origin",
                "TIFF metadata shape",
                "raw SWC node count",
            ],
            "inputs_forbidden": [
                "label contents",
                "ground-truth missing-edge counts",
                "model predictions",
                "historical validation/test metrics",
            ],
            "overlap_rule": "strictly positive AABB intersection in x, y, and z",
            "group_rule": "transitive connected components of the overlap graph",
            "optimizer": "SciPy MILP minimizing normalized L1 block/node imbalance",
            "largest_three_groups_forced_to_distinct_splits": True,
            "target_ratios": dict(zip(SPLITS, args.ratios)),
        },
        "totals": {
            "blocks": len(blocks),
            "nodes": sum(block.node_count for block in blocks),
            "spatial_overlap_edges": len(overlap_pairs),
            "spatial_groups": len(groups),
            "cross_split_overlap_edges": len(cross_split_overlaps),
        },
        "splits": {},
        "groups": [],
    }
    for split_index, split in enumerate(SPLITS):
        members = split_members[split]
        manifest["splits"][split] = {
            "blocks": len(members),
            "nodes": sum(blocks[index].node_count for index in members),
            "block_fraction": len(members) / len(blocks),
            "node_fraction": (
                sum(blocks[index].node_count for index in members)
                / sum(block.node_count for block in blocks)
            ),
            "spatial_groups": len(split_groups[split]),
            "source_official_train_blocks": sum(
                blocks[index].source_split == "train" for index in members
            ),
            "source_official_test_blocks": sum(
                blocks[index].source_split == "test" for index in members
            ),
            "filenames": [blocks[index].name for index in members],
        }
    for group_index, group in enumerate(groups):
        members = group["members"]
        manifest["groups"].append(
            {
                "id": group["id"],
                "assigned_split": SPLITS[solution["assignment"][group_index]],
                "blocks": group["block_count"],
                "nodes": group["node_count"],
                "filenames": [blocks[index].name for index in members],
            }
        )

    staging = create_link_tree(output_root, data_root, blocks, split_members)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii")
    (staging / "manifest.json").write_bytes(payload)
    (staging / "manifest.sha256").write_text(
        f"{sha256_bytes(payload)}  manifest.json\n", encoding="ascii"
    )
    staging.rename(output_root)
    print(json.dumps({"output_root": str(output_root), **manifest["totals"], "splits": {
        split: {
            key: value
            for key, value in manifest["splits"][split].items()
            if key not in ("filenames",)
        }
        for split in SPLITS
    }}, indent=2))


if __name__ == "__main__":
    main()
