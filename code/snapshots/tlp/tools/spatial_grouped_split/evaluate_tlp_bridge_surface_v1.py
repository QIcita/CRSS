#!/usr/bin/env python3
"""Evaluate bridge-surface TLP against frozen spatial-grouped comparators."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.repair_dataset import RepairDataset
from tools.evaluate_tlp_extended_metrics import (
    compare_datasets,
    component_labels,
    dataset_graph_audit,
    evaluate_model,
    load_model,
    predict_dataset,
    select_f1_threshold,
    synchronize,
)


BRIDGE_MODE = "point_surface_bridge_v1_tlp"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(value):
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def relabel_validation_threshold(metrics):
    thresholded = metrics["thresholded"]
    calibrated = thresholded.pop("train_calibrated_best_missing_f1")
    calibrated["calibration_on_validation"] = calibrated.pop(
        "calibration_on_train"
    )
    thresholded["validation_calibrated_best_missing_f1"] = calibrated


def load_bridge_model(checkpoint, device):
    from model.check_model import TLP

    model = TLP(feature_mode=BRIDGE_MODE).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_bridge_dataset(model, dataset, device):
    blocks = []
    maximum_asymmetry = 0.0
    synchronize(device)
    with torch.inference_mode():
        for index in range(len(dataset)):
            (
                xyz,
                target,
                adjacency,
                path,
                morphology,
                point_mask,
                components,
                pair_index,
                bridge_features,
                bridge_mask,
            ) = dataset[index]
            output = model(
                xyz.to(device),
                adjacency.to(device),
                morphology.to(device),
                point_mask=point_mask.to(device),
                component_ids=components.to(device),
                pair_index=pair_index.to(device),
                bridge_features=bridge_features.to(device),
                bridge_mask=bridge_mask.to(device),
            )
            probability = torch.sigmoid(output).cpu().numpy().astype(np.float32)
            maximum_asymmetry = max(
                maximum_asymmetry,
                float(np.max(np.abs(probability - probability.T))),
            )
            observed = adjacency.numpy() > 0
            blocks.append(
                {
                    "name": Path(path).name,
                    "target": target.numpy() > 0.5,
                    "observed": observed,
                    "component": (
                        components.numpy().astype(np.int64)
                        if components is not None
                        else component_labels(observed)
                    ),
                    "score": probability,
                }
            )
    synchronize(device)
    return blocks, maximum_asymmetry


def build_datasets(data_root):
    datasets = {}
    for split in ("validation", "test"):
        split_root = data_root / split
        datasets[(split, "spe")] = RepairDataset(
            str(split_root), False, feature_mode="spe"
        )
        datasets[(split, "point_surface_v3b_tlp")] = RepairDataset(
            str(split_root),
            False,
            feature_mode="point_surface_v3b_tlp",
            point_root=str(split_root / "point_surface_mc_v1"),
        )
        datasets[(split, BRIDGE_MODE)] = RepairDataset(
            str(split_root),
            False,
            feature_mode=BRIDGE_MODE,
            point_root=str(split_root / "point_surface_mc_v1"),
            bridge_root=str(split_root / "point_surface_bridge_v1"),
        )
        compare_datasets(
            datasets[(split, "spe")],
            datasets[(split, "point_surface_v3b_tlp")],
            split,
        )
        compare_datasets(
            datasets[(split, "spe")], datasets[(split, BRIDGE_MODE)], split
        )
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spe-checkpoint", type=checkpoint_path, required=True)
    parser.add_argument("--v3b-checkpoint", type=checkpoint_path, required=True)
    parser.add_argument("--bridge-checkpoint", type=checkpoint_path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/link_pred_data/spatial_grouped_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-threshold", type=float, default=0.99)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_root = args.data_root.resolve()
    datasets = build_datasets(data_root)
    arms = (
        ("Fresh spatial-grouped TLP-Net", "spe", args.spe_checkpoint),
        (
            "Fresh spatial-grouped V3-B + MC",
            "point_surface_v3b_tlp",
            args.v3b_checkpoint,
        ),
        (
            "Fresh spatial-grouped V3-B + MC + bridge surface v1",
            BRIDGE_MODE,
            args.bridge_checkpoint,
        ),
    )
    results = []
    for name, mode, checkpoint in arms:
        model = (
            load_bridge_model(checkpoint, device)
            if mode == BRIDGE_MODE
            else load_model(checkpoint, mode, device)
        )
        predictor = predict_bridge_dataset if mode == BRIDGE_MODE else predict_dataset
        validation_blocks, validation_asymmetry = predictor(
            model, datasets[("validation", mode)], device
        ) if mode == BRIDGE_MODE else predictor(
            model, datasets[("validation", mode)], mode, device
        )
        frozen_threshold, validation_selection = select_f1_threshold(
            validation_blocks
        )
        synchronize(device)
        started = time.perf_counter()
        test_blocks, test_asymmetry = predictor(
            model, datasets[("test", mode)], device
        ) if mode == BRIDGE_MODE else predictor(
            model, datasets[("test", mode)], mode, device
        )
        synchronize(device)
        test_elapsed = time.perf_counter() - started
        metrics = evaluate_model(
            test_blocks, validation_blocks, args.fixed_threshold, args.ece_bins
        )
        relabel_validation_threshold(metrics)
        reported = metrics["thresholded"][
            "validation_calibrated_best_missing_f1"
        ]["calibration_on_validation"]
        if reported["threshold"] != frozen_threshold:
            raise RuntimeError("Validation threshold changed after test inference")
        results.append(
            {
                "model": name,
                "feature_mode": mode,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "maximum_probability_asymmetry": {
                    "validation": validation_asymmetry,
                    "test": test_asymmetry,
                },
                "validation_threshold_frozen_before_test_inference": {
                    "threshold": frozen_threshold,
                    "selection_metrics": validation_selection,
                },
                "test_inference_seconds_per_block": test_elapsed
                / len(datasets[("test", mode)]),
                **metrics,
            }
        )
        del model, validation_blocks, test_blocks
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "protocol": "NREDD_TLP_spatial_grouped_bridge_surface_v1",
        "status": {
            "run_completed": True,
            "engineering_valid": True,
            "innovation_supported": None,
        },
        "dataset": {
            "root": str(data_root),
            "manifest_sha256": sha256(data_root / "manifest.json"),
            "validation_blocks": len(datasets[("validation", "spe")]),
            "test_blocks": len(datasets[("test", "spe")]),
            "test_graph_audit": dataset_graph_audit(datasets[("test", "spe")]),
        },
        "training_boundary": (
            "Only the candidate-conditioned bridge cache, its encoder, and model "
            "input wiring differ. Split, seed, optimizer, loss, schedule, epochs, "
            "batch size, full-adjacency objective, and checkpoint policy are frozen."
        ),
        "threshold_policy": (
            "Select missing-edge micro-F1 threshold independently on validation; "
            "freeze it before one test evaluation."
        ),
        "paper_metric_scope": (
            "Remove diagonal, retain both directed matrix triangles, compute AUC/AP "
            "per test block, then macro-average."
        ),
        "models": results,
        "paper_historical_reference": {
            "auc_percent": 99.57,
            "ap_percent": 77.51,
            "same_split_comparable": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
