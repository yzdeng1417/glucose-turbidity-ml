"""Paper training operations, preserved from the grouped research implementation."""
from __future__ import annotations
import copy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from cmp.dataset import CMPArrays
from cmp.models import BlockEncoderConfig, FusionConfig, MultiTowerClassifier, MultiTowerConfig
from cmp.scaling import _fit_stats, _standardize
from cmp.torch_data import _BlockDataset, _set_seed

def subset(arr: CMPArrays, indices: np.ndarray) -> CMPArrays:
    return CMPArrays({k: v[indices] for k, v in arr.blocks.items()},
                     arr.columns, None if arr.y is None else arr.y[indices], arr.feature_names)


def build_model_config(block_dims: dict, protocol: dict) -> MultiTowerConfig:
    cfg = protocol["model"]
    return MultiTowerConfig(
        block_dims=block_dims,
        encoder=BlockEncoderConfig(
            out_dim=cfg["encoder_out_dim"], dropout=cfg["encoder_dropout"],
            hidden_channels=cfg["hidden_channels"], kernel_size=cfg["kernel_size"],
            curve_conv_depth=cfg["curve_conv_depth"], curve_batch_norm=cfg["curve_batch_norm"],
            curve_use_diff_channel=cfg["curve_use_diff_channel"],
        ),
        fusion=FusionConfig(kind=cfg["fusion"], n_heads=cfg["attention_heads"],
                            hidden_dim=cfg["head_hidden_dim"], dropout=cfg["head_dropout"]),
    )


def predict(model, stats, arr: CMPArrays, batch_size: int) -> np.ndarray:
    loader = DataLoader(_BlockDataset(_standardize(arr, stats), None), batch_size=batch_size, shuffle=False)
    model.eval()
    outputs = []
    with torch.no_grad():
        for blocks, _ in loader:
            outputs.append(torch.sigmoid(model(blocks)).numpy().astype(float))
    return np.concatenate(outputs)


def fit_member(arr: CMPArrays, fit_indices: np.ndarray, stop_indices: np.ndarray,
               seed: int, protocol: dict, checkpoint: Path):
    cfg = protocol["training"]
    stop_cfg = protocol["early_stopping"]
    _set_seed(seed)
    fit_arr, stop_arr = subset(arr, fit_indices), subset(arr, stop_indices)
    model_cfg = build_model_config(arr.block_dims, protocol)
    stats = _fit_stats(fit_arr, model_cfg.timeseries_blocks)
    fit_blocks = _standardize(fit_arr, stats)
    stop_blocks = _standardize(stop_arr, stats)
    train_loader = DataLoader(_BlockDataset(fit_blocks, fit_arr.y), batch_size=cfg["batch_size"], shuffle=True)
    stop_loader = DataLoader(_BlockDataset(stop_blocks, stop_arr.y), batch_size=cfg["batch_size"], shuffle=False)
    model = MultiTowerClassifier(model_cfg).cpu()
    positives = int((fit_arr.y == 1).sum())
    negatives = int((fit_arr.y == 0).sum())
    if not positives or not negatives:
        raise ValueError("Both classes are required for training")
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    history = []
    best_auc, best_epoch, best_state, stale = -np.inf, 0, None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for blocks, truth in train_loader:
            loss = loss_function(model(blocks), truth)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            optimizer.step()
            loss_sum += float(loss.item()) * len(truth)
        model.eval()
        stop_scores = []
        with torch.no_grad():
            for blocks, _ in stop_loader:
                stop_scores.append(torch.sigmoid(model(blocks)).numpy())
        auc = float(roc_auc_score(stop_arr.y, np.concatenate(stop_scores)))
        history.append({"epoch": epoch, "optimizer_loss": loss_sum / len(fit_indices), "grouped_early_stop_auc": auc})
        if auc > best_auc + stop_cfg["minimum_improvement"]:
            best_auc, best_epoch, stale = auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= stop_cfg["patience"]:
                break
    if best_state is None:
        raise RuntimeError("No finite grouped early-stop model")
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(), "model_config": asdict(model_cfg),
        "means": stats.means, "stds": stats.stds, "standardize": stats.standardize,
        "seed": seed, "best_epoch": best_epoch, "optimizer_indices": fit_indices,
        "early_stop_indices": stop_indices,
    }, checkpoint)
    pd.DataFrame(history).to_csv(checkpoint.with_suffix(".history.tsv"), sep="\t", index=False, float_format="%.17g")
    return model, stats, {"seed": seed, "best_epoch": best_epoch, "epochs_run": len(history),
                           "best_grouped_early_stop_auc": best_auc,
                           "optimizer_rows": len(fit_indices), "early_stop_rows": len(stop_indices)}
