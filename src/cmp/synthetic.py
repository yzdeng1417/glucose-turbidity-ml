"""Independent toy data generator: no study data, fitted parameters or downloads."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(seed=2026, n_groups=48, heldout_groups=12):
    """Simulate eight rows per group with shared offsets and latent response.

    Signals have arbitrary units. The label is a Bernoulli draw from a toy
    latent process, not a measured endpoint or an experimentally derived label.
    """
    if n_groups < 12 or not 2 <= heldout_groups <= n_groups - 10:
        raise ValueError("Need >=12 groups and >=10 development / >=2 held-out groups")
    rng = np.random.default_rng(seed)
    heldout = set(rng.permutation(n_groups)[:heldout_groups])
    time = np.linspace(0, 1, 8)
    rows, metadata = [], []
    for group in range(n_groups):
        species = int(rng.integers(3))
        tolerance = rng.normal(0, 0.6)
        glucose_offset, turbidity_offset = rng.normal(0, 0.3, 2)
        for drug in range(2):
            for level in range(4):
                latent = 1.2 - 0.9 * level + tolerance + (drug - 0.5) * 0.4
                probability = 1 / (1 + np.exp(-latent))
                growth = int(rng.random() < probability)
                activity = np.clip(probability + rng.normal(0, 0.15), 0, 1)
                glucose = 2 + glucose_offset - activity * time + rng.normal(0, 0.08, 8)
                turbidity = 0.3 + turbidity_offset + activity * time**1.5 + rng.normal(0, 0.08, 8)
                row = {f"species_{name}": int(species == i) for i, name in enumerate("ABC")}
                row.update({f"antibiotic_{name}": int(drug == i) for i, name in enumerate("AB")})
                row.update({f"antibiotic_concentration_{i}": int(level == i) for i in range(4)})
                row.update({f"glu_time{i+1}": value for i, value in enumerate(glucose)})
                row.update({f"ntu_time{i+1}": value for i, value in enumerate(turbidity)})
                row["growth"] = growth
                rows.append(row)
                metadata.append({
                    "row_id": f"SIM_R{len(rows):04d}", "group_id": f"SIM_G{group+1:03d}",
                    "split": "heldout" if group in heldout else "development",
                })
    return pd.DataFrame(rows), pd.DataFrame(metadata)
