"""Train the surrogate's MLP ensemble on `gen-surrogate-data`'s dataset.

Each member is a 17 -> 512 -> 512 -> 512 -> 6 ReLU MLP trained with AdamW
and cosine LR decay on standardized inputs and outputs (statistics from
the train split), with early stopping on the validation split's weighted
loss.  The loss weights each component by 1/tolerated-std^2 (sqrt diag
CALIBRATED_R), so training minimizes what the acceptance gate measures.
Inference runs in numpy (see surrogate.py); this module is the only torch
user.

The command's summary compares the ensemble's validation errors, rotated
into the world frame, against sqrt(diag(CALIBRATED_R)) — the prediction
error the estimator already tolerates between the Drake plant and the
truth sim.  The surrogate must add little on top of that.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .estimator import CALIBRATED_R
from .surrogate import N_INPUTS, N_OUTPUTS, MLPWeights, SurrogateEnsemble, save_ensemble

HIDDEN = 512
BATCH_SIZE = 1024


def dataset_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the dataset into (N, 17) inputs, (N, 6) outputs and the
    validation mask, in surrogate.py's column order."""
    with np.load(path, allow_pickle=True) as data:
        inputs = np.concatenate(
            [
                data["eta"],
                data["start_pose"],
                data["start_velocity"],
                data["site_position"],
                data["site_normal"],
                data["speed"][:, None],
                data["time"][:, None],
                data["tip_path"][:, :2].reshape(len(data["time"]), -1),
            ],
            axis=1,
        )
        outputs = np.concatenate(
            [data["displacement"], data["push_end_velocity"]], axis=1
        )
        return inputs, outputs, data["is_val"].astype(bool)


def loss_weights(output_std: np.ndarray) -> np.ndarray:
    """Get per-output loss weights that make the standardized MSE measure
    what the acceptance gate measures: each component's raw error against
    its tolerated plant-vs-truth std (sqrt diag R)."""
    tolerated = np.sqrt(np.diag(CALIBRATED_R))
    return (output_std / tolerated) ** 2


def train_member(
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    loss_w: torch.Tensor,
    epochs: int,
    patience: int,
) -> tuple[MLPWeights, float]:
    """Train one member, returning its best weights (by validation loss)
    and that loss.  Weights come back as float64 numpy for numpy inference."""
    generator = torch.Generator().manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(N_INPUTS, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, N_OUTPUTS),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    def weighted_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((prediction - target) ** 2 * loss_w).mean()

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for _epoch in range(epochs):
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss: torch.Tensor = weighted_mse(model(x_train[batch]), y_train[batch])
            # torch 2.15's stubs leave Tensor.backward untyped.
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            val_loss = float(weighted_mse(model(x_val), y_val))
        if val_loss < best_loss - 1e-9:
            best_loss, stale = val_loss, 0
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, len(model), 2):
        linear = cast_linear(model[i])
        layers.append(
            (
                linear.weight.detach().numpy().astype(np.float64),
                linear.bias.detach().numpy().astype(np.float64),
            )
        )
    return tuple(layers), best_loss


def cast_linear(module: nn.Module) -> nn.Linear:
    """Narrow a Sequential's module to its Linear layers."""
    assert isinstance(module, nn.Linear)
    return module


def world_frame_rmse(
    outputs: np.ndarray, targets: np.ndarray, start_yaw: np.ndarray
) -> np.ndarray:
    """Get the (6,) RMS of start-frame output errors, rotated into the world
    frame per sample (the measurement's component order: x, y, yaw, vx, vy,
    wz)."""
    errors = outputs - targets
    c, s = np.cos(start_yaw), np.sin(start_yaw)
    to_body = np.stack([np.stack([c, s], axis=-1), np.stack([-s, c], axis=-1)], axis=-2)
    world_xy = np.einsum("nij,nj->ni", to_body, errors[:, :2])
    world = np.column_stack(
        [
            world_xy,
            errors[:, 2],
            np.einsum("nij,nj->ni", to_body, errors[:, 3:5]),
            errors[:, 5],
        ]
    )
    return np.sqrt((world**2).mean(axis=0))


def train_ensemble(
    dataset_path: Path,
    out_path: Path,
    members: int,
    epochs: int,
    patience: int,
    seed: int,
) -> str:
    """Train the ensemble on the dataset, save it, and return the summary:
    validation RMSE against the estimator's tolerated prediction error."""
    x, y, is_val = dataset_arrays(dataset_path)
    train, val = ~is_val, is_val
    input_mean, input_std = x[train].mean(axis=0), x[train].std(axis=0)
    output_mean, output_std = y[train].mean(axis=0), y[train].std(axis=0)
    # A near-zero std means a constant column (float dust included); its
    # standardized values would explode on any deploy-time deviation.
    input_std[input_std < 1e-6] = 1.0
    output_std[output_std < 1e-6] = 1.0

    x_train = torch.tensor((x[train] - input_mean) / input_std, dtype=torch.float32)
    y_train = torch.tensor((y[train] - output_mean) / output_std, dtype=torch.float32)
    x_val = torch.tensor((x[val] - input_mean) / input_std, dtype=torch.float32)
    y_val = torch.tensor((y[val] - output_mean) / output_std, dtype=torch.float32)
    loss_w = torch.tensor(loss_weights(output_std), dtype=torch.float32)

    weights = []
    for i in range(members):
        member_weights, best_loss = train_member(
            seed + i, x_train, y_train, x_val, y_val, loss_w, epochs, patience
        )
        weights.append(member_weights)
        print(f"member {i}: best weighted val MSE {best_loss:.6f}", flush=True)

    ensemble = SurrogateEnsemble(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        members=tuple(weights),
    )
    save_ensemble(out_path, ensemble)

    # Acceptance check: mean prediction's val RMSE in the world frame
    # against the estimator's tolerated plant-vs-truth error.
    predictions = ensemble.predict(x[val])
    rmse = world_frame_rmse(predictions, y[val], x[val, 2])
    tolerated = np.sqrt(np.diag(CALIBRATED_R))
    labels = ("x", "y", "yaw", "vx", "vy", "wz")
    lines = [
        f"ensemble of {members} members saved to {out_path}",
        "component   surrogate RMSE   tolerated (sqrt diag R)   ratio",
    ]
    worst = 0.0
    for label, got, want in zip(labels, rmse, tolerated):
        ratio = float(got / want)
        worst = max(worst, ratio)
        lines.append(f"{label:<10} {got:12.3e} {want:19.3e} {ratio:14.2f}")
    lines.append(
        f"worst ratio {worst:.2f} — {'PASS' if worst <= 0.5 else 'FAIL'} (gate 0.5)"
    )

    # Ensemble spread: the epistemic noise the env's randomization scales.
    member_outputs = np.stack(
        [ensemble.predict(x[val], member=i) for i in range(members)]
    )
    spread = member_outputs.std(axis=0).mean(axis=0)
    lines.append(
        "member spread (std over members, start frame): "
        + "  ".join(f"{v:.2e}" for v in spread)
    )
    return "\n".join(lines)
