"""Nitro Contestant Cloud pre-judging script for Panda MNIST.

Pattern: Data task. Contestant uploads a TorchScript model (.pt file) or cloudpickled
object. The script loads the model, runs inference on test images, computes accuracy per
subtask, and writes a single packed answer row per subtask into submission.csv.

Each answer cell is:
  "<partial_macro_acc> <complete_macro_acc> <param_count>"

The platform judge reconstructs the real subtask points from that payload.

Scoring formula (per subtask):
  Both subtasks: macro-averaged accuracy (equal weight per scanner).
  Subtask 1: 3 scanners. S = accuracy*100 - P/7500. Score = 0 if S <= 90 else (S - 90)/10 * 30, capped at 30.
  Subtask 2: 8 scanners. S = macro_A*100 - P/7500. Score = 0 if S <= 65 else (S - 65)/25 * 70, capped at 70.
"""

import io
import zipfile

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PENALTY_DIVISOR = 7_500
S_THRESHOLD = {1: 90.0, 2: 65.0}
S_CEILING = {1: 100.0, 2: 90.0}

# Real subtask points returned by the Nitro judge
SUBTASK_MAX = {1: 30, 2: 70}

# Which datasets each subtask uses
SUBTASK_DATASETS = {
    1: ["ds1", "ds2", "ds3"],
    2: ["ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7", "ds8"],
}

PARTIAL_SEED = 42
PARTIAL_FRAC = 0.5


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_torchscript(buf: bytes):
    """Load a single TorchScript model from bytes."""
    model = torch.jit.load(io.BytesIO(buf))
    model.eval()
    return model


def load_models(raw_bytes: bytes) -> tuple:
    """Load subtask 1 and subtask 2 models from a zip of .pt files.

    Expected zip contents:
      model_sub1.pt  — model for subtask 1 (accepts (N, 1, 28, 28) float32)
      model_sub2.pt  — model for subtask 2 (accepts (N, 3, 28, 28) float32)

    Returns (model_sub1, model_sub2).
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            if "model_sub1.pt" not in zf.namelist():
                raise ValueError("zip missing model_sub1.pt")
            if "model_sub2.pt" not in zf.namelist():
                raise ValueError("zip missing model_sub2.pt")
            m1 = _load_torchscript(zf.read("model_sub1.pt"))
            m2 = _load_torchscript(zf.read("model_sub2.pt"))
        return m1, m2
    except zipfile.BadZipFile:
        raise ValueError(
            "Submission must be a .zip containing model_sub1.pt and model_sub2.pt"
        )


def count_params(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_test_images(test_images_path: str = "test_data"):
    """Load test images from the Test Data .npz file uploaded to Nitro.

    In Contestant Cloud, this is the "test_data" file (the .npz).
    All values normalized to [0, 1] float32.

    Returns dict with keys:
      scanner1, scanner2, scanner3  — (N, 1, 28, 28) grayscale
      scanner4                     — (N, 3, 28, 28) RGB (Ishihara)
      scanner5                     — (N, 1, 28, 28) grayscale (barcode)
      scanner6                     — (N, 1, 28, 28) grayscale (braille)
      scanner7                     — (N, 1, 28, 28) grayscale (Roman)
      scanner8                     — (N, 1, 28, 28) grayscale (Arabic)
    """
    data = dict(np.load(test_images_path))
    result = {}
    # 1-channel grayscale scanners
    for key in ["scanner1", "scanner2", "scanner3",
                "scanner5", "scanner6", "scanner7", "scanner8"]:
        x = data[key].astype(np.float32) / 255.0
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x.transpose(0, 3, 1, 2)
        elif x.ndim == 3:
            x = x[:, np.newaxis, :, :]
        result[key] = x
    # 3-channel scanner4
    x4 = data["scanner4"].astype(np.float32) / 255.0
    if x4.ndim == 4 and x4.shape[-1] == 3:
        x4 = x4.transpose(0, 3, 1, 2)
    result["scanner4"] = x4
    return result


def load_test_labels(test_images_path: str = "test_data") -> dict:
    """Load per-scanner test labels from the Test Data .npz.

    Labels are stored as "scannerN_labels" keys in the .npz.
    Returns {scanner_name: numpy_array (N,) int64}.
    """
    data = dict(np.load(test_images_path))
    result = {}
    for src in ["scanner1", "scanner2", "scanner3", "scanner4",
                "scanner5", "scanner6", "scanner7", "scanner8"]:
        key = f"{src}_labels"
        if key in data:
            result[src] = data[key].astype(np.int64).ravel()
    return result


def run_inference(model, images: np.ndarray, device: torch.device) -> np.ndarray:
    """Run model on a batch of images, return predicted classes (N,) int64."""
    x = torch.from_numpy(images).to(device)  # Move input to device
    with torch.no_grad():
        output = model(x)
    # Output may be (N, 10) logits or (N,) class indices
    if output.ndim == 2:
        preds = output.argmax(dim=1).cpu().numpy().astype(np.int64)  # Add .cpu()
    else:
        preds = output.cpu().numpy().astype(np.int64)  # Add .cpu()
    return preds

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def compute_score(accuracy: float, params: int, max_points: int, subtask_id: int = 2) -> int:
    """Apply scoring formula. Returns integer subtask points.

    Subtask 1: threshold 90, ceiling 100, max 30 points.
    Subtask 2: threshold 65, ceiling 90, max 70 points.
    """
    threshold = S_THRESHOLD[subtask_id]
    ceiling = S_CEILING[subtask_id]
    s = accuracy * 100.0 - params / PENALTY_DIVISOR
    if s <= threshold:
        return 0
    points = (s - threshold) / (ceiling - threshold) * max_points
    points = min(points, float(max_points))
    return int(round(points))


def build_submission(raw_bytes, test_images_path="test_data", verbose=True):
    """Run the full pipeline and return a submission DataFrame.

    raw_bytes: zip containing model_sub1.pt and model_sub2.pt.
      - model_sub1: accepts (N, 1, 28, 28) float32, returns (N, 10) logits
      - model_sub2: accepts (N, 3, 28, 28) float32, returns (N, 10) logits

    Two datapoints per subtask:
      - "partial"  — score on 50% random split (seed=42)
      - "complete" — score on 100% of test data
    """
    device = torch.device("mps")
    
    model1, model2 = load_models(raw_bytes)
    model1.to(device)  # Move model 1 to GPU
    model2.to(device)  # Move model 2 to GPU
    
    params1 = count_params(model1)
    params2 = count_params(model2)    
    if verbose:
        print(f"Subtask 1 model: {params1:,} params")
        print(f"Subtask 2 model: {params2:,} params")

    test_images = load_test_images(test_images_path)
    test_labels = load_test_labels(test_images_path)

    # --- Subtask 1: per-scanner macro-averaged accuracy (3 scanners, model1) ---
    sub1_scanner_accs = []
    for scanner_key in ["scanner1", "scanner2", "scanner3"]:
        imgs = test_images[scanner_key]  # (N, 1, 28, 28)
        preds = run_inference(model1, imgs, device=torch.device("mps"))
        lbls = test_labels[scanner_key]
        sub1_scanner_accs.append(float((preds == lbls).mean()))
    acc1_full = float(np.mean(sub1_scanner_accs))

    # --- Subtask 2: per-scanner macro-averaged accuracy (8 scanners, model2) ---
    # Build per-scanner accuracy list
    sub2_scanner_accs = []
    sub2_scanner_map = {
        "scanner1": (test_images["scanner1"], 1),  # 1-ch -> replicate to 3
        "scanner2": (test_images["scanner2"], 1),
        "scanner3": (test_images["scanner3"], 1),
        "scanner4": (test_images["scanner4"], 3),  # 3-ch native
        "scanner5": (test_images["scanner5"], 1),  # barcode -> replicate
        "scanner6": (test_images["scanner6"], 1),  # braille -> replicate
        "scanner7": (test_images["scanner7"], 1),  # Roman -> replicate
        "scanner8": (test_images["scanner8"], 1),  # Arabic -> replicate
    }
    for scanner_key, (imgs, ch) in sub2_scanner_map.items():
        if ch == 1:
            imgs = np.repeat(imgs, 3, axis=1)  # (N,1,28,28) -> (N,3,28,28)
        preds = run_inference(model2, imgs, device=torch.device("mps"))
        lbls = test_labels.get(scanner_key)
        if lbls is not None:
            sub2_scanner_accs.append(float((preds == lbls).mean()))
    acc2_full = float(np.mean(sub2_scanner_accs)) if sub2_scanner_accs else 0.0

    # --- Partial split: 50% random per-scanner, seed fixed ---
    rng = np.random.default_rng(PARTIAL_SEED)
    # Subtask 1 partial: per-scanner macro-averaged
    sub1_partial_accs = []
    for scanner_key in ["scanner1", "scanner2", "scanner3"]:
        imgs = test_images[scanner_key]
        preds = run_inference(model1, imgs, device=torch.device("mps"))
        lbls = test_labels[scanner_key]
        mask = rng.random(len(lbls)) < PARTIAL_FRAC
        sub1_partial_accs.append(float((preds[mask] == lbls[mask]).mean()))
    acc1_partial = float(np.mean(sub1_partial_accs))
    sub2_partial_accs = []
    for scanner_key, (imgs, ch) in sub2_scanner_map.items():
        if ch == 1:
            imgs = np.repeat(imgs, 3, axis=1)
        preds = run_inference(model2, imgs, device=torch.device("mps"))
        lbls = test_labels.get(scanner_key)
        if lbls is not None:
            mask2 = rng.random(len(lbls)) < PARTIAL_FRAC
            sub2_partial_accs.append(float((preds[mask2] == lbls[mask2]).mean()))
    acc2_partial = float(np.mean(sub2_partial_accs)) if sub2_partial_accs else 0.0

    # Build answer string: partial_macro_acc complete_macro_acc param_count
    # Two rows, one per subtask. Judge reconstructs scores from the formula.
    rows = []
    for sub_id, acc_partial, acc_full, params in [
        (1, acc1_partial, acc1_full, params1),
        (2, acc2_partial, acc2_full, params2),
    ]:
        answer = f"{acc_partial:.6f} {acc_full:.6f} {params}"
        rows.append({
            "subtaskID": sub_id,
            "datapointID": "scores",
            "answer": answer,
        })
        if verbose:
            score_p = compute_score(acc_partial, params, SUBTASK_MAX[sub_id], subtask_id=sub_id)
            score_c = compute_score(acc_full, params, SUBTASK_MAX[sub_id], subtask_id=sub_id)
            print(f"Subtask {sub_id}: partial acc={acc_partial:.4f} score={score_p}/{SUBTASK_MAX[sub_id]}  "
                  f"complete acc={acc_full:.4f} score={score_c}/{SUBTASK_MAX[sub_id]}  params={params:,}")

    return pd.DataFrame(rows, columns=["subtaskID", "datapointID", "answer"])


def main():
    # Read the file directly from submission2.zip in binary mode
    try:
        with open("submission2.zip", "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        print("Error: 'submission2.zip' was not found in the current directory.")
        return

    # In cloud: "test_data" is the .npz (images + labels)
    submission = build_submission(
        raw,
        test_images_path="test_data.npz",
        verbose=False,
    )
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()