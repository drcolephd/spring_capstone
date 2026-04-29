from __future__ import annotations
import json
import numpy as np
from numpy.typing import NDArray


class MlpGestureModel:
    """
    Two-layer MLP gesture classifier (float32 inference).

    Architecture:
      input [feature_dim]
      → Dense(128) → BatchNorm(128) → ReLU
      → Dense(K)   → Softmax (with temperature scaling)

    Weights are stored as float32 and serialised to/from JSON.
    """

    VERSION = 1

    def __init__(
        self,
        class_labels: list[int],
        feature_mean: NDArray[np.float32],
        feature_std: NDArray[np.float32],
        w1: NDArray[np.float32],          # (128, feature_dim)
        b1: NDArray[np.float32],          # (128,)
        bn_gamma: NDArray[np.float32],    # (128,)
        bn_beta: NDArray[np.float32],     # (128,)
        bn_run_mean: NDArray[np.float32], # (128,)
        bn_run_var: NDArray[np.float32],  # (128,)
        w2: NDArray[np.float32],          # (K, 128)
        b2: NDArray[np.float32],          # (K,)
        temperature: float = 1.0,
    ):
        self.class_labels = class_labels
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.w1 = w1
        self.b1 = b1
        self.bn_gamma = bn_gamma
        self.bn_beta = bn_beta
        self.bn_run_mean = bn_run_mean
        self.bn_run_var = bn_run_var
        self.w2 = w2
        self.b2 = b2
        self.temperature = max(float(temperature), 1e-6)

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

    @property
    def feature_dim(self) -> int:
        return int(self.feature_mean.size)

    def predict(self, features: NDArray[np.float32]) -> tuple[int, float]:
        """Returns (class_label, confidence) for a feature vector."""
        z = (features - self.feature_mean) / np.maximum(self.feature_std, 1e-6)
        a1 = self.w1 @ z + self.b1
        h1 = self.bn_gamma * (a1 - self.bn_run_mean) / np.sqrt(self.bn_run_var + 1e-5) + self.bn_beta
        h1 = np.maximum(h1, 0.0)
        logits = (self.w2 @ h1 + self.b2) / self.temperature
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()
        idx = int(np.argmax(probs))
        return self.class_labels[idx], float(probs[idx])

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "version":      self.VERSION,
            "model_type":   "mlp_py",
            "class_labels": self.class_labels,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std":  self.feature_std.tolist(),
            "w1":           self.w1.tolist(),
            "b1":           self.b1.tolist(),
            "bn_gamma":     self.bn_gamma.tolist(),
            "bn_beta":      self.bn_beta.tolist(),
            "bn_run_mean":  self.bn_run_mean.tolist(),
            "bn_run_var":   self.bn_run_var.tolist(),
            "w2":           self.w2.tolist(),
            "b2":           self.b2.tolist(),
            "temperature":  self.temperature,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "MlpGestureModel | None":
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except Exception:
            return None

    @classmethod
    def from_dict(cls, obj: dict) -> "MlpGestureModel | None":
        try:
            if obj.get("model_type") != "mlp_py":
                return None

            def fa(key: str) -> NDArray[np.float32]:
                return np.array(obj[key], dtype=np.float32)

            return cls(
                class_labels=list(obj["class_labels"]),
                feature_mean=fa("feature_mean"),
                feature_std=fa("feature_std"),
                w1=fa("w1"),
                b1=fa("b1"),
                bn_gamma=fa("bn_gamma"),
                bn_beta=fa("bn_beta"),
                bn_run_mean=fa("bn_run_mean"),
                bn_run_var=fa("bn_run_var"),
                w2=fa("w2"),
                b2=fa("b2"),
                temperature=float(obj.get("temperature", 1.0)),
            )
        except Exception:
            return None
