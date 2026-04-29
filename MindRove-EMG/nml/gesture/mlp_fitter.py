from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from nml.gesture.mlp_model import MlpGestureModel


class MlpFitter:
    """
    Trains a two-layer MLP for gesture classification.

    Uses Adam + stratified 85/15 split + early stopping + LR plateau reduction
    + temperature calibration via golden-section search on the validation set.

    Returns None when there are fewer than 64 total samples or fewer than
    2 classes with at least 16 samples each.

    Mirrors the Kotlin MlpFitter architecture and hyperparameters.
    """

    HIDDEN = 128
    DROPOUT = 0.2
    LR = 1e-3
    EPOCHS = 200
    BATCH = 32
    BN_MOMENTUM = 0.1
    BN_EPS = 1e-5

    ADAM_B1 = 0.9
    ADAM_B2 = 0.999
    ADAM_EP = 1e-8

    WEIGHT_DECAY = 1e-4
    LR_PLATEAU_PATIENCE = 6
    LR_REDUCE_FACTOR = 0.5
    LR_MIN = 1e-5

    VAL_SPLIT = 0.15
    PATIENCE = 12
    PATIENCE_MIN_DELTA = 1e-4

    @staticmethod
    def fit(
        samples: list[tuple[NDArray[np.float32], int]],
        rng: np.random.Generator | None = None,
    ) -> MlpGestureModel | None:
        """
        Fit an MLP from windowed feature samples.

        samples — list of (feature_vector, label) pairs from the gesture extractor.
        Returns None when there is insufficient data.
        """
        if len(samples) < 64:
            return None
        if rng is None:
            rng = np.random.default_rng()

        feat_dim = len(samples[0][0])
        classes = sorted(set(s[1] for s in samples))
        if len(classes) < 2:
            return None
        K = len(classes)
        label_to_idx = {c: i for i, c in enumerate(classes)}
        N = len(samples)

        for c in classes:
            if sum(1 for s in samples if s[1] == c) < 16:
                return None

        Y = np.array([label_to_idx[s[1]] for s in samples], dtype=np.int32)
        X_raw = np.stack([s[0] for s in samples]).astype(np.float64)  # (N, feat_dim)

        # Stratified train / val split
        train_idx, val_idx = [], []
        for c in classes:
            c_idx = np.where(Y == label_to_idx[c])[0].tolist()
            rng.shuffle(c_idx)
            n_val = max(1, int(len(c_idx) * MlpFitter.VAL_SPLIT))
            val_idx.extend(c_idx[:n_val])
            train_idx.extend(c_idx[n_val:])
        train_arr = np.array(train_idx, dtype=np.int32)
        val_idx_arr = np.array(val_idx, dtype=np.int32)
        n_train = len(train_arr)

        # Input normalisation (training split only)
        mean = X_raw[train_arr].mean(axis=0)
        std = np.maximum(X_raw[train_arr].std(axis=0), 1e-6)
        X = (X_raw - mean) / std

        H = MlpFitter.HIDDEN
        he = lambda fan_in: np.sqrt(2.0 / fan_in)

        w1 = rng.standard_normal((H, feat_dim)) * he(feat_dim)
        b1 = np.zeros(H)
        bn_gamma = np.ones(H)
        bn_beta = np.zeros(H)
        bn_run_mean = np.zeros(H)
        bn_run_var = np.ones(H)
        w2 = rng.standard_normal((K, H)) * he(H)
        b2 = np.zeros(K)

        # Adam moment buffers
        mW1, vW1 = np.zeros_like(w1), np.zeros_like(w1)
        mB1, vB1 = np.zeros_like(b1), np.zeros_like(b1)
        mGm, vGm = np.zeros_like(bn_gamma), np.zeros_like(bn_gamma)
        mBt, vBt = np.zeros_like(bn_beta), np.zeros_like(bn_beta)
        mW2, vW2 = np.zeros_like(w2), np.zeros_like(w2)
        mB2, vB2 = np.zeros_like(b2), np.zeros_like(b2)

        adam_t = 0
        current_lr = MlpFitter.LR
        lr_best_val = np.inf
        lr_plateau_count = 0

        best_val_loss = np.inf
        patience_count = 0
        best = (w1.copy(), b1.copy(), bn_gamma.copy(), bn_beta.copy(),
                bn_run_mean.copy(), bn_run_var.copy(), w2.copy(), b2.copy())

        def adam_step(param, grad, m, v, t, lr, wd=0.0):
            g = grad + wd * param
            m[:] = MlpFitter.ADAM_B1 * m + (1 - MlpFitter.ADAM_B1) * g
            v[:] = MlpFitter.ADAM_B2 * v + (1 - MlpFitter.ADAM_B2) * g * g
            mh = m / (1 - MlpFitter.ADAM_B1 ** t)
            vh = v / (1 - MlpFitter.ADAM_B2 ** t)
            param -= lr * mh / (np.sqrt(np.maximum(vh, 0.0)) + MlpFitter.ADAM_EP)

        def val_loss(vw1, vb1, vgm, vbt, vrm, vrv, vw2, vb2, temp=1.0) -> float:
            if len(val_idx_arr) == 0:
                return 0.0
            a1_ = X[val_idx_arr] @ vw1.T + vb1                             # (V, H)
            h1_ = np.maximum(
                vgm * (a1_ - vrm) / np.sqrt(vrv + MlpFitter.BN_EPS) + vbt, 0.0
            )
            logits_ = (h1_ @ vw2.T + vb2) / max(temp, 1e-6)               # (V, K)
            logits_ -= logits_.max(axis=1, keepdims=True)
            exp_l = np.exp(logits_)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            correct_probs = probs[np.arange(len(val_idx_arr)), Y[val_idx_arr]]
            return float(-np.log(np.maximum(correct_probs, 1e-10)).mean())

        for _epoch in range(MlpFitter.EPOCHS):
            rng.shuffle(train_arr)
            start = 0
            while start < n_train:
                end = min(start + MlpFitter.BATCH, n_train)
                bsz = end - start
                ids = train_arr[start:end]
                adam_t += 1

                # ── Forward ──────────────────────────────────────────────
                Xb = X[ids]                                     # (bsz, D)
                a1b = Xb @ w1.T + b1                            # (bsz, H)

                bn_mean = a1b.mean(axis=0)
                bn_var = a1b.var(axis=0)
                inv_std = 1.0 / np.sqrt(bn_var + MlpFitter.BN_EPS)
                a1hat = (a1b - bn_mean) * inv_std
                h1b = bn_gamma * a1hat + bn_beta                # (bsz, H)

                mask = rng.random((bsz, H)) >= MlpFitter.DROPOUT
                r1b = np.where((h1b > 0.0) & mask, h1b / (1.0 - MlpFitter.DROPOUT), 0.0)

                logits_b = r1b @ w2.T + b2                      # (bsz, K)
                max_l = logits_b.max(axis=1, keepdims=True)
                exp_l = np.exp(logits_b - max_l)
                probs_b = exp_l / exp_l.sum(axis=1, keepdims=True)

                # ── Softmax gradient ──────────────────────────────────────
                d_logits = probs_b.copy()
                d_logits[np.arange(bsz), Y[ids]] -= 1.0
                d_logits /= bsz

                # ── Backward w2, b2 ───────────────────────────────────────
                gW2 = d_logits.T @ r1b
                gB2 = d_logits.sum(axis=0)
                dR1 = d_logits @ w2                             # (bsz, H)

                # ── Backward dropout + ReLU ───────────────────────────────
                dH1 = np.where((h1b > 0.0) & mask, dR1 / (1.0 - MlpFitter.DROPOUT), 0.0)

                # ── Backward BatchNorm ────────────────────────────────────
                gGamma = (dH1 * a1hat).sum(axis=0)
                gBeta = dH1.sum(axis=0)
                dXhat = dH1 * bn_gamma
                dVar = (dXhat * (a1b - bn_mean) * (-0.5) * inv_std ** 3).sum(axis=0)
                dMean = (dXhat * (-inv_std)).sum(axis=0) + dVar * (-2.0 / bsz) * (a1b - bn_mean).sum(axis=0)
                dA1 = dXhat * inv_std + dVar * 2.0 * (a1b - bn_mean) / bsz + dMean / bsz

                gW1 = dA1.T @ Xb
                gB1 = dA1.sum(axis=0)

                # ── Adam updates ──────────────────────────────────────────
                adam_step(w1, gW1, mW1, vW1, adam_t, current_lr, MlpFitter.WEIGHT_DECAY)
                adam_step(b1, gB1, mB1, vB1, adam_t, current_lr)
                adam_step(bn_gamma, gGamma, mGm, vGm, adam_t, current_lr)
                adam_step(bn_beta, gBeta, mBt, vBt, adam_t, current_lr)
                adam_step(w2, gW2, mW2, vW2, adam_t, current_lr, MlpFitter.WEIGHT_DECAY)
                adam_step(b2, gB2, mB2, vB2, adam_t, current_lr)

                bn_run_mean[:] = (1 - MlpFitter.BN_MOMENTUM) * bn_run_mean + MlpFitter.BN_MOMENTUM * bn_mean
                bn_run_var[:] = (1 - MlpFitter.BN_MOMENTUM) * bn_run_var + MlpFitter.BN_MOMENTUM * bn_var

                start = end

            # ── Validation + early stopping ───────────────────────────────
            vl = val_loss(w1, b1, bn_gamma, bn_beta, bn_run_mean, bn_run_var, w2, b2)

            if vl < best_val_loss - MlpFitter.PATIENCE_MIN_DELTA:
                best_val_loss = vl
                patience_count = 0
                best = (w1.copy(), b1.copy(), bn_gamma.copy(), bn_beta.copy(),
                        bn_run_mean.copy(), bn_run_var.copy(), w2.copy(), b2.copy())
            else:
                patience_count += 1
                if patience_count >= MlpFitter.PATIENCE:
                    break

            # ── LR plateau reduction ──────────────────────────────────────
            if vl < lr_best_val - MlpFitter.PATIENCE_MIN_DELTA:
                lr_best_val = vl
                lr_plateau_count = 0
            else:
                lr_plateau_count += 1
                if lr_plateau_count >= MlpFitter.LR_PLATEAU_PATIENCE and current_lr > MlpFitter.LR_MIN:
                    current_lr = max(current_lr * MlpFitter.LR_REDUCE_FACTOR, MlpFitter.LR_MIN)
                    lr_plateau_count = 0

        bw1, bb1, bgm, bbt, brm, brv, bw2, bb2 = best

        # ── Temperature calibration (golden-section search on val set) ────
        temperature = 1.0
        if len(val_idx_arr) >= 4:
            phi = (np.sqrt(5.0) - 1.0) / 2.0
            t_lo, t_hi = 0.5, 5.0
            for _ in range(40):
                t1 = t_hi - phi * (t_hi - t_lo)
                t2 = t_lo + phi * (t_hi - t_lo)
                if (val_loss(bw1, bb1, bgm, bbt, brm, brv, bw2, bb2, t1) <=
                        val_loss(bw1, bb1, bgm, bbt, brm, brv, bw2, bb2, t2)):
                    t_hi = t2
                else:
                    t_lo = t1
            temperature = float((t_lo + t_hi) / 2.0)

        return MlpGestureModel(
            class_labels=classes,
            feature_mean=mean.astype(np.float32),
            feature_std=std.astype(np.float32),
            w1=bw1.astype(np.float32),
            b1=bb1.astype(np.float32),
            bn_gamma=bgm.astype(np.float32),
            bn_beta=bbt.astype(np.float32),
            bn_run_mean=brm.astype(np.float32),
            bn_run_var=np.maximum(brv, MlpFitter.BN_EPS).astype(np.float32),
            w2=bw2.astype(np.float32),
            b2=bb2.astype(np.float32),
            temperature=temperature,
        )
