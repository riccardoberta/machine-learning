from __future__ import annotations

import numpy as np

def _activation_forward(z: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(0.0, z)
    if activation == "tanh":
        return np.tanh(z)
    raise ValueError(f"Unsupported activation: {activation!r} (use 'relu' or 'tanh').")


def _activation_backward(dh: np.ndarray, z: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return dh * (z > 0.0)
    if activation == "tanh":
        t = np.tanh(z)
        return dh * (1.0 - t * t)
    raise ValueError(f"Unsupported activation: {activation!r} (use 'relu' or 'tanh').")


def rnn_cell_forward(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_x: np.ndarray,
    W_h: np.ndarray,
    b: np.ndarray,
    *,
    activation: str = "tanh",
) -> tuple[np.ndarray, np.ndarray]:
    z_t = x_t @ W_x + h_prev @ W_h + b
    h_t = _activation_forward(z_t, activation)
    return h_t, z_t


def rnn_cell_unroll(
    X: np.ndarray,
    W_x: np.ndarray,
    W_h: np.ndarray,
    b: np.ndarray,
    *,
    h0: np.ndarray | None = None,
    activation: str = "tanh",
) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim != 3:
        raise ValueError(f"X must have shape (T, B, input_dim), got {X.shape}.")
    T, B, _ = X.shape
    hidden_dim = b.shape[0]

    h_prev = np.zeros((B, hidden_dim), dtype=X.dtype) if h0 is None else np.asarray(h0)
    if h_prev.shape != (B, hidden_dim):
        raise ValueError(f"h0 must have shape {(B, hidden_dim)}, got {h_prev.shape}.")

    H = np.zeros((T, B, hidden_dim), dtype=X.dtype)
    Z = np.zeros_like(H)
    for t in range(T):
        H[t], Z[t] = rnn_cell_forward(X[t], h_prev, W_x, W_h, b, activation=activation)
        h_prev = H[t]
    return H, Z

def init_rnn_layer(input_dim, hidden_dim, output_dim, seed=0):
    return init_rnn_layer_with_activation(
        input_dim, hidden_dim, output_dim, activation="tanh", seed=seed
    )


def init_rnn_layer_with_activation(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    activation: str = "tanh",
    seed: int = 0,
    dtype: np.dtype | str = np.float64,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def init_weight(fan_in: int, shape: tuple[int, int]) -> np.ndarray:
        if activation == "relu":
            std = np.sqrt(2.0 / fan_in)
        elif activation == "tanh":
            std = np.sqrt(1.0 / fan_in)
        else:
            raise ValueError(
                f"Unsupported activation: {activation!r} (use 'relu' or 'tanh')."
            )
        return rng.normal(0.0, std, size=shape).astype(dtype, copy=False)

    params = {
        "W_x": init_weight(input_dim, (input_dim, hidden_dim)),
        "W_h": init_weight(hidden_dim, (hidden_dim, hidden_dim)),
        "b": np.zeros(hidden_dim, dtype=dtype),
        "W_y": rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, output_dim)).astype(
            dtype, copy=False
        ),
        "b_y": np.zeros(output_dim, dtype=dtype),
    }
    return params


def rnn_layer_forward(
    X: np.ndarray, params: dict[str, np.ndarray], *, h0: np.ndarray | None = None, activation: str = "tanh"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.ndim != 3:
        raise ValueError(f"X must have shape (T, B, input_dim), got {X.shape}.")
    T, B, _ = X.shape

    hidden_dim = params["b"].shape[0]
    output_dim = params["b_y"].shape[0]

    h_prev = np.zeros((B, hidden_dim), dtype=X.dtype) if h0 is None else np.asarray(h0)
    if h_prev.shape != (B, hidden_dim):
        raise ValueError(f"h0 must have shape {(B, hidden_dim)}, got {h_prev.shape}.")

    H = np.zeros((T, B, hidden_dim), dtype=X.dtype)
    Z = np.zeros_like(H)
    Y_hat = np.zeros((T, B, output_dim), dtype=X.dtype)

    for t in range(T):
        H[t], Z[t] = rnn_cell_forward(
            X[t], h_prev, params["W_x"], params["W_h"], params["b"], activation=activation
        )
        Y_hat[t] = H[t] @ params["W_y"] + params["b_y"]
        h_prev = H[t]

    return H, Z, Y_hat, h0 if h0 is not None else np.zeros((B, hidden_dim), dtype=X.dtype)

def mse_loss(Y_hat, Y):
    return np.mean((Y_hat - Y) ** 2)

def rnn_layer_backward(
    X: np.ndarray,
    Y: np.ndarray,
    H: np.ndarray,
    Z: np.ndarray,
    Y_hat: np.ndarray,
    params: dict[str, np.ndarray],
    *,
    h0: np.ndarray | None = None,
    activation: str = "tanh",
) -> dict[str, np.ndarray]:

    T, B, _ = X.shape
    hidden_dim = params["W_h"].shape[0]
    output_dim = params["W_y"].shape[1]

    grads = {
        "W_x": np.zeros_like(params["W_x"]),
        "W_h": np.zeros_like(params["W_h"]),
        "b":   np.zeros_like(params["b"]),
        "W_y": np.zeros_like(params["W_y"]),
        "b_y": np.zeros_like(params["b_y"]),
    }

    h0_used = np.zeros((B, hidden_dim), dtype=X.dtype) if h0 is None else np.asarray(h0)
    if h0_used.shape != (B, hidden_dim):
        raise ValueError(f"h0 must have shape {(B, hidden_dim)}, got {h0_used.shape}.")

    dH_next = np.zeros((B, hidden_dim))

    denom = T * B * output_dim
    for t in reversed(range(T)):

        dY = 2.0 * (Y_hat[t] - Y[t]) / denom

        grads["W_y"] += np.dot(H[t].T, dY)
        grads["b_y"] += dY.sum(axis=0)

        dH = np.dot(dY, params["W_y"].T) + dH_next

        dZ = _activation_backward(dH, Z[t], activation)

        grads["b"]   += dZ.sum(axis=0)
        grads["W_x"] += np.dot(X[t].T, dZ)

        H_prev = H[t - 1] if t > 0 else h0_used
        grads["W_h"] += np.dot(H_prev.T, dZ)

        dH_next = np.dot(dZ, params["W_h"].T)

    return grads

def clip_grad_norm_(grads: dict[str, np.ndarray], max_norm: float) -> float:
    if max_norm <= 0:
        raise ValueError("max_norm must be > 0.")
    total_sq = 0.0
    for g in grads.values():
        total_sq += float(np.sum(g * g))
    norm = float(np.sqrt(total_sq))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for k in grads:
            grads[k] *= scale
    return norm


def sgd_step(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float = 1e-3) -> None:
    for k, v in params.items():
        v -= lr * grads[k]


def init_adam_state(params: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "t": 0,
        "m": {k: np.zeros_like(v) for k, v in params.items()},
        "v": {k: np.zeros_like(v) for k, v in params.items()},
    }


def adam_step(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: dict[str, object],
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    state["t"] = int(state.get("t", 0)) + 1
    t = int(state["t"])
    m = state["m"]
    v = state["v"]

    for k in params:
        m_k = m[k] = beta1 * m[k] + (1.0 - beta1) * grads[k]
        v_k = v[k] = beta2 * v[k] + (1.0 - beta2) * (grads[k] * grads[k])
        m_hat = m_k / (1.0 - beta1**t)
        v_hat = v_k / (1.0 - beta2**t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def rnn_layer_train_step(
    X: np.ndarray,
    Y: np.ndarray,
    params: dict[str, np.ndarray],
    *,
    lr: float = 1e-3,
    activation: str = "tanh",
    h0: np.ndarray | None = None,
    clip_norm: float | None = None,
    optimizer: str = "sgd",
    opt_state: dict[str, object] | None = None,
) -> float:
    H, Z, Y_hat, h0_used = rnn_layer_forward(X, params, h0=h0, activation=activation)
    loss = mse_loss(Y_hat, Y)
    grads = rnn_layer_backward(X, Y, H, Z, Y_hat, params, h0=h0_used, activation=activation)
    if clip_norm is not None:
        clip_grad_norm_(grads, clip_norm)
    if optimizer == "sgd":
        sgd_step(params, grads, lr=lr)
    elif optimizer == "adam":
        if opt_state is None:
            raise ValueError("opt_state is required when optimizer='adam'.")
        adam_step(params, grads, opt_state, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer!r} (use 'sgd' or 'adam').")
    return loss


def make_sequences(series, N):
    
    # Sequences
    X, Y = [], []

    # Create sequences
    for i in range(len(series) - N):

        # Append input sequence
        X.append(series[i:i+N])

        # Append target sequence
        Y.append(series[i+1:i+N+1])

    # Return arrays of sequences
    return np.array(X)[..., None], np.array(Y)[..., None]


def make_sequences_xy(features: np.ndarray, targets: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"features must have shape (T, D), got {features.shape}.")
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape (T,), got {targets.shape}.")
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length.")

    X, Y = [], []
    for i in range(len(targets) - seq_len):
        X.append(features[i : i + seq_len])
        Y.append(targets[i + 1 : i + seq_len + 1])
    X_arr = np.asarray(X)
    Y_arr = np.asarray(Y)[..., None]
    return X_arr, Y_arr


def standardize_fit(x: np.ndarray, *, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.maximum(std, eps)
    return mean, std


def standardize_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def iterate_minibatches(X: np.ndarray, Y: np.ndarray, batch_size: int, *, seed: int | None = None):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], Y[batch_idx]


def main() -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "This script section requires pandas; install it or import rnn.py as a module "
            "to use only the NumPy RNN utilities."
        ) from e

    df = pd.read_csv(
        "./data/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv",
        parse_dates=["service_date"],
    )
    df.columns = ["date", "day_type", "bus", "rail", "total"]
    df = df.sort_values("date").set_index("date")
    df = df.drop("total", axis=1)
    df = df.drop_duplicates()

    start_date = "2019-03-01"
    end_date = "2019-05-31"

    # Baselines (in original scale)
    eval_true = df["rail"][start_date:end_date]
    seasonal_naive = df["rail"].shift(7)[start_date:end_date]
    mae_seasonal_naive = (eval_true - seasonal_naive).abs().mean()
    print(f"Seasonal naive (t-7) MAE: {mae_seasonal_naive:.2f}")

    # Scale to millions to keep values small, then standardize based on training only.
    rail = (df["rail"].astype(float) / 1e6).to_numpy()

    dow = df.index.dayofweek.to_numpy()
    dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
    dow_cos = np.cos(2.0 * np.pi * dow / 7.0)

    day_type_dummies = pd.get_dummies(df["day_type"], prefix="day_type", drop_first=False)
    features = np.column_stack([rail, dow_sin, dow_cos, day_type_dummies.to_numpy(dtype=float)])

    train_mask = (df.index >= "2016-01-01") & (df.index <= "2018-12-31")
    valid_mask = (df.index >= "2019-01-01") & (df.index <= "2019-04-30")
    test_mask = (df.index >= start_date) & (df.index <= end_date)

    feat_mean, feat_std = standardize_fit(features[train_mask])
    features_std = standardize_transform(features, feat_mean, feat_std)

    # Predict rail(t+1); for convenience target is the standardized rail feature column.
    target_std = features_std[:, 0]

    seq_len = 28
    X_train, Y_train = make_sequences_xy(features_std[train_mask], target_std[train_mask], seq_len)
    X_valid, Y_valid = make_sequences_xy(features_std[valid_mask], target_std[valid_mask], seq_len)
    X_test, Y_test = make_sequences_xy(features_std[test_mask], target_std[test_mask], seq_len)

    input_dim = X_train.shape[-1]
    hidden_dim = 20
    output_dim = 1
    activation = "tanh"

    params = init_rnn_layer_with_activation(
        input_dim, hidden_dim, output_dim, activation=activation, seed=42, dtype=np.float64
    )

    epochs = 20
    batch_size = 32
    lr = 3e-3
    clip_norm = 1.0
    optimizer = "adam"
    opt_state = init_adam_state(params) if optimizer == "adam" else None

    for epoch in range(epochs):
        losses = []
        for Xb, Yb in iterate_minibatches(X_train, Y_train, batch_size, seed=epoch):
            Xb = Xb.transpose(1, 0, 2)  # (T, B, D)
            Yb = Yb.transpose(1, 0, 2)  # (T, B, 1)
            loss = rnn_layer_train_step(
                Xb,
                Yb,
                params,
                lr=lr,
                activation=activation,
                clip_norm=clip_norm,
                optimizer=optimizer,
                opt_state=opt_state,
            )
            losses.append(loss)
        print(f"Epoch {epoch + 1:02d} | Train loss: {np.mean(losses):.6f}")

    # Evaluate one-step-ahead predictions over the test interval.
    # Map predictions back to original rail scale:
    # rail_scaled = rail_million => standardize => y_std; so y_million = y_std * std + mean.
    rail_mean = feat_mean[0]
    rail_std = feat_std[0]
    test_positions = np.flatnonzero(test_mask)
    y_preds = []
    for idx in test_positions:
        x_window = features_std[idx - seq_len : idx]  # (T, D)
        x_window = x_window[None, :, :].transpose(1, 0, 2)  # (T, 1, D)
        _, _, Y_hat, _ = rnn_layer_forward(x_window, params, activation=activation)
        y_preds.append(float(Y_hat[-1, 0, 0]))

    y_pred_std = np.asarray(y_preds)
    y_pred_million = y_pred_std * rail_std + rail_mean
    y_pred_unscaled = y_pred_million * 1e6

    y_true_unscaled = df["rail"][start_date:end_date].values
    mae_rnn = np.mean(np.abs(y_pred_unscaled - y_true_unscaled))
    print(f"RNN MAE: {mae_rnn:.2f}")

if __name__ == "__main__":
    main()
