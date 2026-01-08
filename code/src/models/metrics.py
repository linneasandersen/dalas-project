import numpy as np
from sklearn.metrics import mean_squared_error


def compute_mape_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    nonzero_mask = y_true != 0
    mape = (
        np.mean(
            np.abs(
                (y_true[nonzero_mask] - y_pred[nonzero_mask])
                / y_true[nonzero_mask]
            )
        )
        * 100
    )

    smape = (
        np.mean(
            2 * np.abs(y_pred - y_true)
            / (np.abs(y_true) + np.abs(y_pred))
        )
        * 100
    )

    return mape, smape


def factor_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    
    mask = (y_true != 0) & (y_pred != 0)
    ratio = np.maximum(y_pred[mask] / y_true[mask],
                   y_true[mask] / y_pred[mask])

    return np.exp(np.mean(np.log(ratio)))


def evaluate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Clip to avoid log1p issues
    y_true_clip = np.clip(y_true, 0, None)
    y_pred_clip = np.clip(y_pred, 0, None)

    mape, smape = compute_mape_smape(y_true_clip, y_pred_clip)

    return {
        "MAPE": mape,
        "SMAPE": smape,
        "RMSE": np.sqrt(mean_squared_error(y_true_clip, y_pred_clip)),
        "RMSE_log": np.sqrt(
            mean_squared_error(np.log1p(y_true_clip), np.log1p(y_pred_clip))
        ),
        "Factor_Error": factor_error(y_true_clip, y_pred_clip),
    }

