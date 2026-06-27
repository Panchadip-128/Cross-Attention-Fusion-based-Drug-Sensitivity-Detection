import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_det_batched(model, data, device, batch_size=8192):
    """
    Batched deterministic evaluation to avoid GPU OOM.
    """
    model.eval()
    preds_all = []
    n = data.x.size(0)

    from ..data.graph import make_batch

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = torch.arange(start, end, device=device)
            batch = make_batch(data, idx)
            preds = model(batch)
            preds_all.append(preds.detach().cpu())

    y_pred = torch.cat(preds_all).numpy()
    y_true = data.y.detach().cpu().numpy()

    metrics = {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }

    model.train()
    return metrics

def mc_dropout_predict(model, data, device, n_passes=50, batch_size=8192):
    """
    Run MC Dropout for uncertainty quantification.
    """
    model.train() # Keep dropout active
    all_pass_preds = []

    for _ in range(n_passes):
        preds_all = []
        n = data.x.size(0)
        from ..data.graph import make_batch
        
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = torch.arange(start, end, device=device)
                batch = make_batch(data, idx)
                preds = model(batch)
                preds_all.append(preds.detach().cpu())
                
        all_pass_preds.append(torch.cat(preds_all).numpy())

    all_pass_preds = np.stack(all_pass_preds, axis=0) # [n_passes, N]
    mean_preds = np.mean(all_pass_preds, axis=0)
    std_preds = np.std(all_pass_preds, axis=0)

    model.eval() # Restore deterministic eval mode
    return mean_preds, std_preds
