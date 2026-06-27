import torch
import torch.nn as nn
import numpy as np
import copy
from .evaluate import evaluate_det_batched

def train_model(
    model: nn.Module,
    data_train,
    data_val,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion,
    device: torch.device,
    max_epochs: int = 200,
    patience: int = 20,
    clip_norm: float = 1.0,
    batch_size: int = 8192,
):
    """
    Trains the model with early stopping based on validation R2 score.
    """
    history = {
        'train_loss': [],
        'val_r2': [],
        'val_mse': [],
        'val_mae': [],
        'lr': [],
    }

    best_val_r2 = -float('inf')
    best_state = None
    best_epoch = 0
    patience_count = 0
    n_train = data_train.x.size(0)

    print(f"\nStarting training: max_epochs={max_epochs} patience={patience} clip_norm={clip_norm}")
    print("─" * 80)
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val R²':>8}  {'Val MSE':>9}  {'Val MAE':>9}  {'LR':>10}")
    print("─" * 80)

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_losses = []

        from ..data.graph import make_batch

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            batch = make_batch(data_train, idx)
            optimizer.zero_grad(set_to_none=True)

            train_preds = model(batch)
            train_loss = criterion(train_preds, batch.y)
            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            epoch_losses.append(train_loss.item())

        scheduler.step()
        avg_train_loss = float(np.mean(epoch_losses))
        current_lr = scheduler.get_last_lr()[0]

        val_metrics = evaluate_det_batched(model, data_val, device, batch_size=batch_size)
        val_r2, val_mse, val_mae = val_metrics['R2'], val_metrics['MSE'], val_metrics['MAE']

        history['train_loss'].append(avg_train_loss)
        history['val_r2'].append(val_r2)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            patience_count = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            marker = " ◀ best" if epoch == best_epoch else ""
            print(
                f"{epoch:>6}  {avg_train_loss:>11.5f}  {val_r2:>8.4f}  "
                f"{val_mse:>9.5f}  {val_mae:>9.5f}  {current_lr:>10.2e}{marker}"
            )

        if patience_count >= patience:
            print(f"\n⏹ Early stopping at epoch {epoch} (no validation R² improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch} with Validation R² = {best_val_r2:.4f}")

    return model, history, best_epoch
