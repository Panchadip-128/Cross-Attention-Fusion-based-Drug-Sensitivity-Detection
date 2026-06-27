import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_training_diagnostics(history: dict, save_path: str = None):
    """
    Plot training loss, validation MSE/MAE, and validation R2.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Training Loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Validation Errors
    axes[1].plot(history['val_mse'], label='Val MSE', color='green')
    axes[1].plot(history['val_mae'], label='Val MAE', color='orange')
    axes[1].set_title('Validation Errors')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Error')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Validation R2
    axes[2].plot(history['val_r2'], label='Val R²', color='red')
    best_r2 = max(history['val_r2'])
    axes[2].axhline(y=best_r2, color='r', linestyle='--', alpha=0.5, label=f'Best R²: {best_r2:.4f}')
    axes[2].set_title('Validation R²')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('R² Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_predictions(y_true, y_pred, save_path: str = None):
    """
    Plot predicted vs actual IC50 and residual distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(y_true, y_pred, alpha=0.2, s=2)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[0].set_title('Predicted vs Actual IC50')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_title('Residual Distribution')
    axes[1].set_xlabel('Residual')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
