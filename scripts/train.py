import os
import argparse
import torch
import torch.nn as nn
from src.data.loader import load_and_merge_gdsc
from src.data.preprocess import encode_categorical, impute_missing, scale_features
from src.data.split import scaffold_blind_split
from src.data.graph import build_data
from src.models.architectures import CrossAttentionDrugModel
from src.training.trainer import train_model
from src.utils.seed import set_seed
from src.utils.plotting import plot_training_diagnostics, plot_predictions

def main():
    parser = argparse.ArgumentParser(description="Train Cross-Attention Drug-Genomic Fusion Model")
    parser.add_argument('--gdsc1', type=str, default='../GDSC1.csv', help='Path to GDSC1.csv')
    parser.add_argument('--gdsc2', type=str, default='../GDSC2.csv', help='Path to GDSC2.csv')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs to train')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # For stable math attention
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # 1. Load Data
    print("Loading datasets...")
    try:
        df = load_and_merge_gdsc(args.gdsc1, args.gdsc2)
    except FileNotFoundError:
        print(f"Error: Could not find GDSC CSV files at {args.gdsc1} or {args.gdsc2}.")
        print("Please ensure the data files are present or provide the correct paths.")
        return

    # 2. Preprocess
    CATEGORICAL_COLS = ['Drug name', 'Feature Name', 'Tissue Type', 'Screening Set']
    df, label_encoders = encode_categorical(df, CATEGORICAL_COLS)
    
    FEATURE_COLS = [
        'n_feature_pos', 'log_ic50_mean_pos', 'log_ic50_mean_neg',
        'feature_pos_ic50_var', 'feature_neg_ic50_var', 'feature_delta_mean_ic50',
        'feature_ic50_t_pval', 'feature_pval', 'tissue_pval', 'msi_pval'
    ]
    df = impute_missing(df, FEATURE_COLS + ['ic50_effect_size'])

    # 3. Split
    print("Performing scaffold-blind split...")
    df_train, df_val, df_test = scaffold_blind_split(df, label_encoders['Drug name'])

    # 4. Scale and Build PyG Data
    (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, y_test, ids_test), scaler = scale_features(
        df_train, df_val, df_test, FEATURE_COLS, 'ic50_effect_size'
    )
    
    data_train = build_data(X_train, y_train, ids_train).to(device)
    data_val = build_data(X_val, y_val, ids_val).to(device)
    data_test = build_data(X_test, y_test, ids_test).to(device)

    # 5. Initialize Model
    num_drugs = df['Drug name'].nunique()
    model = CrossAttentionDrugModel(
        input_dim=len(FEATURE_COLS),
        hidden_dim=64,
        num_drugs=num_drugs,
        n_heads=4,
        lstm_hidden=32,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    # 6. Train
    model, history, best_epoch = train_model(
        model, data_train, data_val, optimizer, scheduler, criterion, device,
        max_epochs=args.epochs, patience=20, clip_norm=1.0, batch_size=args.batch_size
    )

    # Save
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/best_model.pth')
    plot_training_diagnostics(history, save_path='results/training_diagnostics.png')
    print("Training complete. Results saved in 'results/' directory.")

if __name__ == '__main__':
    main()
