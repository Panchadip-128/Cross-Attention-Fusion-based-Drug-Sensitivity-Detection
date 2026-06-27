import os
import argparse
import torch
from src.data.loader import load_and_merge_gdsc
from src.data.preprocess import encode_categorical, impute_missing, scale_features
from src.data.split import scaffold_blind_split
from src.data.graph import build_data
from src.models.architectures import CrossAttentionDrugModel
from src.training.evaluate import evaluate_det_batched, mc_dropout_predict
from src.utils.plotting import plot_predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cross-Attention Model")
    parser.add_argument('--model_path', type=str, default='results/best_model.pth', help='Path to saved model')
    parser.add_argument('--gdsc1', type=str, default='../GDSC1.csv', help='Path to GDSC1.csv')
    parser.add_argument('--gdsc2', type=str, default='../GDSC2.csv', help='Path to GDSC2.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    try:
        df = load_and_merge_gdsc(args.gdsc1, args.gdsc2)
    except FileNotFoundError:
        print("Data files not found.")
        return

    CATEGORICAL_COLS = ['Drug name', 'Feature Name', 'Tissue Type', 'Screening Set']
    df, label_encoders = encode_categorical(df, CATEGORICAL_COLS)
    
    FEATURE_COLS = [
        'n_feature_pos', 'log_ic50_mean_pos', 'log_ic50_mean_neg',
        'feature_pos_ic50_var', 'feature_neg_ic50_var', 'feature_delta_mean_ic50',
        'feature_ic50_t_pval', 'feature_pval', 'tissue_pval', 'msi_pval'
    ]
    df = impute_missing(df, FEATURE_COLS + ['ic50_effect_size'])
    _, _, df_test = scaffold_blind_split(df, label_encoders['Drug name'])

    # Get scaler from entire df for evaluation (in a real scenario, load saved scaler)
    df_train, df_val, _ = scaffold_blind_split(df, label_encoders['Drug name'])
    _, _, (X_test, y_test, ids_test), _ = scale_features(
        df_train, df_val, df_test, FEATURE_COLS, 'ic50_effect_size'
    )
    
    data_test = build_data(X_test, y_test, ids_test).to(device)

    # 2. Load Model
    num_drugs = df['Drug name'].nunique()
    model = CrossAttentionDrugModel(
        input_dim=len(FEATURE_COLS), hidden_dim=64, num_drugs=num_drugs,
        n_heads=4, lstm_hidden=32, dropout=0.1
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Evaluate
    print("Running deterministic evaluation...")
    metrics = evaluate_det_batched(model, data_test, device)
    print(f"Test R2: {metrics['R2']:.4f} | Test MSE: {metrics['MSE']:.5f} | Test MAE: {metrics['MAE']:.5f}")

    print("Running MC Dropout Uncertainty Quantification (50 passes)...")
    mean_preds, std_preds = mc_dropout_predict(model, data_test, device, n_passes=50)
    print(f"Mean Uncertainty (σ): {std_preds.mean():.4f}")

    # Plot
    plot_predictions(y_test, mean_preds, save_path='results/test_predictions.png')
    print("Evaluation complete. Plot saved to results/test_predictions.png")

if __name__ == '__main__':
    main()
