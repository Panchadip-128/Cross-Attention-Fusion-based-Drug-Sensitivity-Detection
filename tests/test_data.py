import pytest
from src.data.split import murcko_scaffold_split
import pandas as pd

def test_murcko_scaffold_split():
    # Mock a small dataframe with SMILES strings
    data = {
        'SMILES': [
            'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5', # Imatinib
            'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5', # Same scaffold
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine (different scaffold)
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
            'CC(=O)OC1=CC=CC=C1C(=O)O' # Aspirin (different scaffold)
        ],
        'IC50': [1.0, 1.2, 0.5, 0.6, 5.0]
    }
    df = pd.DataFrame(data)
    
    # We should get exactly 3 unique scaffolds
    train_df, val_df, test_df = murcko_scaffold_split(df, smiles_col='SMILES', frac_train=0.6, frac_val=0.2, frac_test=0.2, seed=42)
    
    # Check that they are disjoint sets of SMILES
    train_smiles = set(train_df['SMILES'])
    val_smiles = set(val_df['SMILES'])
    test_smiles = set(test_df['SMILES'])
    
    assert train_smiles.isdisjoint(val_smiles)
    assert train_smiles.isdisjoint(test_smiles)
    assert val_smiles.isdisjoint(test_smiles)
    
    # Ensure total rows equal original
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
