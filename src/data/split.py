import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 15 GDSC drugs SMILES
GDSC_SMILES = {
    'Erlotinib': 'COCCOC1=CC2=C(C=C1OCCOC)C(=NC=N2)NC3=CC=CC(=C3)C#C',
    'Gefitinib': 'COC1=CC2=C(C=C1OCCCN3CCOCC3)C(=NC=N2)NC4=CC(=C(C=C4)F)Cl',
    'Lapatinib': 'CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1',
    'Afatinib': 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1',
    'Imatinib': 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1',
    'Dasatinib': 'Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1',
    'Nilotinib': 'Cc1ccc(C(=O)Nc2ccc(C)c(Nc3nccc(-c4cccnc4)n3)c2)cc1C(F)(F)F',
    'Sorafenib': 'CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1',
    'Sunitinib': 'CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\C(=O)Nc3ccc(F)cc32)c1C',
    'Vemurafenib': 'CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1F',
    'Selumetinib': 'Cc1cc2c(Nc3ccc(I)c(F)c3F)[n]c(=O)[n]c2c(Cl)c1',
    'Paclitaxel': 'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(=C2OC(=O)C5=CC=CC=C5)C)OC(=O)C)(CO4)OC(=O)C)O)C)OC(=O)C6=CC=CC=C6',
    'Camptothecin': 'CCC1(O)C(=O)OCc2c1cc1ccc3ccnc4ccc(c2)c1c34',
    'Temsirolimus': 'C[C@@H]1CC[C@H]2C[C@@H](/C(=C/[C@@H]3CC(=O)[C@H](C/C(=C/[C@H](C(=O)[C@@H](OC(=O)[C@@H]([C@@H](C[C@@H]([C@@H]1OC)O)OC)CC(=O)O2)C)C)C)C3=O)O)OC',
    'Vorinostat': 'O=C(CCCCCCC(=O)Nc1ccccc1)NO',
}

def get_scaffold(smiles: str, generic: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
    else:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold else ''

def scaffold_blind_split(df: pd.DataFrame, drug_le, test_size=0.20, val_size=0.20, random_state=42):
    """
    Perform a partial scaffold-blind split.
    """
    drug_scaffolds = {name: get_scaffold(smi) for name, smi in GDSC_SMILES.items()}
    name_to_id = {name: idx for idx, name in enumerate(drug_le.classes_)}
    
    drug_id_to_scaffold = {}
    for drug_name, encoded_id in name_to_id.items():
        if drug_name in drug_scaffolds and drug_scaffolds[drug_name]:
            drug_id_to_scaffold[encoded_id] = drug_scaffolds[drug_name]
        else:
            drug_id_to_scaffold[encoded_id] = f'__unknown_{encoded_id}__'
            
    scaffold_to_drug_ids = defaultdict(list)
    for drug_id, scaffold in drug_id_to_scaffold.items():
        scaffold_to_drug_ids[scaffold].append(drug_id)
        
    all_scaffolds = list(scaffold_to_drug_ids.keys())
    
    train_scaffolds, test_scaffolds = train_test_split(
        all_scaffolds, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    final_train_scaffolds, val_scaffolds = train_test_split(
        train_scaffolds, test_size=val_size, random_state=random_state, shuffle=True
    )
    
    final_train_drugs = {drug_id for sc in final_train_scaffolds for drug_id in scaffold_to_drug_ids[sc]}
    val_drugs = {drug_id for sc in val_scaffolds for drug_id in scaffold_to_drug_ids[sc]}
    test_drugs = {drug_id for sc in test_scaffolds for drug_id in scaffold_to_drug_ids[sc]}
    
    df_train = df[df['Drug name'].isin(final_train_drugs)].reset_index(drop=True)
    df_val = df[df['Drug name'].isin(val_drugs)].reset_index(drop=True)
    df_test = df[df['Drug name'].isin(test_drugs)].reset_index(drop=True)
    
    return df_train, df_val, df_test
