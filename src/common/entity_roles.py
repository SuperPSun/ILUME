"""Charge-derived entity roles shared by downstream stages."""
from rdkit import Chem


def molecular_role(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid entity SMILES: {smiles}")
    charge = Chem.GetFormalCharge(molecule)
    return "cation" if charge > 0 else "anion" if charge < 0 else "neutral"
