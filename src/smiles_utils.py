from typing import Optional
from rdkit import Chem


def to_canonical(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol)
    return canonical_smiles


def is_canonical(smiles: str) -> Optional[bool]:
    canonical = to_canonical(smiles)
    if canonical is None:
        return None
    return canonical == smiles
