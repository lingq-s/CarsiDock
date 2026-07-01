from rdkit import Chem


def safe_remove_hs(mol, sanitize=True):
    if mol is None:
        return None
    try:
        return Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception:
        try:
            Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            return Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            return mol


def safe_remove_all_hs(mol, sanitize=True):
    if mol is None:
        return None
    try:
        return Chem.RemoveAllHs(mol, sanitize=sanitize)
    except Exception:
        try:
            Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            return Chem.RemoveAllHs(mol, sanitize=False)
        except Exception:
            return mol
