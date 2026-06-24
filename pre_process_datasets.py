"""
One-shot helper that materialises every processed dataset (writes each
``processed/data.pt``). Run once after placing the raw files under
``Datasets/`` (see README). Subsequent runs of ``main.py`` load the cached
processed tensors directly.
"""
from config import DATASET_PATHS


def process_raw_data():
    from pre_processing import (
        EllipticDataset, IBMAMLDataset_HiSmall, IBMAMLDataset_LiSmall,
        IBMAMLDataset_HiMedium, IBMAMLDataset_LiMedium, AMLSimDataset,
    )

    EllipticDataset(root=DATASET_PATHS["Elliptic"])[0]
    IBMAMLDataset_HiSmall(root=DATASET_PATHS["IBM_AML_HiSmall"])[0]
    IBMAMLDataset_LiSmall(root=DATASET_PATHS["IBM_AML_LiSmall"])[0]
    IBMAMLDataset_HiMedium(root=DATASET_PATHS["IBM_AML_HiMedium"])[0]
    IBMAMLDataset_LiMedium(root=DATASET_PATHS["IBM_AML_LiMedium"])[0]
    AMLSimDataset(root=DATASET_PATHS["AMLSim"])[0]
    print("All pre_processing complete.")


if __name__ == "__main__":
    process_raw_data()
