import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metadata_utils import get_original_metadata

if __name__ == "__main__":
    print("Running metadata extraction...")
    meta = get_original_metadata()
    out_path = '/home/rajd2/rds/hpc-work/snRNAseq_2026/results/metadata_utils_meta.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta.to_csv(out_path)
    print(f"Saved metadata to {out_path} with shape {meta.shape}")
