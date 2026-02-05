import shutil
import os

# Artifact Directory
artifact_dir = "/home/rajd2/.gemini/antigravity/brain/2c752f5a-f4d3-4322-be8b-8616a8089d18"
files = [
    ("/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/combined_full/donor_age_distribution.png", "donor_age_distribution.png"),
    ("/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/combined_full/cell_age_histogram.png", "cell_age_histogram.png")
]

for src, name in files:
    dst = os.path.join(artifact_dir, name)
    try:
        shutil.copy(src, dst)
        print(f"Copied {src} to {dst}")
    except Exception as e:
        print(f"Failed to copy {name}: {e}")
