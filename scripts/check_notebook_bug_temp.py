import scanpy as sc
import numpy as np

RDS = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
DATA_FILE = f"{RDS}/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld/scvi_output/integrated.h5ad"

print("Loading data...")
adata = sc.read_h5ad(DATA_FILE, backed="r")

print(f"Is adata.X identical to adata.layers['counts']? {np.all(adata.X[0:10].toarray() == adata.layers['counts'][0:10].toarray())}")

raw_counts = adata.X[0:10].toarray()
scvi_counts = adata.layers['scvi_normalized'][0:10]

print("Adata.X max:", raw_counts.max())
print("scvi_normalized max:", scvi_counts.max())
