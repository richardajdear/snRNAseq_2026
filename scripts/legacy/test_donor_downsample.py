
import sys
import os
import argparse
from unittest.mock import MagicMock

# Add code directory to path
sys.path.append(os.path.join(os.getcwd(), 'code'))

import downsample

# Mock args
class MockArgs:
    def __init__(self):
        self.input = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort_10k.h5ad"
        self.output = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort_10k_PFC_lessOld.h5ad"
        self.dataset_type = "Aging"
        self.pfc_only = True
        self.age_downsample = True
        self.n_cells = None
        self.seed = 42

def run_test():
    # We can't easily refactor main() to accept args object without changing signature.
    # But we can monkeypatch argparse or sys.argv
    sys.argv = [
        "downsample.py",
        "--input", "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort_10k.h5ad",
        "--output", "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort_10k_PFC_lessOld.h5ad",
        "--dataset_type", "Aging",
        "--pfc_only",
        "--age_downsample"
    ]
    
    print("Running downsample.main() via python script...")
    try:
        downsample.main()
        print("Success!")
    except SystemExit as e:
        print(f"Exited with code: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_test()
