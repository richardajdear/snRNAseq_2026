import os


def get_environment():
    """Detect local vs HPC and return canonical paths.

    Returns a dict with keys: name, rds_dir, code_dir, ref_dir.

    Detection: if the repo root contains 'rds-cam-psych-transc-Pb9UGUlrwWc'
    (the locally-mounted RDS volume) we are local; otherwise HPC.
    """
    code_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(code_dir)

    is_local = os.path.isdir(os.path.join(repo_root, 'rds-cam-psych-transc-Pb9UGUlrwWc'))

    if is_local:
        name         = 'local'
        rds_dir      = os.path.join(repo_root, 'rds-cam-psych-transc-Pb9UGUlrwWc')
        code_dir_out = code_dir
        ref_dir      = os.path.join(repo_root, 'reference')
    else:
        name         = 'hpc'
        _base        = '/home/rajd2/rds'
        rds_dir      = os.path.join(_base, 'rds-cam-psych-transc-Pb9UGUlrwWc')
        code_dir_out = os.path.join(_base, 'hpc-work/snRNAseq_2026/code')
        ref_dir      = os.path.join(_base, 'hpc-work/snRNAseq_2026/reference')

    return dict(name=name, rds_dir=rds_dir, code_dir=code_dir_out, ref_dir=ref_dir)
