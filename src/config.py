import os

# Dir paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPTS_DIR = os.path.join(BASE_DIR, "src")
PRETRAINED_MODELS_DIR = os.path.join(BASE_DIR, "pretrained_models")

# pMHC prediction tools paths
PREDIG_PATH = "/Users/alexascunceparis/Desktop/BSC/d_programs/PredIG"
NETMHCPAN_PATH = "/Users/alexascunceparis/Desktop/BSC/d_programs/netMHCpan-4.2/netMHCpan"

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
PDB_DIR = os.path.join(DATA_DIR, "pdbs")
PDB_INFO_CSV = os.path.join(DATA_DIR, "af3_training.csv")
DEFAULT_STAGING_DIR = os.path.join(BASE_DIR, "af3_staging")
CHAIN_INFO_PATH= os.path.join(DATA_DIR,"chain_info.txt")

# Model quality assessment tools paths
IPSAE_SCRIPT_PATH = os.path.join(SCRIPTS_DIR, "ipsae.py")
PDOCKQ2_SCRIPT_PATH = os.path.join(SCRIPTS_DIR, "pdockq2_pae.py")
PDOCKQ_SCRIPT_PATH = os.path.join(SCRIPTS_DIR, "pdockq.py")
RF_PATH = os.path.join(PRETRAINED_MODELS_DIR, "rf_quality.pkl")

# PyDock
PYDOCK_DIR = os.path.join(BASE_DIR, "pydock")
PYDOCK_SIF = os.path.join(PYDOCK_DIR, "pydock.sif")

# Modelos y resultados
MODELS_DIR = os.path.join(BASE_DIR, "pretrained_models")
POTENTIALS_DIR = os.path.join(MODELS_DIR, "potentials")
ESMC_DIR = "/Users/alexascunceparis/.cache/huggingface/hub/models--biohub--ESMC-6B"



