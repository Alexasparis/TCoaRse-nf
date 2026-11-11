import os
import sys
import argparse
import subprocess
import glob
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from config import PYDOCK_DIR, PYDOCK_SIF

def run(cmd):
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run PyDock pipeline")
    parser.add_argument("--input", default="./results/examples_pdb/", help="Input PDB directory")
    parser.add_argument("--output", default="./results/examples_pydock/", help="PyDock output directory")
    args = parser.parse_args()
    chunk_id = 0
    config = os.path.join(PYDOCK_DIR, "config.yaml")
    chunk = os.path.join(args.output, "chunks", f"chunk_{chunk_id:06d}.tsv")
    print(f"Using config: {config}")
    run([sys.executable, os.path.join(PYDOCK_DIR, "01_make_manifest.py"), "--input", args.input, "--output", args.output])
    run([sys.executable, os.path.join(PYDOCK_DIR, "02_make_chunks.py"), "--config", config])
    run([sys.executable, os.path.join(PYDOCK_DIR, "03_validate_chain_mapping.py"), "--config", config])
    run([sys.executable, os.path.join(PYDOCK_DIR, "worker_chunk.py"), "--config", config, "--chunk", chunk, "--chunk-id", str(chunk_id), "--local-sif", PYDOCK_SIF])
    archives = glob.glob(os.path.join(args.output, "archives", "shards", "*", f"chunk_{args.chunk_id:06d}.tar"))
    shutil.copy2(archives[0], os.path.join(args.output, f"{args.basename}_ene.tar"))
    print("\nPyDock pipeline completed successfully!")

if __name__ == "__main__":
    main()
