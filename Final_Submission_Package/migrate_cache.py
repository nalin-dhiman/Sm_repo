import os
import sys
import json
import numpy as np
import scipy.sparse
import hashlib

# Add root
sys.path.append(os.getcwd())
from smdyn.data_loader import content_hash

DATA_DIR = "outputs/data"

def migrate():
    print("Migrating Cache...")
    
    # 1. IDs
    old_id_file = os.path.join(DATA_DIR, "sm_ids_783_Sm.*.json")
    if os.path.exists(old_id_file):
        with open(old_id_file, 'r') as f:
            ids = json.load(f)
        
        # New name: hash of regex
        regex = "Sm.*"
        regex_hash = hashlib.sha256(regex.encode('utf-8')).hexdigest()[:8]
        new_name = f"cache_ids_783_{regex_hash}.json"
        new_path = os.path.join(DATA_DIR, new_name)
        
        # Save structured
        data = {"regex": regex, "ids": ids, "materialization": "783"}
        with open(new_path, 'w') as f:
            json.dump(data, f)
        print(f"Migrated IDs to {new_name}")
        
    else:
        print("Legacy ID file not found.")
        return

    # 2. Counts
    old_counts = os.path.join(DATA_DIR, "W_counts_783_4463x4463.npz")
    if os.path.exists(old_counts):
        W = scipy.sparse.load_npz(old_counts)
        N = len(ids)
        assert W.shape == (N, N), f"Shape mismatch {W.shape} vs {N}"
        
        # Compute Hash
        chash = content_hash(ids, ids, salt="783")
        new_counts_name = f"cache_W_{N}x{N}_{chash}.npz"
        new_counts_path = os.path.join(DATA_DIR, new_counts_name)
        
        # Save in robust format (with metadata)
        np.savez_compressed(new_counts_path,
                            data=W.data,
                            indices=W.indices,
                            indptr=W.indptr,
                            shape=W.shape,
                            pre_ids=ids,
                            post_ids=ids,
                            materialization="783")
        print(f"Migrated Counts to {new_counts_name}")
        
    else:
        print("Legacy Counts file not found.")

if __name__ == "__main__":
    migrate()
