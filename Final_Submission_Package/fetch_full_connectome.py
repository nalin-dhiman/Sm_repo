import numpy as np
import sys
import os
import argparse

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.data_loader import FlyWireClient
from src.config import FLYWIRE_API_TOKEN

def fetch_connectome():
    print("=== Fetching Full Sm Connectome ===")
    
    try:
        client = FlyWireClient(token=FLYWIRE_API_TOKEN)
        
        # 1. Get All Sm IDs
        print("Querying for ALL 'Sm' neurons...")
        # Broad regex for all Serpentine Medulla types
        sm_ids = client.get_cell_ids("Sm.*")
        print(f"Found {len(sm_ids)} Sm neurons.")
        
        if len(sm_ids) == 0:
            print("No neurons found. Check API/Criteria.")
            return

        # 2. Sort for consistency
        sm_ids.sort()
        
        # 3. Fetch Matrix
        # This handles batching and caching internally in data_loader
        print(f"Fetching connectivity for {len(sm_ids)}x{len(sm_ids)} matrix...")
        # It auto-caches to cache_W_NxN.npy
        W = client.get_connectivity_matrix(sm_ids, sm_ids)
        
        print(f"Successfully fetched matrix with shape {W.shape}")
        print("Done.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_connectome()
