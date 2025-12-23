import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

class ConnectivityData(ABC):
    """Abstract base class for accessing connectome data."""
    
    @abstractmethod
    def get_cell_ids(self, cell_type_regex: str) -> List[int]:
        """Get IDs of cells matching a type pattern."""
        pass

    @abstractmethod
    def get_connectivity_matrix(self, presynaptic_ids: List[int], postsynaptic_ids: List[int]) -> np.ndarray:
        """Get the weight matrix W[post, pre]."""
        pass
    
    @abstractmethod
    def get_metadata(self, cell_ids: List[int]) -> pd.DataFrame:
        """Get metadata (type, layer, etc.) for a list of cells."""
        pass

class MockConnectome(ConnectivityData):
    """Generates synthetic connectivity data for testing."""
    
    def __init__(self, n_sm_types: int = 5, n_cells_per_type: int = 20, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_sm_types = n_sm_types
        self.n_cells_per_type = n_cells_per_type
        self.total_cells = n_sm_types * n_cells_per_type
        
        # Generate synthetic cell IDs and types
        self.cell_ids = np.arange(1, self.total_cells + 1)
        self.types = []
        for i in range(n_sm_types):
            self.types.extend([f"Sm{i:02d}"] * n_cells_per_type)
        
        self.metadata = pd.DataFrame({
            "root_id": self.cell_ids,
            "cell_type": self.types,
            "neuropil": ["Medulla"] * self.total_cells
        })
        
        # Generate synthetic adjacency matrix (sparse, directed)
        # Structure: High recurrence within Sm types, sparse between others
        self.W = self._generate_synthetic_weights()

    def _generate_synthetic_weights(self) -> np.ndarray:
        W = np.zeros((self.total_cells, self.total_cells))
        
        # Create block structure
        for i in range(self.n_sm_types):
            start_i = i * self.n_cells_per_type
            end_i = start_i + self.n_cells_per_type
            
            # Recurrent block (self-excitation/inhibition within type)
            block = self.rng.random((self.n_cells_per_type, self.n_cells_per_type))
            mask = block > 0.7 # 30% connectivity
            W[start_i:end_i, start_i:end_i] = block * mask * 0.1 # Small weights
            
            # Cross-type connections (random)
            for j in range(self.n_sm_types):
                if i == j: continue
                start_j = j * self.n_cells_per_type
                end_j = start_j + self.n_cells_per_type
                
                block_cross = self.rng.random((self.n_cells_per_type, self.n_cells_per_type))
                mask_cross = block_cross > 0.9 # 10% connectivity
                W[start_i:end_i, start_j:end_j] = block_cross * mask_cross * 0.05

        return W

    def get_cell_ids(self, cell_type_regex: str) -> List[int]:
        # Simple string match for regex simulation
        import re
        pattern = re.compile(cell_type_regex)
        return self.metadata[self.metadata["cell_type"].apply(lambda x: bool(pattern.match(x)))]["root_id"].tolist()

    def get_connectivity_matrix(self, presynaptic_ids: List[int], postsynaptic_ids: List[int]) -> np.ndarray:
        # Map IDs to indices
        pre_indices = [np.where(self.cell_ids == pid)[0][0] for pid in presynaptic_ids if pid in self.cell_ids]
        post_indices = [np.where(self.cell_ids == pid)[0][0] for pid in postsynaptic_ids if pid in self.cell_ids]
        
        if not pre_indices or not post_indices:
            return np.zeros((len(postsynaptic_ids), len(presynaptic_ids)))
            
        return self.W[np.ix_(post_indices, pre_indices)]

    def get_metadata(self, cell_ids: List[int]) -> pd.DataFrame:
        return self.metadata[self.metadata["root_id"].isin(cell_ids)]

class FlyWireClient(ConnectivityData):
    """Real FlyWire/Codex API interaction using FAFBSEG."""
    
    def __init__(self, token: str):
        self.token = token
        try:
            from fafbseg import flywire
            # Authenticate
            # Warning: overwrite=True might be aggressive for user global config, but necessary here.
            flywire.set_chunkedgraph_secret(token, overwrite=True)
            print("Authenticated with FlyWire via FAFBSEG.")
            
            # Set default materialization if needed. 
            # We'll use 'auto' or basic queries which default to latest usually.
            # User snippet used 783, but that might be old. Let's try to not specify it to get latest,
            # or if it fails, default to a known recent one.
            
        except ImportError:
            raise ImportError("fafbseg not installed. Please install it.")
        except Exception as e:
            print(f"Warning: Could not connect to FlyWire: {e}")

    def get_cell_ids(self, cell_type_regex: str) -> List[int]:
        from fafbseg import flywire
        from fafbseg.flywire import NeuronCriteria as NC
        
        print(f"Querying FlyWire for type matching '{cell_type_regex}'...")
        
        try:
            # Create criteria
            # We map "Sm.*" regex to something fafbseg understands. 
            # NC accepts regex=True.
            crit = NC(cell_type=cell_type_regex, regex=True)
            
            ids = crit.get_roots()
            
            if len(ids) == 0:
                 # Check if string specific?
                 print("No neurons found directly. Trying broad search...")
            
            # Convert to list of ints
            return [int(x) for x in ids]
            
        except Exception as e:
            print(f"Error querying cell IDs: {e}")
            return []

    def get_connectivity_matrix(self, presynaptic_ids: List[int], postsynaptic_ids: List[int]) -> np.ndarray:
        """
        Returns SIGNED connectivity matrix. 
        W[post, pre] > 0 for Exc, < 0 for Inh.
        Units: Synapse Counts.
        """
        import os
        from fafbseg import flywire
        import pandas as pd
        
        # Check cache
        cache_file = f"cache_W_{len(presynaptic_ids)}x{len(postsynaptic_ids)}.npy"
        if os.path.exists(cache_file):
            print(f"Loading cached connectivity from {cache_file}...")
            return np.load(cache_file)
            
        print(f"Fetching connectivity and neurotransmitters for {len(presynaptic_ids)} pre x {len(postsynaptic_ids)} post...")
        
        # 1. Get Adjacency (Synapse Counts)
        try:
            adj_df = flywire.synapses.get_adjacency(
                sources=presynaptic_ids, 
                targets=postsynaptic_ids
            )
            
            # Reindexing to ensure alignment
            adj_df = adj_df.reindex(index=presynaptic_ids, columns=postsynaptic_ids, fill_value=0)
            W_counts = adj_df.values.T # Shape (n_post, n_pre)
            
            # 2. Get Neurotransmitters for PRE-synaptic neurons
            print("Fetching NT predictions (this may take time)...")
            
            # Batching predictions to avoid timeouts/size limits if list is huge
            # fafbseg might handle it, but being safe
            signs = []
            sign_map = {
                'acetylcholine': 1, 'gaba': -1, 'glutamate': -1,
                'octopamine': 1, 'serotonin': 1, 'dopamine': 1, None: 1
            }
            
            batch_size = 500
            for i in range(0, len(presynaptic_ids), batch_size):
                print(f"  Batch {i}/{len(presynaptic_ids)}")
                batch_ids = presynaptic_ids[i:i+batch_size]
                try:
                    nt_preds = flywire.get_transmitter_predictions(batch_ids, single_pred=True)
                    
                    for pid in batch_ids:
                        pred = nt_preds.get(pid)
                        nt = pred.transmitter if pred else None
                        signs.append(sign_map.get(nt, 1))
                except Exception as e:
                    print(f"  Batch failed: {e}. Defaulting batch to Excitatory.")
                    signs.extend([1] * len(batch_ids))
            
            signs = np.array(signs) # Shape (n_pre,)
            
            # 3. Apply signs
            W_signed = W_counts * signs[None, :]
            
            # Ensure valid numeric type
            W_signed = W_signed.astype(float)
            
            # Cache it
            np.save(cache_file, W_signed)
            print(f"Saved connectivity to {cache_file}")
            
            return W_signed
            
        except Exception as e:
            print(f"Error fetching connectivity: {e}")
            return np.zeros((len(postsynaptic_ids), len(presynaptic_ids)))

    def get_metadata(self, cell_ids: List[int]) -> pd.DataFrame:
        # Placeholder or use flywire.fetch_annotations if needed
        return pd.DataFrame({'root_id': cell_ids})
