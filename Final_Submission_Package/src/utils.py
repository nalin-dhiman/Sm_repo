import os
import json
import datetime
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

import hashlib

def get_file_checksum(filepath):
    """Computes SHA256 checksum of a file."""
    if not os.path.exists(filepath): return "missing"
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_provenance(output_path, meta):
    """
    Saves metadata about the run (seeds, versions, config, checksums).
    """
    ensure_dir(os.path.dirname(output_path))
    meta['timestamp'] = datetime.datetime.now().isoformat()
    import sys
    import platform
    meta['python_version'] = sys.version
    meta['platform'] = platform.platform()
    try:
        import subprocess
        meta['git_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except:
        meta['git_hash'] = "unknown"
        
    # Serialize Config if present in meta or import it
    # We expect the caller to pass 'config' in meta if they want it saved
    
    def default(o):
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(o)
        elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError
        
    with open(output_path, 'w') as f:
        json.dump(meta, f, indent=4, default=default)

def log(msg, verbose=True):
    if verbose:
        print(f"[SmPipeline] {msg}")

import tempfile

def atomic_json_dump(obj, path):
    path = str(path)
    ensure_dir(os.path.dirname(path))
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".json")
    try:
        # Create a custom encoder that handles numpy types
        def default(o):
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                return float(o)
            elif isinstance(o, (np.ndarray,)):
                return o.tolist()
            raise TypeError
            
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=4, default=default)
            
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except OSError:
            pass
