import faiss
import numpy as np
import pickle
import os

# Constants for data storage paths
DATA_DIR = "data"
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
CHUNK_DATA_FILE = os.path.join(DATA_DIR, "chunk_data.pkl")

def create_faiss_index(embeddings):
    """
    Create a FAISS index using given embeddings.

    Args:
        embeddings (np.array): Array of text embeddings.

    Returns:
        faiss.IndexFlatL2: FAISS index object.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index

def save_faiss_index(index):
    """
    Save FAISS index to disk.
    """
    faiss.write_index(index, FAISS_INDEX_FILE)

def load_faiss_index():
    """
    Load FAISS index from disk.

    Returns:
        faiss.IndexFlatL2: Loaded FAISS index object.
    """
    return faiss.read_index(FAISS_INDEX_FILE)

def save_chunks(chunks):
    """
    Save text chunks to disk using pickle.
    """
    with open(CHUNK_DATA_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks():
    """
    Load text chunks from disk.

    Returns:
        list: Loaded list of text chunks.
    """
    with open(CHUNK_DATA_FILE, "rb") as f:
        return pickle.load(f)

def faiss_index_exists():
    """
    Check whether FAISS index and chunk data exist on disk.

    Returns:
        bool: True if both files exist, False otherwise.
    """
    return os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNK_DATA_FILE)
