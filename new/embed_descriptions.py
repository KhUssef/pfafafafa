"""
Script to embed city descriptions using LM Studio embeddings API.
"""

import csv
import json
import os
import numpy as np
import requests
import faiss
from pathlib import Path
from tqdm import tqdm

# Configuration
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBED_MODEL = "qwen/text-embedding-qwen3-embedding-0.6b"

# Input/Output paths
INPUT_CSV = "filtered_cities_with_descriptions.csv"
OUTPUT_DIR = "embeddings"


def load_csv(filepath):
    """Load CSV file with city, country, and description."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def get_embedding(text, max_retries=3):
    """Get embedding for a single text from LM Studio."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                LM_STUDIO_URL,
                json={
                    "model": EMBED_MODEL,
                    "input": text
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result['data'][0]['embedding']
                return embedding
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    print(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                    continue
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                continue
            return None
    
    return None


def embed_descriptions(data, output_dir="embeddings", save_checkpoint=True):
    """Embed all descriptions and save results."""
    embeddings = []
    metadata = []
    checkpoint_path = os.path.join(output_dir, "embeddings_checkpoint.json")
    embeddings_checkpoint_npy = os.path.join(output_dir, "embeddings_checkpoint.npy")
    
    # Load checkpoint if it exists
    start_idx = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
            metadata = checkpoint['metadata']
            start_idx = checkpoint['last_index'] + 1
        
        if os.path.exists(embeddings_checkpoint_npy):
            embeddings = np.load(embeddings_checkpoint_npy).tolist()
    
    print(f"Starting from index {start_idx}...")
    
    for i in tqdm(range(start_idx, len(data)), desc="Embedding descriptions"):
        row = data[i]
        text = row['description']
        
        embedding = get_embedding(text)
        
        if embedding is None:
            print(f"Failed to embed: {row['city']}")
            continue
        
        embeddings.append(embedding)
        metadata.append({
            "city": row['city'],
            "country": row['country'],
            "description": text
        })
        
        # Save checkpoint every 10 items
        if save_checkpoint and (i + 1) % 10 == 0:
            np.save(embeddings_checkpoint_npy, np.array(embeddings))
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": metadata,
                    "last_index": i
                }, f, indent=2)
            print(f"Checkpoint saved at index {i}")
    
    return np.array(embeddings), metadata


def create_faiss_index(embeddings, output_dir="embeddings"):
    """Create and save FAISS index from embeddings."""
    print("\nCreating FAISS index...")
    
    # Convert to float32 if needed
    embeddings_fp32 = np.array(embeddings, dtype=np.float32)
    
    # Create index
    dimension = embeddings_fp32.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_fp32)
    
    # Save index
    index_path = os.path.join(output_dir, "embeddings.index")
    faiss.write_index(index, index_path)
    
    print(f"✓ FAISS index saved to {index_path}")
    print(f"  Index type: L2 (Euclidean distance)")
    print(f"  Vectors indexed: {index.ntotal}")
    print(f"  Dimension: {dimension}")


def main(output_dir=OUTPUT_DIR):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading CSV from {INPUT_CSV}...")
    data = load_csv(INPUT_CSV)
    print(f"Loaded {len(data)} entries")
    
    print(f"\nStarting embeddings with model: {EMBED_MODEL}")
    print(f"API URL: {LM_STUDIO_URL}")
    
    embeddings, metadata = embed_descriptions(data, output_dir=output_dir)
    
    # Save final results
    print(f"\nSaving {len(embeddings)} embeddings...")
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    metadata_path = os.path.join(output_dir, "embeddings_metadata.json")
    
    np.save(embeddings_path, embeddings)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Embeddings saved to {embeddings_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    
    # Create FAISS index
    create_faiss_index(embeddings, output_dir)
    
    # Clean up checkpoint files
    checkpoint_path = os.path.join(output_dir, "embeddings_checkpoint.json")
    checkpoint_npy = os.path.join(output_dir, "embeddings_checkpoint.npy")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists(checkpoint_npy):
        os.remove(checkpoint_npy)
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Output directory: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
