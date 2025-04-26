import faiss
import os
import numpy as np

def write_bin_file(feats, bin_path: str, type_model, method='cosine', feature_shape=256): 
    feats = feats.astype(np.float32)

    if method == 'L2':
        # Normalize trước khi dùng L2 để mô phỏng cosine similarity
        #faiss.normalize_L2(feats)
        index = faiss.IndexFlatL2(feature_shape)
    elif method == 'cosine':
        # Với cosine FAISS dùng dot-product nên cần chuẩn hóa
        #faiss.normalize_L2(feats)
        index = faiss.IndexFlatIP(feature_shape)
    else:
        raise ValueError(f"{method} not supported")

    index.add(feats)
    save_path = os.path.join(bin_path, f"faiss_{type_model}_{method}.bin")
    faiss.write_index(index, save_path)
    print(f'Saved {save_path}')

def load_bin_file(bin_file: str):
    return faiss.read_index(bin_file)
