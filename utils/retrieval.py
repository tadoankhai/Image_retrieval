import torch
import warnings
from PIL import Image
import faiss
import numpy as np
import os
from transformers.image_utils import load_image
import clip

from utils.faiss_processing import load_bin_file
from utils.extract_features import load_model
from utils import config
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load Faiss (FlatIP)

# Image
def encode_image_path(image_path):
    model, preprocess = load_model(type_model="img")
    # image = load_image(image_path)
    image = Image.open(image_path)
    inputs = preprocess(images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        feature = model.get_image_features(**inputs).cpu().detach().numpy()
    
    return feature

def encode_image_crop(image_crop):
    model, preprocess = load_model(type_model="img")
    inputs = preprocess(images=[image_crop], return_tensors="pt").to(model.device)
    with torch.no_grad():
        feature = model.get_image_features(**inputs).cpu().detach().numpy()
    
    return feature

def encode_text_from_image(image_path):
    model_txt, preprocess_txt = load_model(type_model="text")
    image = Image.open(image_path)
    inputs = preprocess_txt(images=[image], return_tensors="pt").to(model_txt.device)
    with torch.no_grad():
        text = model_txt.generate(**inputs)
        text = preprocess_txt.decode(text[0], skip_special_tokens=True)

    # Load CLIP model for text encoding
    model_img, _ = load_model(type_model="img")
    token = clip.tokenize([text]).to(model_img.device)
    with torch.no_grad():
        feature = model_img.get_text_features(token).cpu().detach().numpy()

    return feature, text

# Return index from image path
def return_image_id(img_path=None, top_k=100):
    bin_file_path = os.path.join(config.DICT_PATH, config.FILE_BIN_NAME)
    index = load_bin_file(bin_file_path)
    if img_path is not None:
        image_feature = encode_image_path(img_path).astype(np.float32)
        faiss.normalize_L2(image_feature)  # Normalize before querying
        D, I = index.search(image_feature, top_k)
        # Sort by distance ascending → score cao đến thấp
        return D[0], I[0]
    return None, None

# Return index from image crop
def return_image_crop_id(img_crop=None, top_k=100):
    bin_file_path = os.path.join(config.DICT_PATH, config.FILE_BIN_NAME)
    index = load_bin_file(bin_file_path)
    if img_crop is not None:
        image_feature = encode_image_crop(img_crop).astype(np.float32)
        faiss.normalize_L2(image_feature)  # Normalize before querying
        D, I = index.search(image_feature, top_k)
        # Sort by distance ascending → score cao đến thấp
        return D[0], I[0]
    return None, None

def return_image_text_id(img_path=None, top_k=100):
    bin_file_path = os.path.join(config.DICT_PATH, config.FILE_BIN_NAME)
    index = load_bin_file(bin_file_path)
    if img_path is not None:
        text_feature, text = encode_text_from_image(img_path)
        image_feature = encode_image_path(img_path)
        faiss.normalize_L2(image_feature.astype(np.float32))  # Normalize before querying
        faiss.normalize_L2(text_feature.astype(np.float32))
        final_feature = config.SCORE_TEXT * text_feature + config.SCORE_IMG * image_feature
        faiss.normalize_L2(final_feature.astype(np.float32))
        D, I = index.search(final_feature, top_k)
        # Sort by distance ascending → score cao đến thấp
        return D[0], I[0], text
    return None, None, None

