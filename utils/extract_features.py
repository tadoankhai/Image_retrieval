import os
from tqdm import tqdm
import numpy as np
import torch

from transformers import AutoModel, AutoProcessor, AutoModelForImageTextToText
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.image_utils import load_image

from utils.faiss_processing import write_bin_file
from utils.create_batch_data import create_dataset
from utils import config

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(type_model="img"):
    if type_model=="img":
        processor = AutoProcessor.from_pretrained(config.MODEL_IMAGE_NAME)
        model = AutoModel.from_pretrained(config.MODEL_IMAGE_NAME, device_map="cuda:0").eval()
    elif type_model=="text":
        processor = BlipProcessor.from_pretrained(config.MODEL_TEXT_NAME)
        model = BlipForConditionalGeneration.from_pretrained(config.MODEL_TEXT_NAME, device_map="cuda:0").eval()
    elif type_model=="llm":
        processor = AutoProcessor.from_pretrained(config.MODEL_LLM_NAME)
        model = AutoModelForImageTextToText.from_pretrained(config.MODEL_LLM_NAME, device_map="cuda:0")

    return model, processor

def extract_features_siglip(data_path, dict_path, model_name, method):
    model, preprocess = load_model(type_model="img")
    features_list = []
    list_image = sorted(os.listdir(data_path))  # 
    for image in tqdm(list_image):
        image_path = os.path.join(data_path, image)
        image = load_image(image_path)
        inputs = preprocess(images=[image], return_tensors="pt").to(model.device)

        with torch.no_grad():
            feature = model.get_image_features(**inputs)[0].cpu().detach().numpy()
        # 
        features_list.append(feature)
    feature_npy = np.array(features_list).astype(np.float32)
    #
    print(feature_npy.shape)
    write_bin_file(feature_npy, dict_path, model_name, method, feature_npy.shape[1])