from PIL import Image
import os
import json
import streamlit as st

import time
from utils.extract_features import load_model
from utils.retrieval import return_image_id, return_image_crop_id
from utils.person_crop import crop_person
from utils import config
# --- Tải và Cache Model, Index ---
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

dict_id2img_json=os.path.join(config.DICT_PATH, config.FILE_DICT_NAME)
with open(dict_id2img_json, "r", encoding="utf-8") as f:
    dict_id2img = json.load(f)

@st.cache_resource
def get_model():
    return load_model(type_model="img")

model, preprocess = get_model()

# Return index
def return_id(img_path=None):
    if img_path is not None:
        crop_image = crop_person(img_path)
        if crop_image is None:
            return [], [], None
        #score, idx = return_image_crop_id(crop_image, top_k=config.TOP_K)
        score, idx = return_image_id(img_path, top_k=config.TOP_K)
    else:
        return [], [], None
    return score, idx, crop_image

# --- Streamlit ---

col1, col2 = st.columns(2)

with col1:
    with st.form("image_form"):
        input_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        submitted_image = st.form_submit_button("Submit Image")

    if submitted_image and input_image:
        image = Image.open(input_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    if submitted_image and input_image is not None:
        scores, indices, crop_image = return_id(img_path=input_image)

        # Hiển thị ảnh đã crop (người)
        if crop_image:
            st.image(crop_image, caption="Cropped Person", use_container_width=True)
        else:
            st.warning("No person detected in the image.")

        # Hiển thị các kết quả truy hồi
        for i, (score, id_) in enumerate(zip(scores, indices)):
            image_path = os.path.join("./Database", dict_id2img[str(id_)])
            try:
                image = Image.open(image_path)
                st.image(image, caption=f"Result {i+1} — Score: {score:.4f}")
            except FileNotFoundError:
                st.error(f"Image not found: {image_path}")
