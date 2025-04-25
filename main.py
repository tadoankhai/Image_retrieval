from utils import config
from utils.extract_features import extract_features_siglip
from utils.create_dict_image_path import key_to_image

# extract_features_siglip(data_path=config.ROOT_DIR, dict_path=config.DICT_PATH, model_name=config.MODEL_NAME, method=config.METHOD)
# key_to_image(data_path=config.ROOT_DIR, dict_path=config.DICT_PATH,file_dict_name=config.FILE_DICT_NAME)

from utils.retrieval import encode_text_from_image
image_path = "/home/atin/ai_t4/khaitd/Image_retrieval/crops/person_7.jpg"
feature = encode_text_from_image(image_path)
print(feature)