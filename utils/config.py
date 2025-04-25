MODEL_NAME="clip"
# TYPE_MODEL="vit"
# TYPE_MODEL="siglip2"
ROOT_DIR = "./Database"
DICT_PATH = "./Dict"
FILE_BIN_NAME = "faiss_clip_cosine.bin"
FILE_DICT_NAME = "keyframe_id2path.json"
METHOD="cosine" # cosine or L2


# MODEL_IMAGE_NAME="google/siglip2-so400m-patch14-384"
MODEL_IMAGE_NAME="openai/clip-vit-large-patch14"
# MODEL_IMAGE_NAME="google/vit-base-patch16-224-in21k"
MODEL_TEXT_NAME="Salesforce/blip-image-captioning-large"


TOP_K=100
SCORE_TEXT=0.3
SCORE_IMG=0.7