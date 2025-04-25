from tqdm import tqdm
import os
import json
import warnings
warnings.filterwarnings("ignore")

# Create json key to image
def key_to_image(data_path, dict_path, file_dict_name):
    # Create Dictionary
    dic = {}
    i = 0
    list_image = sorted(os.listdir(data_path))  # 
    for image in tqdm(list_image):
        image_path = os.path.join(data_path, image)
        result = image_path.split("Database/")[-1]  # 
        dic[i] = result
        i += 1

    # Save file json
    json_path = os.path.join(dict_path,file_dict_name)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dic, f, indent=4, ensure_ascii=False)




