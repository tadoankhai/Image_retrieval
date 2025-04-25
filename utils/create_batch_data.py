from PIL import Image
import os
import json
import tqdm

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

# json_path = "/home/g3090/khaitd/Retrieval/Dict/keyframe_id2path.json"
# database_dir = "/home/g3090/khaitd/Retrieval/Database"

# Transform
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384,384))
        #transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])


class NewsDataset(Dataset):
    def __init__(self, json_path, root_dir, transform):
        self.root_dir = root_dir
        
        with open(json_path, "r", encoding="utf-8") as f:
            dict_id2img = json.load(f)
        self.dict_id2img = dict_id2img
        self.transform = transform

    def get_image(self, idx):
        """Gets the image for a given row"""
        image_path = os.path.join(self.root_dir, self.dict_id2img[str(idx)])
        image = Image.open(image_path).convert("RGB")  # Đảm bảo ảnh có 3 kênh màu
        image = self.transform(image)  # Áp dụng transform để chuyển đổi ảnh thành tensor
        return image

    def __len__(self):
        return len(self.dict_id2img)

    def __getitem__(self, idx):
        image = self.get_image(idx)
        return image
    
def create_dataset(database_dir, json_path, BATCH_SIZE, NUM_WORKERS, transform=transform):
    dataset = NewsDataset(json_path = json_path, root_dir = database_dir, transform=transform)
    dl_dataset = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
    return dl_dataset