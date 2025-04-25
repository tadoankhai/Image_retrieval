import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
import clip
from PIL import Image
import torch.nn.functional as F

# load the model and processor
ckpt = "google/siglip2-base-patch16-224"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

# load the image
image1_path = "/home/atin/ai_t4/khaitd/Image_retrieval/crops/image.png"
image2_path = "/home/atin/ai_t4/khaitd/Image_retrieval/Database/bf667c36-7ead-4800-8206-2327bcebc93a.jpg"
image1 = load_image(image1_path)
image2 = load_image(image2_path)
inputs1 = processor(images=[image1], return_tensors="pt").to(model.device)
inputs2 = processor(images=[image2], return_tensors="pt").to(model.device)

# run inference
with torch.no_grad():
    image_embeddings1 = model.get_image_features(**inputs1)    
    image_embeddings2 = model.get_image_features(**inputs2)    

# Tính cosine similarity
print(image_embeddings1.shape)
cos_sim = F.cosine_similarity(image_embeddings1, image_embeddings2).item()
print("Cosine similarity siglip:", cos_sim)


################ CLIP #####################
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image1 = preprocess(Image.open(image1_path)).unsqueeze(0).to(device)
image2 = preprocess(Image.open(image2_path)).unsqueeze(0).to(device)
with torch.no_grad():
    image1_features = model.encode_image(image1)
    image2_features = model.encode_image(image2)

# Tính cosine similarity
cos_sim_clip = F.cosine_similarity(image1_features, image2_features).item()
print("Cosine similarity clip:", cos_sim_clip)

################ VIT #####################
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

image1 = Image.open(image1_path).convert("RGB")   
image2 = Image.open(image2_path).convert("RGB")
# load the model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
inputs1 = processor(images=image1, return_tensors="pt")
inputs2 = processor(images=image2, return_tensors="pt")
outputs1 = model(**inputs1)
outputs2 = model(**inputs2)
last_hidden_states1 = outputs1.last_hidden_state
last_hidden_states2 = outputs2.last_hidden_state

cls_emb1 = last_hidden_states1[:, 0, :]  # shape: (1, 768)
cls_emb2 = last_hidden_states2[:, 0, :]
cos_sim = F.cosine_similarity(cls_emb1, cls_emb2).item()
print("Cosine similarity vit:", cos_sim)


##################### DINOv2#####################
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

image1 = Image.open(image1_path)   
image2 = Image.open(image2_path)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')

inputs1 = processor(images=image1, return_tensors="pt")
inputs2 = processor(images=image2, return_tensors="pt")
outputs1 = model(**inputs1)
outputs2 = model(**inputs2)
last_hidden_states1 = outputs1.last_hidden_state
last_hidden_states2 = outputs2.last_hidden_state

cls_emb1 = last_hidden_states1[:, 0, :]  # shape: (1, 768)
cls_emb2 = last_hidden_states2[:, 0, :]
cos_sim = F.cosine_similarity(cls_emb1, cls_emb2).item()
print("Cosine similarity DINO:", cos_sim)

##################### BLIP Image captioning #####################
from transformers import pipeline

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
image1 = Image.open(image1_path)
image1_caption = pipe(image1)[0]['generated_text']
print("BLIP Image captioning:")
print("Image 1 caption:", image1_caption)