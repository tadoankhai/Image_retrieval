import requests
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to("cuda")

img_url = '/home/atin/ai_t4/khaitd/Image_retrieval/crops/image.png' 
raw_image = Image.open(img_url).convert('RGB')

question = "face people"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

itm_scores = model(**inputs)[0]
cosine_score = model(**inputs, use_itm_head=False)[0]

print(cosine_score)

from transformers import pipeline
from transformers.image_utils import load_image

# load pipeline
ckpt = "google/siglip2-so400m-patch14-384"
image_classifier = pipeline(model=ckpt, task="zero-shot-image-classification")

# load image and candidate labels
url = "/home/atin/ai_t4/khaitd/Image_retrieval/crops/image.png"
candidate_labels = ["The person is wearing a dark-colored shirt with short sleeves", "face people"]
image = load_image(url)
# run inference
outputs = image_classifier(image=image, candidate_labels=candidate_labels)
print(outputs)
