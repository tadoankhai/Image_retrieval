from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

def crop_person(image_path, output_dir="crops"):
    # Load model YOLOv8
    model = YOLO("yolov8n.pt")  

    # Đọc ảnh bằng PIL
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)  # chuyển sang NumPy để crop

    results = model(image)[0]  # lấy kết quả đầu tiên

    # Tạo thư mục lưu nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    cropped_images = []

    for box in results.boxes:
        cls_id = int(box.cls[0])  # class id
        conf = box.conf[0]        # confidence
        label = results.names[cls_id]

        if label == "person" and conf > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_np = image_np[y1:y2, x1:x2]  # crop bằng NumPy
            cropped = Image.fromarray(cropped_np)  # chuyển lại về ảnh PIL nếu cần

            # Lưu ảnh
            save_path = os.path.join(output_dir, f"person_{count}.jpg")
            cropped.save(save_path)

            cropped_images.append(cropped)
            count += 1

    print(f"Cropped and saved {count} person(s).")
    return cropped_images[0] if cropped_images else None  # trả về 1 ảnh (như trước)

    
# if __name__=="__main__":
#     image_path = "/home/atin/ai_t4/khaitd/image_094.jpg"
#     crop_person(image_path)