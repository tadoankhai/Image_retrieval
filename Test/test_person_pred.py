from ultralytics import YOLO
import cv2

# Load model YOLOv8 
model = YOLO("yolov8n.pt")  

# Dự đoán từ mô hình
link_image = "/home/atin/ai_t4/khaitd/Image_retrieval/captured_images/image_002.jpg"
image = cv2.imread(link_image)
results = model(image)[0]  # lấy kết quả đầu tiên

for box in results.boxes:
    cls_id = int(box.cls[0])  # class id
    conf = box.conf[0]        # confidence
    label = results.names[cls_id]

    if label == "person" and conf > 0.6:  # Thêm điều kiện kiểm tra confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Hiển thị ảnh kết quả (hoặc lưu lại nếu cần)
cv2.imwrite("output.jpg", image)
