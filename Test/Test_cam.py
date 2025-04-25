import cv2
import os
import time

# Cấu hình
rtsp_url = "rtsp://admin:T4123456@192.168.1.16/Streaming/Channels/1"
output_folder = "captured_images"
images_per_second = 10
total_images = 1000

# Tạo thư mục lưu ảnh nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Kết nối tới camera
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Không thể kết nối tới camera.")
    exit()

count = 0
interval = 1 / images_per_second

print("Đang bắt đầu chụp ảnh...")

while count < total_images:
    start_time = time.time()
    for _ in range(images_per_second):
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(output_folder, f"image_{count+1:03}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"Đã lưu {filename}")
        if count >= total_images:
            break
        time.sleep(interval)
    # Đảm bảo đúng 1 giây giữa mỗi lần cắt
    elapsed = time.time() - start_time
    if elapsed < 1:
        time.sleep(1 - elapsed)

cap.release()
print("Đã lưu đủ 50 ảnh.")
