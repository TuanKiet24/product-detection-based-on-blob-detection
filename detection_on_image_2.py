# An image processing based object counting approach for machine vision application

import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_product_number(img, number):
    # Kích thước ảnh
    height, width, _ = img.shape

    # Kích thước và vị trí của khung trắng góc trái
    frame_width = 250
    frame_height = 50
    frame_x = 10
    frame_y = 10

    # Vẽ khung trắng góc trái
    cv2.rectangle(img, (frame_x, frame_y), (frame_x + frame_width, frame_y + frame_height), (255, 255, 255), -1)

    # Chọn một font và kích thước chữ
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    # Kích thước chữ và vị trí
    text_size = cv2.getTextSize("Product Number: " + str(number), font, font_scale, font_thickness)[0]
    text_x = frame_x + (frame_width - text_size[0]) // 2
    text_y = frame_y + (frame_height + text_size[1]) // 2

    # Điền nội dung vào khung
    cv2.putText(img, "Product Number: " + str(number), (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)

    # Đẩy ra
    return img

image_color = cv2.imread('image\image_2.jpg')
image_color = cv2.resize(image_color,(960,540))

# Chuyển sang ảnh xám
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Loại bỏ nhiễu
blur = cv2.GaussianBlur(image_gray,(3,3),0)

# Canny edge detection
edges = cv2.Canny(blur, 25, 150)
edges = cv2.GaussianBlur(edges,(3,3),0)

# Phát hiện đường tròn bằng SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Thiết lập các tham số cho detector
# Ngưỡng phát hiện cạnh bằng canny
params.minThreshold = 0
params.maxThreshold = 255

# True: có sử dụng bộ lọc area
params.filterByArea = True
# minArea và maxArea để xác định khoảng diện tích hình tròn
params.minArea = 1000

# True: có sử dụng bộ lọc circularity
params.filterByCircularity = True
params.minCircularity = 0.6

# Filter by Convexity (độ lồi - nguyên vẹn)
params.filterByConvexity = True
params.minConvexity = 0.9

# Filter by Inertia (tỉ lệ dài trục 0-1)
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(edges)

# Vẽ đường tròn lên ảnh gốc
img_with_keypoints = cv2.drawKeypoints(image_color, keypoints, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_with_product_number = display_product_number(img_with_keypoints, len(keypoints))

for keypoint in keypoints:
    x = int(keypoint.pt[0])  # Tọa độ x của tâm
    y = int(keypoint.pt[1])  # Tọa độ y của tâm
    radius = int(keypoint.size / 2)  # Bán kính

    print(f"Circle: Center=({x}, {y}), Radius={radius}")

# Hiển thị ảnh
cv2.imshow("Detected Circles", img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()