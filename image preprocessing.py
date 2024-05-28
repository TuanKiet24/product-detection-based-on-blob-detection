# https://viblo.asia/p/dem-so-luong-trung-trong-cac-khu-cong-nghiep-dua-tren-cac-phuong-phap-xu-ly-anh-bWrZnpOQ5xw
# https://learnopencv.com/edge-detection-using-opencv/#sobel-edge
# An image processing based object counting approach for machine vision application

import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_product_number(img, number):
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

image_color = cv2.imread('image/image.jpg')
image_color = cv2.resize(image_color, (848,480))

# Tọa độ (x, y) của góc trái trên cùng của vùng cần cắt
x = 0
y = 250

# Chiều rộng và chiều cao của vùng cần cắt
width = 800
height = 220

# Cắt ảnh
cropped_image = image_color[y:y+height, x:x+width]

# Chuyển sang ảnh xám
image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# Loại bỏ nhiễu
blur = cv2.GaussianBlur(image_gray,(5,5),0)

# Ngưỡng
# ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_ , dst1 = cv2.threshold(blur, 110, 255, cv2.THRESH_TOZERO_INV)
_ , dst = cv2.threshold(dst1, 30, 255, cv2.THRESH_BINARY_INV)
# dst = cv2.medianBlur(dst,ksize=7)

# Canny edge detection
edges = cv2.Canny(blur, 30, 50)
# edges_1 = cv2.GaussianBlur(edges_1,(7,7),0)
# edges = cv2.bilateralFilter(src=edges, d=5, sigmaColor=150, sigmaSpace=150)

# Kiểm tra và tạo hình ảnh kết hợp
combined_image = np.where(edges == 0, dst, edges)

# Xác định kernel cho dilation
kernel = np.ones((5, 5), np.uint8)

# Áp dụng dilation để mở rộng vùng trắng
expanded_image = cv2.dilate(combined_image, kernel, iterations=1)
expanded_image = cv2.medianBlur(expanded_image ,ksize=7)


plt.subplot(4, 1, 1), plt.imshow(blur,'gray')
plt.title('Cropped Image'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 1, 2), plt.imshow(dst,'gray')
plt.title('Blured Gray Image'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 1, 3), plt.imshow(edges,'gray')
plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 1, 4), plt.imshow(expanded_image,'gray')
plt.title('Detected Circles'), plt.xticks([]), plt.yticks([])
plt.show()


# Phát hiện đường tròn bằng SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Thiết lập các tham số cho detector
# Ngưỡng phát hiện
params.minThreshold = 0
params.maxThreshold = 255

# True: có sử dụng bộ lọc area
params.filterByArea = True
# minArea và maxArea để xác định khoảng diện tích hình tròn
params.minArea = 480

# True: có sử dụng bộ lọc circularity
params.filterByCircularity = True
params.minCircularity = 0.2

# Filter by Convexity (độ lồi - nguyên vẹn)
params.filterByConvexity = True
params.minConvexity = 0.4

# Filter by Inertia (tỉ lệ dài trục 0-1)
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(expanded_image)

cropped_img_with_keypoints = cv2.drawKeypoints(expanded_image, keypoints, np.array([]), (0, 255, 0),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Duyệt qua từng keypoint và cập nhật tọa độ
for kp in keypoints:
    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

# Lấy ra danh sách độ lớn
areas = [keypoint.size for keypoint in keypoints]

# Chuyển danh sách thành mảng NumPy
areas_array = np.array(areas)

# Tìm giá trị có trung bình bình phương cực tiểu
min_mse_value = np.inf  # Giả sử giá trị vô cùng lớn ban đầu
min_mse_index = -1

for i in range(len(areas) - 1):
    # Tính trung bình bình phương của phần còn lại của danh sách
    mse_value = np.mean((areas_array[i+1:] - areas_array[i])**2)
    
    # Kiểm tra xem có phải giá trị mới nhỏ nhất không
    if mse_value < min_mse_value:
        min_mse_value = mse_value
        min_mse_index = i

# Giá trị có trung bình bình phương cực tiểu
min_mse_area = areas[min_mse_index]

threshold = 0.2 * min_mse_area

# Loại bỏ các giá trị có chênh lệch lớn hơn 5 so với giá trị trung bình bình phương cực tiểu
filtered_keypoints = [keypoint for keypoint in keypoints if abs(keypoint.size - min_mse_area) <= threshold]

# Vẽ đường tròn lên ảnh gốc
img_with_keypoints = cv2.drawKeypoints(image_color, filtered_keypoints, np.array([]), (0, 255, 0),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_with_product_number = display_product_number(img_with_keypoints, len(keypoints))

plt.subplot(3, 2, 1), plt.imshow(cropped_image)
plt.title('Cropped Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 2), plt.imshow(blur,'gray')
plt.title('Blured Gray Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 3), plt.imshow(dst,'gray')
plt.title('Thresholding'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 4), plt.imshow(edges,'gray')
plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 5), plt.imshow(expanded_image,'gray')
plt.title('Combined Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 6), plt.imshow(cropped_img_with_keypoints,'gray')
plt.title('Detected Circles'), plt.xticks([]), plt.yticks([])
plt.show()

# Hiển thị kết quả
cv2.imshow("Result Image", img_with_product_number)
cv2.waitKey(0)
cv2.destroyAllWindows()