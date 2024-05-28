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

image_color = cv2.imread('image\image_1.png')

# Tọa độ (x, y) của góc trái trên cùng của vùng cần cắt
x = 300
y = 400

# Chiều rộng và chiều cao của vùng cần cắt
width = 600
height = 70

# Cắt ảnh
cropped_image = image_color[y:y+height, x:x+width]

# Chuyển sang ảnh xám
image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# Loại bỏ nhiễu
# blur = cv2.GaussianBlur(image_gray,(5,5),0)
blur = cv2.bilateralFilter(src=image_gray, d=3, sigmaColor=50, sigmaSpace=50)
# blur = image_gray

# Ngưỡng
# ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# _ , th1 = cv2.threshold(blur, 36, 255, cv2.THRESH_TOZERO_INV)
# _ , th = cv2.threshold(th1, 22, 255, cv2.THRESH_BINARY_INV)
_ , dst = cv2.threshold(blur, 32, 255, cv2.THRESH_BINARY)

dst = cv2.GaussianBlur(dst,(3,3),0)
# th = cv2.medianBlur(src=th, ksize=3)

# Phát hiện đường tròn bằng SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Thiết lập các tham số cho detector
# Ngưỡng phát hiện cạnh bằng canny
params.minThreshold = 0
params.maxThreshold = 255

# True: có sử dụng bộ lọc area
params.filterByArea = True
# minArea và maxArea để xác định khoảng diện tích hình tròn
params.minArea = 40
params.maxArea = 55

# True: có sử dụng bộ lọc circularity
params.filterByCircularity = True
params.minCircularity = 0.4

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.9

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.4

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(dst)

# Vẽ đường tròn lên ảnh gốc
cropped_img_with_keypoints = cv2.drawKeypoints(dst, keypoints, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Duyệt qua từng keypoint và cập nhật tọa độ
for kp in keypoints:
    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

# Vẽ đường tròn lên ảnh gốc
img_with_keypoints = cv2.drawKeypoints(image_color, keypoints, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_with_product_number = display_product_number(img_with_keypoints, len(keypoints))


# plt.subplot(2, 2, 1), plt.imshow(image_color)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 3), plt.imshow(img_with_product_number)
# plt.title('Result Image'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 1, 1), plt.imshow(cropped_image,'gray')
plt.title('Cropped Image'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 1, 2), plt.imshow(blur,'gray')
plt.title('Blured Gray Image'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 1, 3), plt.imshow(dst,'gray')
plt.title('Binary Image'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 1, 4), plt.imshow(cropped_img_with_keypoints,'gray')
plt.title('Detected Circles'), plt.xticks([]), plt.yticks([])
plt.show()

# Hiển thị kết quả
cv2.imshow("Result Image", img_with_product_number)
cv2.waitKey(0)
cv2.destroyAllWindows()