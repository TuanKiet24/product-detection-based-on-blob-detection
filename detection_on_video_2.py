import cv2
import numpy as np

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

    # Đẩy ảnh ra
    return img

# Mở video
cap = cv2.VideoCapture('image/video_2.mp4')

# Khởi tạo video writer
fps = 29
size = (1280,720)
out = cv2.VideoWriter('image/video_2_result.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

# Kiểm tra video mở chưa
if not cap.isOpened():
    print("Không thể mở video.")
else:
    while True:
        ret, image_color = cap.read()

        if not ret:
            break

        # Tọa độ (x, y) của góc trái trên cùng của vùng cần cắt
        x = 330
        y = 380

        # Chiều rộng và chiều cao của vùng cần cắt
        width = 720
        height = 180

        # Cắt ảnh
        cropped_image = image_color[y:y+height, x:x+width]

        # Chuyển sang ảnh xám
        image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # Loại bỏ nhiễu ảnh
        blur = cv2.GaussianBlur(image_gray,(3,3),0)

        # Canny edge detection
        edges = cv2.Canny(blur, 25, 150)
        # Loại bỏ nhiễu cạnh
        edges = cv2.GaussianBlur(edges,(7,7),0)
        
        # Phát hiện đường tròn bằng SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()

        # Thiết lập các tham số cho detector
        # Ngưỡng phát hiện
        params.minThreshold = 0
        params.maxThreshold = 255

        # Filter by Area
        params.filterByArea = True
        # minArea và maxArea
        params.minArea = 1000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.6

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(edges)

        # Duyệt qua từng keypoint và cập nhật tọa độ
        for kp in keypoints:
            kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

        # Vẽ đường tròn lên ảnh gốc
        img_with_keypoints = cv2.drawKeypoints(image_color, keypoints, np.array([]), (0, 255, 0),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        img_with_product_number = display_product_number(img_with_keypoints, len(keypoints))

        # Hiển thị ảnh
        cv2.imshow("Detected Circles", img_with_product_number)

        # Chờ 10 mili giây để tiếp tục đọc frame tiếp theo
        cv2.waitKey(10)

        # Điền frame vào sau video ra
        out.write(img_with_keypoints)
    
    cv2.destroyAllWindows()
    cap.release()
    out.release()