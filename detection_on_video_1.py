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
cap = cv2.VideoCapture('image/video_1.mp4')

# Khởi tạo video writer
fps = 29
size = (848,480)
out = cv2.VideoWriter('image/video_1_result.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

# Kiểm tra video mở chưa
if not cap.isOpened():
    print("Không thể mở video.")
else:
    while True:
        ret, image_color = cap.read()

        if not ret:
            break

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
        _ , dst1 = cv2.threshold(blur, 110, 255, cv2.THRESH_TOZERO_INV)
        _ , dst = cv2.threshold(dst1, 30, 255, cv2.THRESH_BINARY_INV)

        # Canny edge detection
        edges = cv2.Canny(blur, 30, 50)

        # Kiểm tra và tạo hình ảnh kết hợp
        combined_image = np.where(edges == 0, dst, edges)

        # Xác định kernel cho dilation
        kernel = np.ones((5, 5), np.uint8)

        # Áp dụng dilation để mở rộng vùng trắng
        expanded_image = cv2.dilate(combined_image, kernel, iterations=1)
        expanded_image = cv2.medianBlur(expanded_image ,ksize=7)
        
        # Phát hiện đường tròn bằng SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()

        # Thiết lập các tham số cho detector
        # Ngưỡng phát hiện
        params.minThreshold = 0
        params.maxThreshold = 255

        # Filter by Area
        params.filterByArea = True
        # minArea và maxArea
        params.minArea = 450

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.2

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.1

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(expanded_image)

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

        # Giá trị trung bình bình phương cực tiểu
        min_mse_area = areas[min_mse_index]

        # Chọn ngưỡng sai số là 20% của giá trị trung bình bình phương cực tiểu
        threshold = 0.20 * min_mse_area

        # Loại bỏ các giá trị có chênh lệch lớn hơn 5 so với giá trị trung bình bình phương cực tiểu
        filtered_keypoints = [keypoint for keypoint in keypoints if abs(keypoint.size - min_mse_area) <= threshold]

        # Vẽ đường tròn lên ảnh gốc
        img_with_keypoints = cv2.drawKeypoints(image_color, filtered_keypoints, np.array([]), (0, 255, 0),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        img_with_product_number = display_product_number(img_with_keypoints, len(filtered_keypoints))

        # Hiển thị ảnh
        cv2.imshow("Detected Circles", img_with_product_number)

        # Chờ 10 mili giây để tiếp tục đọc frame tiếp theo
        cv2.waitKey(10)

        # Điền frame vào sau video ra
        out.write(img_with_keypoints)
    
    cv2.destroyAllWindows()
    cap.release()
    out.release()