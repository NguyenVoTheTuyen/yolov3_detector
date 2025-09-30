import cv2
import numpy as np
import time

# --- CẤU HÌNH CƠ BẢN ---
# Đảm bảo các file này nằm cùng thư mục với file code
WEIGHTS_PATH = "yolov3.weights"
CONFIG_PATH = "yolov3.cfg"
LABELS_PATH = "coco.names"
IMAGE_PATH = "test_image.jpg" # Thay thế bằng tên ảnh của bạn

CONF_THRESHOLD = 0.5    # Ngưỡng tin cậy (Confidence threshold)
NMS_THRESHOLD = 0.4     # Ngưỡng NMS (Non-Maximum Suppression)

# Kích thước đầu vào của YOLO
YOLO_INPUT_SIZE = 416 

# --- HÀM HỖ TRỢ ---

def load_yolo_model():
    """Tải các nhãn và mô hình YOLO."""
    # 1. Tải nhãn (Labels)
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    # 2. Tải mạng nơ-ron từ cấu hình và trọng số
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
    
    # Thiết lập backend ưu tiên (Nếu có GPU, hãy thay đổi)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # Lấy tên các lớp đầu ra (unconnected layers)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Tạo màu ngẫu nhiên cho từng lớp để hiển thị
    colors = np.random.uniform(0, 255, size=(len(labels), 3))
    
    return net, labels, output_layers, colors

def detect_objects(net, output_layers, image):
    """Tiến hành phát hiện đối tượng trên hình ảnh."""
    
    (H, W) = image.shape[:2]

    # 1. Tiền xử lý (Pre-processing): Tạo blob
    # blobFromImage: Chuẩn hóa, thay đổi kích thước và chuyển đổi ảnh sang định dạng đầu vào mạng.
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), 
                                 swapRB=True, crop=False)
    
    # 2. Chạy dự đoán (Forward Pass)
    net.setInput(blob)
    start_time = time.time()
    layer_outputs = net.forward(output_layers)
    end_time = time.time()
    print(f"Thời gian Inference: {end_time - start_time:.3f} giây")

    # 3. Xử lý Hậu kỳ (Post-processing) và NMS
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            # Lấy điểm số lớp và độ tin cậy
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Lọc các dự đoán có độ tin cậy thấp
            if confidence > CONF_THRESHOLD:
                # Tính toán tọa độ hộp giới hạn (Bounding box)
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                width = int(detection[2] * W)
                height = int(detection[3] * H)

                # Tọa độ góc trên bên trái
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 4. Áp dụng Non-Maximum Suppression (NMS)
    # Loại bỏ các hộp giới hạn bị trùng lặp
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    
    return idxs, boxes, confidences, class_ids, (H, W)


def draw_detections(image, idxs, boxes, confidences, class_ids, labels, colors):
    """Vẽ hộp giới hạn và nhãn lên ảnh."""
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, w, h) = boxes[i]
            
            # Lấy nhãn và độ tin cậy
            label = labels[class_ids[i]]
            confidence = confidences[i]
            
            # Lấy màu dựa trên class_id
            color = [int(c) for c in colors[class_ids[i]]]

            # Vẽ hộp giới hạn và nhãn
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image


# --- CHƯƠNG TRÌNH CHÍNH ---

if __name__ == "__main__":
    
    print("Đang tải mô hình YOLO...")
    try:
        net, labels, output_layers, colors = load_yolo_model()
    except Exception as e:
        print("\n*** LỖI: Không thể tải mô hình hoặc các file cấu hình! ***")
        print(f"Hãy đảm bảo đã tải và đặt 3 file sau vào cùng thư mục:\n  - {WEIGHTS_PATH}\n  - {CONFIG_PATH}\n  - {LABELS_PATH}")
        print(f"Chi tiết lỗi: {e}")
        exit()
        
    print(f"Mô hình đã tải. Đang đọc ảnh: {IMAGE_PATH}")
    
    # 1. Đọc ảnh
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"\n*** LỖI: Không thể đọc ảnh '{IMAGE_PATH}'. Hãy kiểm tra tên và đường dẫn ảnh! ***")
        exit()

    # 2. Phát hiện đối tượng
    idxs, boxes, confidences, class_ids, (H, W) = detect_objects(net, output_layers, image.copy()) # Dùng image.copy() để giữ ảnh gốc

    # 3. Vẽ kết quả
    output_image = draw_detections(image, idxs, boxes, confidences, class_ids, labels, colors)

    # 4. Hiển thị và lưu ảnh
    print("Đã phát hiện đối tượng. Đang hiển thị ảnh...")
    
    cv2.imshow("YOLO Object Detection Result", output_image)
    cv2.imwrite("output_result.jpg", output_image)
    print("Đã lưu kết quả vào file: output_result.jpg")
    
    # Đợi người dùng nhấn phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()