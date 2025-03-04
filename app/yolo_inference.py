import cv2
import numpy as np
from ultralytics import YOLO

def load_yolo_model(weights_path: str = "yolov8n.pt"):
    """
    加载 Ultralytics YOLOv8 模型
    """
    model = YOLO(weights_path)
    return model

def detect_objects(model, image: np.ndarray):
    """
    使用 Ultralytics 模型进行推理，返回检测结果
    结果示例： [{'bbox': [x1, y1, x2, y2], 'label': 'cat', 'score': 0.95}, ...]
    """
    # Ultralytics YOLO 直接支持 np.ndarray 作为输入
    results = model.predict(source=image, conf=0.25)  # 可根据需要调整 conf 等参数
    
    detections = []
    # Ultralytics 返回的 results 可能包含多张图片的结果，这里只取一张
    if len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            # box.xyxy[0] 返回 [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0]
            # 获取类别索引、置信度
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = model.names[cls_id]  # 类别名称

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'label': label,
                'score': score
            })
    return detections

def annotate_image(image: np.ndarray, detections: list):
    """
    在图像上绘制边界框和标签
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        score = det['score']

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制文本标签
        text = f"{label} {score:.2f}"
        cv2.putText(image, text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image
