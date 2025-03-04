import cv2
import numpy as np
from ultralytics import YOLO

def load_yolo_model(weights_path: str = "/content/yolo-microscopy-detection/runs/detect/train7/weights/best.pt"):
    """
    加载 Ultralytics YOLOv8 模型
    :param weights_path: 权重文件路径
    :return: 加载好的 YOLO 模型对象
    """
    model = YOLO(weights_path)
    return model

def detect_objects(model, image: np.ndarray, conf_thres=0.25):
    """
    使用 Ultralytics 模型进行推理，返回检测结果
    结果示例: [{'bbox': [x1, y1, x2, y2], 'label': 'cat', 'score': 0.95}, ...]
    
    :param model: YOLO 模型对象
    :param image: OpenCV 格式 (H, W, C) 的图像
    :param conf_thres: 置信度阈值 (默认 0.25)
    :return: list(dict)
    """
    # 使用 Ultralytics YOLO 进行推理
    results = model.predict(source=image, conf=conf_thres)
    
    detections = []
    # results 可能包含多张图的推理，这里只处理一张
    if len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            # box.xyxy[0] 返回 [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0]
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = model.names[cls_id]  # 获取类别名称

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'label': label,
                'score': score
            })
    return detections

def annotate_image(image: np.ndarray, detections: list):
    """
    在图像上绘制检测框和标签
    
    :param image: OpenCV 图像
    :param detections: detect_objects 返回的列表
    :return: 标注后的 OpenCV 图像
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
