from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np

from yolo_inference import load_yolo_model, detect_objects, annotate_image

app = FastAPI()

# 如果前后端分离，且需要跨域访问，可开启CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 在启动时加载 YOLO 模型
model = load_yolo_model("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "FastAPI + Ultralytics YOLOv8 server is up!"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # 1. 读取上传的图片数据
    image_bytes = await file.read()

    # 2. 转为 OpenCV 格式
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "无法解析图片"}

    # 3. 执行推理
    detections = detect_objects(model, image)

    # 4. 标注图像
    annotated_image = annotate_image(image, detections)

    # 5. 编码为JPEG
    _, encoded_img = cv2.imencode(".jpg", annotated_image)

    # 6. 以二进制流形式返回
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    # 本地开发测试
    uvicorn.run(app, host="0.0.0.0", port=5000)
