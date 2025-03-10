from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# ✅ Load YOLO Model
model_path = os.getenv("MODEL_PATH", "best.pt")  # Get from env, fallback to local file
model = YOLO(model_path)  # Ensure model exists

@app.route("/")
def home():
    return jsonify({"message": "🚀 YOLOv8 Flask API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # ✅ Read Image
    file = request.files["image"].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # ✅ Run YOLO Inference
    results = model(img)

    # ✅ Extract Detections & Draw Boxes
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]

            detections.append({
                "class": class_name,
                "confidence": conf,
                "box": [x1, y1, x2, y2]
            })

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ✅ Save Processed Image
    output_path = "detected_image.jpg"
    cv2.imwrite(output_path, img)

    return jsonify({"detections": detections, "image_url": "/download"})

@app.route("/download")
def download():
    return send_file("detected_image.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
