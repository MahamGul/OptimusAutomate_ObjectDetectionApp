from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("sample_images/test3.jpg")

for result in results:
    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(
            f"Object: {class_name}, Confidence: {confidence:.2f}"
        )