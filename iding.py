import cv2
import numpy as np
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import os

# Load OSNet ReID model
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='',  # leave empty to download pretrained ImageNet weights
    device='cpu'
)
person_detector = YOLO('yolov8n.pt')


def scan_and_save_embedding():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "videos/personA.mp4")
    cap = cv2.VideoCapture(video_path)

    embeddings = []
    high_conf_crops = []

    print("[INFO] Scanning video to extract person embedding...")

    frame_count = 0
    while frame_count < 400:
        ret, frame = cap.read()
        if not ret:
            break

        results = person_detector(frame, classes=[0], conf=0.5)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            frame_count += 1
            continue

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < 0.7:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 100 or crop.shape[1] < 50:
                continue

            high_conf_crops.append((conf, crop))

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Sort by confidence and pick top 30 crops
    top_crops = sorted(high_conf_crops, key=lambda x: x[0], reverse=True)[:30]
    print(f"[INFO] Selected {len(top_crops)} best person crops.")

    for conf, crop in top_crops:
        resized = cv2.resize(crop, (128, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        emb = extractor(rgb)
        embeddings.append(emb.squeeze())

    reference_embedding = np.mean(embeddings, axis=0, keepdims=True)
    np.save("personA_embedding.npy", reference_embedding)
    print("[INFO] Cyclist embedding saved as 'personA_embedding.npy'")


def recognize_live():
    frame_count = 0
    reference_embedding = np.load("personA_embedding.npy")

    print(f"[DEBUG] Reference embedding shape: {reference_embedding.shape}")
    print("[INFO] Recognizing using live webcam... press 'q' to quit")

    cap = cv2.VideoCapture(0)  # 0 means default webcam
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        results = person_detector(frame, classes=[0], conf=0.5)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results else []

        print(f"[DEBUG] Frame {frame_count} - Detected persons: {len(boxes)}")

        for box in boxes:
            color = (0, 255, 0)
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                continue

            frame_resized = cv2.resize(person_crop, (128, 256))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            test_embedding = extractor(frame_rgb)
            score = cosine_similarity(test_embedding, reference_embedding)[0][0]

            print(f"[DEBUG] Cosine similarity score: {score:.2f}")

            label = f"Match Score: {score:.2f}"
            if score > 0.8:
                label += " - MATCH"
            else:
                label += " - NO MATCH"

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Live Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()



# Uncomment this line to generate embedding first
#scan_and_save_embedding(

recognize_live()
