

import cv2
import numpy as np
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
# Load OSNet ReID model
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='',  # leave empty to download pretrained ImageNet weights
    device='cpu'
)
person_detector = YOLO('yolov8n.pt')


def scan_and_save_embedding():
    cap = cv2.VideoCapture(0)
    embeddings = []
    frame_count = 0
    print("[INFO] Starting 10-second scan. Please stand in front of the camera.")

    while frame_count < 100: 
        ret, frame = cap.read()
        if not ret:
            break
        results = person_detector(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results else []

        for box in boxes:
           color=(0,0,255)
           x1, y1, x2, y2 = map(int, box[:4])
           cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        frame_resized = cv2.resize(frame, (128, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        emb = extractor(frame_rgb)
        embeddings.append(emb.squeeze())

        cv2.putText(frame, f"Scanning frame {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Scan", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    reference_embedding = np.mean(embeddings, axis=0, keepdims=True)
    np.save("cyclist_embedding.npy", reference_embedding)
    print("[INFO] Cyclist embedding saved as 'cyclist_embedding.npy'")

def recognize():
    cap = cv2.VideoCapture(0)
    reference_embedding = np.load("cyclist_embedding.npy")

    print("[INFO] Recognizing... press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = person_detector(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results else []

        for box in boxes:
           color=(0,0,255)
           x1, y1, x2, y2 = map(int, box[:4])
           cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        frame_resized = cv2.resize(frame, (128, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        test_embedding = extractor(frame_rgb)
        score = cosine_similarity(test_embedding, reference_embedding)[0][0]

        label = f"Match Score: {score:.2f}"
        if score > 0.8:
            label += " - MATCH "
            cv2.putText(frame,)
        else:
            label += " - NO MATCH "

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#scan_and_save_embedding()
recognize()
