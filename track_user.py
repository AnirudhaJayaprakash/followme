import cv2
import torch
import torchreid
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# -----------------------------
# Load saved embedding
# -----------------------------
print("✅ Loading saved user embedding...")
user_data = torch.load("user_embedding.pth", map_location="cpu")
user_embedding = user_data["embedding"]

# -----------------------------
# Initialize YOLO
# -----------------------------
yolo = YOLO("yolov8n.pt")

# -----------------------------
# Initialize Torchreid model
# -----------------------------
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=False
)
torchreid.utils.load_pretrained_weights(model, "osnet_x1_0_imagenet.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# -----------------------------
# Torchreid preprocessing
# -----------------------------
_, preprocess = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    random_erase=False
)

print("🎯 Starting user tracking... Press 'q' to quit.")
cap = cv2.VideoCapture(0)

KNOWN_PERSON_HEIGHT_CM = 170  # assumed person height in cm
FOCAL_LENGTH_PIXELS = 600     # camera-dependent; can be tuned

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)

        for det in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, det)
            crop = frame[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            crop_tensor = preprocess(crop_pil).unsqueeze(0).to(device)
            features = model(crop_tensor)

            cosine_sim = F.cosine_similarity(features.cpu(), user_embedding.cpu()).item()

            # Estimate distance based on bounding box height
            box_height = y2 - y1
            distance_cm = (KNOWN_PERSON_HEIGHT_CM * FOCAL_LENGTH_PIXELS) / box_height if box_height > 0 else 0

            if cosine_sim > 0.75:  # Match threshold
                color = (0, 255, 0)  # Green
                label = f"USER ✅ ({cosine_sim:.2f}) Dist: {distance_cm:.1f}cm"
            else:
                color = (0, 0, 255)  # Red
                label = f"NOT USER ❌ ({cosine_sim:.2f}) Dist: {distance_cm:.1f}cm"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Track User", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

