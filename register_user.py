import cv2
import torch
import torchreid
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# -----------------------------
# Initialize YOLO person detector
# -----------------------------
print("✅ Loading YOLO model...")
yolo = YOLO("yolov8n.pt")

# -----------------------------
# Initialize Torchreid model
# -----------------------------
print("✅ Initializing Torchreid model...")
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
print("✅ Building Torchreid transforms...")
_, preprocess = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    random_erase=False
)

print("📸 Please stand in front of the camera. Press 'c' to capture and 'q' to quit.")
cap = cv2.VideoCapture(0)
embeddings = []

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

            # Draw bounding box with capture hint
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture | 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Register User", frame)

        # ✅ Single key handler for both 'c' and 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(results[0].boxes.xyxy) > 0:
            embeddings.append(features.cpu())
            print("✅ Captured embedding.")
        elif key == ord('q'):
            print("👋 Quitting registration...")
            break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# Save embedding
# -----------------------------
if embeddings:
    user_embedding = torch.mean(torch.stack(embeddings), dim=0)
    torch.save({"embedding": user_embedding}, "user_embedding.pth")
    print("✅ User embedding saved to user_embedding.pth")
else:
    print("⚠️ No embeddings captured. Try again.")

