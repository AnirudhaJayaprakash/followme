# Person Re-Identification System (Progress Report)

## Overview

This project is focused on building a real-time person re-identification system. The goal is to recognize whether a specific person ("Person A") appears in a video feed, using a reference video to generate their appearance signature.

---

## Tools and Models Used

* **YOLOv8n**: For real-time person detection.
* **OSNet (osnet\_x1\_0)**: For extracting appearance features of individuals.
* **Torchreid**: Used to extract embeddings.
* **OpenCV & NumPy**: For video frame processing and data handling.
* **Cosine Similarity**: To compare reference and test embeddings.

---

## Implemented Features

### 1. Embedding Generation

* A reference video (`personA.mp4`) is scanned frame-by-frame.
* High-confidence person detections are cropped.
* These crops are passed through OSNet to extract feature vectors.
* The top 30 high-confidence crops are used to calculate a mean embedding.
* The embedding is saved as `personA_embedding.npy`.

### 2. Live Person Recognition

* The system accepts input from webcam or mobile camera (DroidCam).
* For every frame:

  * Detects persons using YOLOv8n.
  * Crops each detection, extracts its embedding.
  * Compares it with the saved reference using cosine similarity.
  * Displays "MATCH" if similarity > 0.8.

---

## Current Status

* Successfully detects and recognizes the person from a live feed.
* Achieves \~0.8 match score for the same person under consistent conditions.
* System can distinguish different people (e.g., scores around 0.60 for others).
* Lighting, camera quality, and angle affect performance.

---

## Next Steps

* Improve embedding accuracy with better lighting and varied conditions.
* Add face verification for enhanced accuracy.
* Build a GUI/Web interface.
* Allow storing multiple identities.

---

## Usage

```python
# Generate embedding from reference video
scan_and_save_embedding()

# Start real-time recognition
recognize_live()
```

---

## Note

Make sure the reference video clearly shows the person in consistent lighting and full-body view. A 20–30 second video works best for embedding extraction.

---

**Developed by: \[Team Arjuna]**
**Date:** July 8 2025
