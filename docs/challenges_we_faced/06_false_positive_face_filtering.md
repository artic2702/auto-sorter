# Challenge 06 — False Positive Face Filtering on Text & Posters

## 🚨 The Symptom
In personal photo libraries containing event decorations, birthday banners, posters, and book covers:
- The face detector would occasionally detect text signs or graphics as faces.
- In user testing (`arnav_birthday`), `Person 95` and `Person 107` showed circular thumbnails of a poster reading:
  *"Save Water The World is in your hands"*.
- These non-face objects were assigned identity cards alongside real humans.

---

## 🔍 Root Cause Analysis

### Why Convolutional Face Detectors Like YuNet Trigger on Posters
YuNet is a lightweight, high-speed neural network optimized for fast inference on CPU:
1. High-contrast lettering (e.g. dark text against a white background or circular graphic illustrations) can trigger activations in YuNet's feature pyramid network that mimic eye-nose-mouth contrast gradients.
2. If the detection confidence reaches $0.75$, the bounding box is emitted as a valid face.
3. Even though SFace subsequently produces a low-confidence embedding for text, the clustering engine would assign it to a new person card if `MIN_QUALITY_TO_SEED_PERSON` was set too low ($0.35$).

---

## 🛠️ The Solution

We implemented a multi-stage filtering pipeline combining **landmark geometric validation** and **cluster seed quality gates**:

### 1. Geometric Landmark Validation (`_is_valid_face_geometry`)
In `autosorter/core/face_detect.py`, we verify the anatomical geometry of the 5 predicted landmarks (left eye, right eye, nose, left mouth, right mouth):
```python
def _is_valid_face_geometry(bbox, landmarks) -> bool:
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or len(landmarks) < 5:
        return False

    # 1. Aspect Ratio: Human faces have bounding boxes 0.65 <= h/w <= 1.65
    aspect = h / float(w)
    if aspect < 0.65 or aspect > 1.65:
        return False

    (lx, ly), (rx, ry), (nx, ny), (lmx, lmy), (rmx, rmy) = landmarks[:5]

    # 2. Eye Spacing: Distance between eyes must be >= 15% of face width
    eye_dist = math.hypot(rx - lx, ry - ly)
    if eye_dist < 0.15 * w:
        return False

    # 3. Eye-to-Mouth Distance: Vertical span must be >= 15% of face height
    mouth_cy = (lmy + rmy) / 2.0
    eye_cy = (ly + ry) / 2.0
    if abs(mouth_cy - eye_cy) < 0.15 * h:
        return False

    # 4. Triangle Area: Eyes and nose must form a non-degenerate 2D triangle
    # (Area >= 1% of bounding box area; collinear landmarks = not a face)
    tri_area = abs(lx * (ry - ny) + rx * (ny - ly) + nx * (ly - ry)) / 2.0
    if tri_area < 0.01 * (w * h):
        return False

    return True
```
- Posters with text rows produce collinear or collapsed landmarks, failing the triangle area and aspect ratio checks.

### 2. Higher Quality Gate for Person Card Seeding
In `autosorter/core/cluster.py`:
- We raised `MIN_QUALITY_TO_SEED_PERSON` from `0.35` to `0.40`.
- Even if a marginal non-face detection passes initial detection, it is blocked from creating a new person card in the library.

---

## 📊 Verification & Tests

Added unit tests in `tests/test_m2_faces.py`:
- Tested non-face images, blank frames, and extreme noise.
- Verified that distorted geometries and collinear landmarks are rejected while profile faces and tilted heads pass.

---

## 🧠 Key Takeaway
**Never trust neural network bounding box confidence scores alone.** Always validate anatomical landmark geometry (eye spacing, triangle area, aspect ratio) and require high confidence to seed new identity entities.
