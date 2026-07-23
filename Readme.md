#               🚦 Traffic Violation Detection Dashboard

A Gradio dashboard that runs several traffic-violation models on **images and
video**, either all together or one at a time. Every run produces an annotated
result, a per-class summary, and a **downloadable CSV** of all detections.

## Highlights

- **Two modes, no dropdowns** — a *Run All Models* tab (one upload → combined
  overlay) and an *Individual Models* tab (each model has its own description,
  upload and results in a collapsible section).
- **Image *and* video** — upload either; video is sampled frame-by-frame, run
  through the models, and re-encoded as an annotated MP4 (capped at
  `MAX_VIDEO_FRAMES = 200` so CPU Spaces stay responsive).
- **Downloadable CSV** on every run — one row per detection with model, class,
  confidence and bounding box.
- **Detectors + a classifier** — four YOLO detectors draw boxes; the vehicle
  CNN labels the whole frame with a banner.

## Models

| # | Model | Framework | Weights | Output |
|---|-------|-----------|---------|--------|
| 1 | Helmet Violation Detection | YOLOv11m | `models/helmet.pt` | Plate, WithHelmet, WithoutHelmet |
| 2 | Driver Monitoring System (DMS) | YOLOv8n | `models/driver.onnx` | Open/Closed Eye, Cigarette, Phone, Seatbelt |
| 3 | Illegal Parking Detection | YOLOv11m (COCO) | `models/illegalpark.pt` | car, motorcycle, bus, truck, bicycle |
| 4 | Traffic Signal & Sign Violations | YOLOv8m | `models/stopwait.pt` | red-light, stop-line, wrong-side, signs |
| 5 | Vehicle Type Classification (CNN) | Keras (224×224) | `models/complete_model_model.h5` | 15 vehicle types (whole-image) |

The four YOLO weights (`.pt` / `.onnx`) load through Ultralytics `YOLO(...)`.
The vehicle classifier is a Keras CNN loaded with TensorFlow; instead of boxes
it predicts one of **15 classes** — Ambulance, Bicycle, Boat, Bus, Car,
Helicopter, Limousine, Motorcycle, PickUp, Segway, Snowmobile, Tank, Taxi,
Truck, Van — and draws a label banner on the frame.

> **Note on Illegal Parking:** `illegalpark.pt` is a COCO-pretrained detector,
> so its raw output is filtered down to vehicle classes in `app.py`
> (`"keep"` set) for a parking-enforcement use-case.

The dashboard **loads every available model at startup** and **gracefully
handles missing weights** — a model with no real weights file simply shows up
marked `(not loaded)` instead of crashing the app.

## Folder structure

```
traffic-violation/
├── app.py                       # Gradio dashboard (entry point)
├── requirements.txt             # Python dependencies
├── README.md                    # This file (also the HF Space config)
├── .gitattributes               # Git LFS tracking for *.pt / *.onnx / *.h5
├── models/
│   ├── helmet.pt                # YOLOv11m helmet detector (LFS)
│   ├── driver.onnx              # YOLOv8n driver-monitoring (LFS)
│   ├── illegalpark.pt           # COCO YOLOv11m, filtered to vehicles (LFS)
│   ├── stopwait.pt              # YOLOv8m signal/sign detector (LFS)
│   └── complete_model_model.h5  # Keras vehicle classifier (LFS)
├── configs/                     # Class names per detector (data.yaml style)
│   ├── helmet.yaml
│   ├── driver.yaml
│   └── illegal_parking.yaml
└── notebooks/                   # Original training notebooks (reference)
    ├── helmetviolation.ipynb
    ├── illegal-parking-detection.ipynb
    ├── stopwait.ipynb
    └── classification_vehicle/  # CNN training + prediction notebooks
```

### Adding or swapping a model

Drop the weights into `models/`, add an entry to the `MODELS` list at the top
of `app.py` (set `"type": "detect"` for YOLO or `"type": "classify"` for a
Keras CNN), and — for detectors — optionally add a `configs/<id>.yaml` with the
class `names`. The model then appears in both tabs automatically.

## Run locally

```bash
# Setup

## 1. Install Git LFS (one-time setup)

```bash
git lfs install
```

## 2. Clone the repository

```bash
git clone https://github.com/Rakesh2901/Traffic-Violation-Detection-Systems.git
cd Traffic-Violation-Detection-Systems
```

## 3. Download the model weights

```bash
git lfs pull
```

> This downloads the actual `.pt` model files instead of the small Git LFS pointer files.

## 4. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 6. Run the application

```bash
python app.py
```
```

Open the printed URL (default http://localhost:7860).

> If a model shows `(not loaded)`, its weights file is missing or is an
> un-pulled Git LFS pointer — run `git lfs pull` and restart.


The Space rebuilds from the `README.md` front-matter (`sdk: gradio`,
`app_file: app.py`, `python_version: "3.11"`) and `requirements.txt`.

### Notes for Spaces
- `python_version: "3.14"` is pinned because the stdlib `audioop` module
  (used transitively by Gradio) was removed in 3.13.
- `tensorflow-cpu==2.15.0` ships Keras 2.x, which loads the legacy `.h5`
  classifier; `numpy` is pinned `<2.0` for TensorFlow compatibility.
- Confirm weights are LFS-tracked before pushing — `git lfs ls-files` should
  list every `.pt` / `.onnx` / `.h5`.
- Free CPU Spaces are fine; pick a GPU Space for faster video processing; All the model are here trained on Nvidia RTX 4050.
