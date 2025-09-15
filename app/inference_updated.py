
from __future__ import annotations
import os
import time
import logging
from functools import lru_cache
from typing import Any, Dict, List, Tuple
import numpy as np

try:
    import torch
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Install ultralytics & torch (see requirements).") from e

# Optional deps
try:
    import cv2  # for ONNX preprocessing if needed
except ImportError:
    cv2 = None

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

# === CONFIG ===
MODEL_REGISTRY: Dict[str, str] = {
    # Example mappings — override with your own
    "yolo8x": "models/yolo8x.pt",
    "yolo10x": "models/yolo10x.pt",
    "yolo11x": "models/yolo11x.pt",
    "yolo12x": "models/yolo12x.pt",
    "detr_l": "models/detr_l.pt",
    "detr_x": "models/detr_x.pt",
}

# Auto discover weights in models/ dir
models_dir = "models"
if os.path.isdir(models_dir):
    for f in os.listdir(models_dir):
        if f.endswith((".pt", ".onnx")):
            name = os.path.splitext(f)[0]
            MODEL_REGISTRY.setdefault(name, os.path.join(models_dir, f))

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[inference] Using device: {DEVICE}")


# === Internal loaders ===
@lru_cache(maxsize=16)
def _load_ultralytics_model(weights_path: str):
    """Load an Ultralytics YOLO model from weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(weights_path)
    model.to(DEVICE)

    # Warmup pass
    try:
        model.predict(
            np.zeros((32, 32, 3), dtype=np.uint8),
            device=DEVICE,
            imgsz=32,
            verbose=False,
        )
        logger.info(f"[inference] Warmed up {weights_path}")
    except Exception as e:
        logger.warning(f"[inference] Warmup failed for {weights_path}: {e}")
    return model


# === Public API ===
def list_models() -> List[str]:
    """Return available model keys."""
    return list(MODEL_REGISTRY.keys())


def _ultralytics_predict(
    model, image: np.ndarray, conf: float, iou: float, imgsz: int
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Run YOLO inference using Ultralytics."""
    t0 = time.time()
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=DEVICE,
        verbose=False,
    )
    dt = time.time() - t0

    r = results[0]
    annotated = r.plot()  # numpy HWC

    dets: List[Dict[str, Any]] = []
    if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names
        for b, c, s in zip(xyxy, clss, confs):
            dets.append(
                {
                    "cls_id": int(c),
                    "cls_name": names.get(int(c), str(int(c)))
                    if isinstance(names, dict)
                    else str(int(c)),
                    "conf": float(s),
                    "box_xyxy": [float(x) for x in b.tolist()],
                }
            )

    meta = {"inference_time": dt, "device": DEVICE, "num_detections": len(dets)}
    return annotated, dets, meta


def _onnx_predict(
    weights_path: str, image: np.ndarray, conf: float, iou: float, imgsz: int
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Run ONNX inference (skeleton — requires postprocessing)."""
    import onnxruntime as ort

    if cv2 is None:
        raise ImportError("cv2 required for ONNX preprocessing.")

    ort_sess = ort.InferenceSession(
        weights_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # Preprocess
    img = cv2.resize(image, (imgsz, imgsz))
    img = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    t0 = time.time()
    outputs = ort_sess.run(None, {ort_sess.get_inputs()[0].name: img})
    dt = time.time() - t0

    # TODO: implement postprocessing (decode, NMS)
    dets: List[Dict[str, Any]] = []
    annotated = image.copy()
    meta = {"inference_time": dt, "device": DEVICE, "note": "ONNX decode not implemented"}

    return annotated, dets, meta


def predict(
    model_name: str,
    image: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> Tuple[np.ndarray | None, List[Dict[str, Any]], Dict[str, Any]]:
    """Public predict API used by app and server.
    Returns: (annotated_image, detections_list, meta)
    """
    if image is None:
        return None, [], {"error": "No image provided"}

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Known: {list(MODEL_REGISTRY)}"
        )

    weights_path = MODEL_REGISTRY[model_name]
    ext = os.path.splitext(weights_path)[1].lower()

    if ext == ".pt":
        model = _load_ultralytics_model(weights_path)
        return _ultralytics_predict(model, image, conf, iou, imgsz)

    if ext == ".onnx":
        return _onnx_predict(weights_path, image, conf, iou, imgsz)

    raise ValueError(f"Unsupported model file type: {ext}")
'''


from __future__ import annotations
import os
import time
import logging
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import torch
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics + torch must be installed (see requirements.txt)") from e

# Optional deps
try:
    import cv2  # for ONNX preprocessing
except ImportError:
    cv2 = None

# === LOGGING ===
logger = logging.getLogger("inference")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# === CONFIG ===
MODEL_REGISTRY: Dict[str, str] = {
    "yolo8x": "models/yolo8x.pt",
    "yolo10x": "models/yolo10x.pt",
    "yolo11x": "models/yolo11x.pt",
    "yolo12x": "models/yolo12x.pt",
    "detr_l": "models/detr_l.pt",
    "detr_x": "models/detr_x.pt",
}

# Auto discover weights in models/ dir
models_dir = "models"
if os.path.isdir(models_dir):
    for f in os.listdir(models_dir):
        if f.endswith((".pt", ".onnx")):
            name = os.path.splitext(f)[0]
            MODEL_REGISTRY.setdefault(name, os.path.join(models_dir, f))

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[inference] Using device: {DEVICE}")


# === Internal loaders ===
@lru_cache(maxsize=16)
def _load_ultralytics_model(weights_path: str):
    """Load and cache a YOLO model from Ultralytics weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(weights_path)

    # Warmup pass (ignore failures)
    try:
        model.predict(
            np.zeros((32, 32, 3), dtype=np.uint8),
            device=DEVICE,
            imgsz=32,
            verbose=False,
        )
        logger.info(f"[inference] Warmed up model {weights_path}")
    except Exception as e:
        logger.warning(f"[inference] Warmup failed for {weights_path}: {e}")
    return model


# === Public API ===
def list_models() -> List[str]:
    """Return available model keys."""
    return list(MODEL_REGISTRY.keys())


def _ultralytics_predict(
    model, image: np.ndarray, conf: float, iou: float, imgsz: int
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Run YOLO inference using Ultralytics."""
    t0 = time.time()
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=DEVICE,
        verbose=False,
    )
    dt = time.time() - t0

    r = results[0]
    annotated = r.plot()  # numpy HWC

    dets: List[Dict[str, Any]] = []
    if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names

        for b, c, s in zip(xyxy, clss, confs):
            if isinstance(names, dict):
                cls_name = names.get(int(c), str(int(c)))
            elif isinstance(names, (list, tuple)):
                cls_name = names[int(c)] if int(c) < len(names) else str(int(c))
            else:
                cls_name = str(int(c))
            dets.append(
                {
                    "cls_id": int(c),
                    "cls_name": cls_name,
                    "conf": float(s),
                    "box_xyxy": [float(x) for x in b.tolist()],
                }
            )

    meta = {"inference_time": dt, "device": DEVICE, "num_detections": len(dets)}
    return annotated, dets, meta


def _onnx_predict(
    weights_path: str, image: np.ndarray, conf: float, iou: float, imgsz: int
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Run ONNX inference (skeleton — requires postprocessing)."""
    import onnxruntime as ort

    if cv2 is None:
        raise ImportError("cv2 required for ONNX preprocessing.")

    ort_sess = ort.InferenceSession(
        weights_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # Preprocess
    img = cv2.resize(image, (imgsz, imgsz))
    img = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    t0 = time.time()
    outputs = ort_sess.run(None, {ort_sess.get_inputs()[0].name: img})
    dt = time.time() - t0

    # TODO: implement postprocessing (decode, NMS)
    logger.warning("[inference] ONNX predict called — postprocessing not implemented, returning empty detections.")

    dets: List[Dict[str, Any]] = []
    annotated = image.copy()
    meta = {
        "inference_time": dt,
        "device": DEVICE,
        "note": "ONNX decode not implemented",
        "raw_outputs_shape": [o.shape for o in outputs],
    }

    return annotated, dets, meta


def predict(
    model_name: str,
    image: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """Public predict API used by app and server.
    Returns: (annotated_image, detections_list, meta)
    """
    if image is None:
        return None, [], {"error": "No image provided"}

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Known: {list(MODEL_REGISTRY)}")

    weights_path = MODEL_REGISTRY[model_name]
    ext = os.path.splitext(weights_path)[1].lower()

    if ext == ".pt":
        model = _load_ultralytics_model(weights_path)
        return _ultralytics_predict(model, image, conf, iou, imgsz)

    if ext == ".onnx":
        return _onnx_predict(weights_path, image, conf, iou, imgsz)

    raise ValueError(f"Unsupported model file type: {ext}")
'''