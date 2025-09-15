
# app.py (fixed - cleaned quotes and wiring)
import os
import io
import base64
import tempfile
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional
import requests
import gradio as gr
import numpy as np
from PIL import Image
import time

try:
    from inference_updated import list_models as local_list_models, predict as local_predict
except ImportError:
    # Fallback if inference module is not available
    def local_list_models():
        return ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    
    def local_predict(model_name, image, conf, iou, imgsz):
        # Dummy fallback - replace with actual implementation
        return image, [], {"inference_time": 0.1}

# --- Config ---
DEFAULT_USE_REMOTE = os.getenv("USE_REMOTE", "false").lower() == "true"
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "http://127.0.0.1:7860/predict").rstrip("/")
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "mypass123")

def _remote_models_url():
    try:
        p = urlparse(REMOTE_API_URL)
        base = f"{p.scheme}://{p.netloc}"
        return base + "/models"
    except Exception:
        return "http://127.0.0.1:7860/models"

# --- Helpers ---
def _to_jpeg_bytes(np_image: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    Image.fromarray(np_image).save(buf, format="JPEG")
    buf.seek(0)
    return buf

def _dets_to_table(dets):
    rows = []
    for d in dets or []:
        box = d.get("box_xyxy", [None]*4)
        rows.append([
            d.get("cls_id"),
            d.get("cls_name"),
            round(float(d.get("conf", 0.0)), 4),
            *[round(float(x), 2) if x is not None else None for x in box]
        ])
    return rows

def _save_temp_image(np_img: np.ndarray) -> str:
    if np_img is None:
        return ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    Image.fromarray(np_img).save(tmp, format="JPEG")
    tmp.flush()
    return tmp.name

def _curl_preview(model_name, conf, iou, imgsz):
    models_url = _remote_models_url()
    predict_url = REMOTE_API_URL
    header = f"-H 'x-api-key: {REMOTE_API_KEY}'" if REMOTE_API_KEY else ""
    return (
        f"# Check server models\n"
        f"curl -s {models_url}\n\n"
        f"# Predict (replace path/to/img.jpg)\n"
        f"curl -X POST {header} \\\n"
        f"  -F 'model_name={model_name}' \\\n"
        f"  -F 'conf={conf}' \\\n"
        f"  -F 'iou={iou}' \\\n"
        f"  -F 'imgsz={imgsz}' \\\n"
        f"  -F 'file=@path/to/img.jpg' \\\n"
        f"  {predict_url}\n"
    )

# --- Inference paths ---
def run_local(model_name, image, conf, iou, imgsz):
    start = time.time()
    annotated, dets, meta = local_predict(model_name, image, conf, iou, imgsz)
    elapsed = meta.get("inference_time", time.time() - start) if isinstance(meta, dict) else (time.time() - start)
    return annotated, dets, elapsed

def run_remote(model_name, image, conf, iou, imgsz):
    if not REMOTE_API_URL:
        return None, {"error": "REMOTE_API_URL not set"}, 0.0

    files = {"file": ("img.jpg", _to_jpeg_bytes(image), "image/jpeg")}
    data = {"model_name": model_name, "conf": str(conf), "iou": str(iou), "imgsz": str(imgsz)}
    headers = {}
    if REMOTE_API_KEY:
        headers["x-api-key"] = REMOTE_API_KEY

    start = time.time()
    try:
        resp = requests.post(REMOTE_API_URL, files=files, data=data, headers=headers, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return None, {"error": str(e)}, 0.0
    elapsed = time.time() - start

    j = resp.json()
    annotated = None
    if "annotated_image_base64" in j and j["annotated_image_base64"]:
        decoded = base64.b64decode(j["annotated_image_base64"])
        annotated = np.array(Image.open(io.BytesIO(decoded)))
    return annotated, j.get("detections", []), j.get("meta", {"inference_time": elapsed})

def decide_and_run(use_remote, model_name, image, conf, iou, imgsz):
    if image is None:
        return None, {}, [], None, gr.update(value="Upload an image to run detection.", visible=True), gr.update(value="")

    if use_remote:
        annotated, dets, meta = run_remote(model_name, image, conf, iou, imgsz)
        elapsed = meta.get("inference_time", 0.0) if isinstance(meta, dict) else 0.0
    else:
        annotated, dets, elapsed = run_local(model_name, image, conf, iou, imgsz)

    if isinstance(dets, dict) and dets.get("error"):
        msg = dets.get("error")
        return None, dets, [], None, gr.update(value=f"❌ {msg}", visible=True), gr.update(value="")

    table = _dets_to_table(dets)
    download_path = _save_temp_image(annotated) if annotated is not None else None
    curl_text = _curl_preview(model_name, conf, iou, imgsz)
    status_text = f"✅ Done. Time: {elapsed:.2f}s — {len(table)} detections."
    return annotated, dets, table, download_path, gr.update(value=status_text, visible=True), gr.update(value=curl_text)

# --- Dynamic model list ---
# Fixed the type annotation for Python < 3.10 compatibility
def fetch_models(use_remote: bool, override_url: Optional[str] = None):
    if not use_remote:
        choices = local_list_models()
        device = "local"
        return choices, f"**Device**: local (uses your machine)"
    try:
        url = override_url.rstrip("/") + "/models" if override_url else _remote_models_url()
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        choices = data.get("models", [])
        device = data.get("device", "remote")
        return choices, f"**Device (remote)**: {device} (from {url})"
    except Exception as e:
        choices = local_list_models()
        return choices, f"**Device**: remote unavailable ({e}); falling back to local list"

# --- UI ---
with gr.Blocks(title="YOLO Model Switcher", theme="soft") as demo:
    gr.Markdown(
        """ 
        # 🪵 Wood Defects Detection — YOLO Switcher
        Upload an image, pick a model, and go brrr.
        Toggle **Remote API** to hit your FastAPI server instead of running local.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            use_remote = gr.Checkbox(label="Use Remote API", value=DEFAULT_USE_REMOTE, info="Hit FastAPI server instead of local inference.")
            remote_url_input = gr.Textbox(label="Remote API URL", value=REMOTE_API_URL, interactive=True, info="Change the remote API base URL (e.g. http://127.0.0.1:7860)")
            api_key_input = gr.Textbox(label="Remote API Key (x-api-key)", value=REMOTE_API_KEY, interactive=True, type="password")
            device_md = gr.Markdown(value="")

            model_dd = gr.Dropdown(choices=[], value=None, label="Model", interactive=True)
            conf = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence", info="Min score to keep detections")
            iou = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="IoU", info="NMS IoU threshold")
            imgsz = gr.Slider(320, 1280, value=640, step=32, label="Image size", info="Inference resolution")

            refresh_btn = gr.Button("🔄 Refresh Models")
            run_btn = gr.Button("🚀 Run")
            sample_btn = gr.Dropdown(choices=["None", "sample_wood1.jpg", "sample_wood2.jpg"], value="None", label="Load sample image")
            paste_url = gr.Textbox(label="Paste image URL and press enter", placeholder="https://.../image.jpg", interactive=True)

        with gr.Column(scale=3):
            in_img = gr.Image(type="numpy", label="Input image")
            out_img = gr.Image(type="numpy", label="Detections (annotated)")
            status = gr.Markdown(visible=False)
            curl_box = gr.Code(label="cURL (remote API)")

    with gr.Row():
        out_json = gr.JSON(label="Detections JSON")
        out_table = gr.Dataframe(
            headers=["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
            label="Detections Table",
            wrap=True,
            interactive=False,
            row_count=(0, "dynamic"),
        )

    download_file = gr.File(label="Download annotated image")

    # --- Wiring ---
    def _init_models(initial_remote, remote_url, api_key):
        global REMOTE_API_URL, REMOTE_API_KEY
        REMOTE_API_URL = remote_url or REMOTE_API_URL
        REMOTE_API_KEY = api_key or REMOTE_API_KEY
        choices, md = fetch_models(initial_remote, override_url=remote_url)
        val = choices[0] if choices else None
        return gr.update(choices=choices, value=val), md

    model_choices, md = fetch_models(DEFAULT_USE_REMOTE)
    model_dd.value = model_choices[0] if model_choices else None
    model_dd.choices = model_choices
    device_md.value = md

    use_remote.change(
        fn=_init_models,
        inputs=[use_remote, remote_url_input, api_key_input],
        outputs=[model_dd, device_md],
        queue=False,
    )

    refresh_btn.click(
        fn=_init_models,
        inputs=[use_remote, remote_url_input, api_key_input],
        outputs=[model_dd, device_md],
        queue=False,
    )

    # sample image loader
    def _load_sample(name):
        if not name or name == "None": 
            return None
        path = Path("samples") / name
        if path.exists():
            return np.array(Image.open(path).convert('RGB'))
        return None

    sample_btn.change(fn=_load_sample, inputs=[sample_btn], outputs=[in_img])

    # paste URL loader
    def _load_url(url):
        if not url: 
            return None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return np.array(Image.open(io.BytesIO(r.content)).convert('RGB'))
        except Exception as e:
            return None

    paste_url.submit(fn=_load_url, inputs=[paste_url], outputs=[in_img])

    # Run on button click or image change
    def _run_and_update(use_remote_val, model_val, img, conf_val, iou_val, imgsz_val, remote_url, api_key):
        # update global remote values from UI
        global REMOTE_API_URL, REMOTE_API_KEY
        REMOTE_API_URL = remote_url or REMOTE_API_URL
        REMOTE_API_KEY = api_key or REMOTE_API_KEY
        res = decide_and_run(use_remote_val, model_val, img, conf_val, iou_val, imgsz_val)
        # decide_and_run returns (annotated, dets, table, download_path, status_update, curl_update)
        if res is None:
            return None, {}, [], None, gr.update(value="Upload an image to run detection.", visible=True), gr.update(value="")
        annotated, dets, table, download_path, status_update, curl_update = res
        return annotated, dets, table, download_path, status_update, curl_update

    run_btn.click(
        fn=_run_and_update,
        inputs=[use_remote, model_dd, in_img, conf, iou, imgsz, remote_url_input, api_key_input],
        outputs=[out_img, out_json, out_table, download_file, status, curl_box],
        api_name="run_detection",
    )

    in_img.change(
        fn=_run_and_update,
        inputs=[use_remote, model_dd, in_img, conf, iou, imgsz, remote_url_input, api_key_input],
        outputs=[out_img, out_json, out_table, download_file, status, curl_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
'''

# app.py (rewritten, robust, with main())
import os
import io
import base64
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

import requests
import numpy as np
from PIL import Image

import gradio as gr

# --- Optional project-specific inference module (fallbacks provided) ---
try:
    from inference_updated import list_models as local_list_models, predict as local_predict
except Exception:
    # Fallback implementations if your inference module isn't available locally.
    def local_list_models() -> List[str]:
        return ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

    def local_predict(model_name: str, image_np: np.ndarray, conf: float, iou: float, imgsz: int) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Dummy local predictor fallback:
          - returns the input image unchanged
          - empty detections list
          - meta dict with inference_time
        Replace with your actual local inference function signature.
        """
        start = time.time()
        # trivial "processing" delay to mimic inference
        time.sleep(0.05)
        return image_np, [], {"inference_time": time.time() - start}


# --- Config ---
DEFAULT_USE_REMOTE = os.getenv("USE_REMOTE", "false").lower() == "true"
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "http://127.0.0.1:7860/predict").rstrip("/") if os.getenv("REMOTE_API_URL") else ""
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "")


def _remote_models_url() -> str:
    """
    Derive a models listing endpoint from REMOTE_API_URL.
    """
    try:
        if not REMOTE_API_URL:
            return "http://127.0.0.1:7860/models"
        from urllib.parse import urlparse

        p = urlparse(REMOTE_API_URL)
        base = f"{p.scheme}://{p.netloc}"
        return base + "/models"
    except Exception:
        return "http://127.0.0.1:7860/models"


# --- Helpers ---
def _to_jpeg_bytes(np_image: np.ndarray) -> bytes:
    """
    Convert an HxWxC numpy image (uint8 or float) into JPEG bytes.
    """
    if np_image is None:
        return b""
    try:
        if np_image.dtype != np.uint8:
            # scale floats to 0-255 if necessary
            if np_image.dtype in (np.float32, np.float64):
                img = np.clip(np_image * 255.0, 0, 255).astype(np.uint8)
            else:
                img = np_image.astype(np.uint8)
        else:
            img = np_image
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:
        return b""


def _dets_to_table(dets: List[Dict[str, Any]]) -> List[List[Any]]:
    """
    Convert detection dictionaries into table rows for the DataFrame.
    Supported detection fields assumed:
      - cls_id
      - cls_name
      - conf
      - box_xyxy (x1,y1,x2,y2)
    """
    rows = []
    for d in dets or []:
        box = d.get("box_xyxy", [None, None, None, None])
        # safe conversion of values
        cls_id = d.get("cls_id")
        cls_name = d.get("cls_name")
        conf = d.get("conf", 0.0)
        try:
            conf_f = round(float(conf), 4)
        except Exception:
            conf_f = 0.0
        coords = []
        for x in box:
            try:
                coords.append(round(float(x), 2) if x is not None else None)
            except Exception:
                coords.append(None)
        rows.append([cls_id, cls_name, conf_f, *coords])
    return rows


def _save_temp_image(np_img: np.ndarray) -> Optional[str]:
    """Save a numpy image to a temp JPEG file and return the path (caller must delete if desired)."""
    if np_img is None:
        return None
    try:
        fd, name = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        if np_img.dtype != np.uint8:
            # scale if float
            if np_img.dtype in (np.float32, np.float64):
                arr = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np_img.astype(np.uint8)
        else:
            arr = np_img
        Image.fromarray(arr).save(name, format="JPEG")
        return name
    except Exception:
        return None


def _curl_preview(model_name: str, conf: float, iou: float, imgsz: int) -> str:
    """Return a short curl snippet the user can run to replicate remote inference."""
    models_url = _remote_models_url()
    predict_url = REMOTE_API_URL or "http://127.0.0.1:7860/predict"
    header_part = f"-H 'x-api-key: {REMOTE_API_KEY}'" if REMOTE_API_KEY else ""
    return (
        f"# Check server models\n"
        f"curl -s {models_url}\n\n"
        f"# Predict (replace path/to/img.jpg)\n"
        f"curl -X POST {header_part} \\\n"
        f"  -F 'model_name={model_name}' \\\n"
        f"  -F 'conf={conf}' \\\n"
        f"  -F 'iou={iou}' \\\n"
        f"  -F 'imgsz={imgsz}' \\\n"
        f"  -F 'file=@path/to/img.jpg' \\\n"
        f"  {predict_url}\n"
    )


# --- Inference paths ---
def run_local(model_name: str, image: np.ndarray, conf: float, iou: float, imgsz: int) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Call your local_predict function. Ensure returned annotated image is a numpy array.
    local_predict signature expected: (model_name, image, conf, iou, imgsz) -> (annotated_np, detections, meta)
    """
    start = time.time()
    try:
        annotated, dets, meta = local_predict(model_name, image, conf, iou, imgsz)
        if annotated is not None and not isinstance(annotated, np.ndarray):
            # try to coerce PIL image to numpy
            try:
                annotated = np.array(annotated)
            except Exception:
                annotated = None
        if meta is None:
            meta = {"inference_time": time.time() - start}
        elif isinstance(meta, dict) and "inference_time" not in meta:
            meta["inference_time"] = time.time() - start
        return annotated, dets or [], meta
    except Exception as e:
        return None, [], {"error": str(e), "inference_time": time.time() - start}


def run_remote(model_name: str, image: np.ndarray, conf: float, iou: float, imgsz: int) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Post image to remote server. Expecting JSON with keys:
      - annotated_image_base64 (optional)
      - detections (list)
      - meta (dict)
    """
    if not REMOTE_API_URL:
        return None, [], {"error": "REMOTE_API_URL not set", "inference_time": 0.0}

    jpeg_bytes = _to_jpeg_bytes(image)
    if not jpeg_bytes:
        return None, [], {"error": "Failed to encode image", "inference_time": 0.0}

    files = {"file": ("img.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}
    data = {"model_name": model_name, "conf": str(conf), "iou": str(iou), "imgsz": str(imgsz)}
    headers = {}
    if REMOTE_API_KEY:
        headers["x-api-key"] = REMOTE_API_KEY

    start = time.time()
    try:
        resp = requests.post(REMOTE_API_URL, files=files, data=data, headers=headers, timeout=60)
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        return None, [], {"error": str(e), "inference_time": time.time() - start}

    annotated = None
    try:
        if "annotated_image_base64" in j and j["annotated_image_base64"]:
            decoded = base64.b64decode(j["annotated_image_base64"])
            annotated = np.array(Image.open(io.BytesIO(decoded)).convert("RGB"))
    except Exception:
        annotated = None

    dets = j.get("detections", []) or []
    meta = j.get("meta", {"inference_time": time.time() - start})
    if isinstance(meta, dict) and "inference_time" not in meta:
        meta["inference_time"] = time.time() - start
    return annotated, dets, meta


def decide_and_run(use_remote: bool, model_name: str, image: Optional[np.ndarray], conf: float, iou: float, imgsz: int) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], List[List[Any]], Optional[str], Dict[str, Any]]:
    """
    Returns:
      annotated_image (or None),
      detections (list),
      table_rows (list),
      download_path (or None),
      status/meta dict for UI
    """
    status = {"ok": False, "msg": "Unknown error", "elapsed": 0.0}
    if image is None:
        status["msg"] = "Upload an image to run detection."
        return None, [], [], None, status

    if model_name is None:
        status["msg"] = "Please choose a model."
        return None, [], [], None, status

    if use_remote:
        annotated, dets, meta = run_remote(model_name, image, conf, iou, imgsz)
    else:
        annotated, dets, meta = run_local(model_name, image, conf, iou, imgsz)

    # meta should be dict
    if meta is None:
        meta = {}
    elapsed = float(meta.get("inference_time", 0.0)) if isinstance(meta, dict) else 0.0

    if isinstance(dets, dict) and dets.get("error"):
        status["msg"] = f"Error: {dets.get('error')}"
        return None, [], [], None, status

    if isinstance(meta, dict) and meta.get("error"):
        status["msg"] = f"Error: {meta.get('error')}"
        status["elapsed"] = elapsed
        return None, [], [], None, status

    table = _dets_to_table(dets)
    download_path = _save_temp_image(annotated) if annotated is not None else None
    status["ok"] = True
    status["msg"] = f"✅ Done. Time: {elapsed:.2f}s — {len(table)} detections."
    status["elapsed"] = elapsed
    status["curl"] = _curl_preview(model_name, conf, iou, imgsz)
    status["download_path"] = download_path
    status["detections"] = dets
    return annotated, dets, table, download_path, status


# --- Model fetcher ---
def fetch_models(use_remote: bool, override_url: Optional[str] = None) -> Tuple[List[str], str]:
    """
    Returns (choices, markdown_info)
    """
    if not use_remote:
        try:
            choices = local_list_models() or []
            return choices, "**Device**: local (uses your machine)"
        except Exception as e:
            return [], f"Local list_models() failed: {e}"

    # Remote attempt
    try:
        url = override_url.rstrip("/") + "/models" if override_url else _remote_models_url()
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        choices = data.get("models", [])
        device = data.get("device", "remote")
        return choices, f"**Device (remote)**: {device} (from {url})"
    except Exception as e:
        # fallback to local list
        try:
            choices = local_list_models() or []
            return choices, f"**Device**: remote unavailable ({e}); falling back to local list"
        except Exception:
            return [], f"Failed to fetch models: {e}"


# --- Gradio UI builder ---
def build_ui() -> gr.Blocks:
    demo = gr.Blocks(title="YOLO Model Switcher", theme="soft")
    with demo:
        gr.Markdown(
            """
            # 🪵 Wood Defects Detection — YOLO Switcher
            Upload an image, pick a model, and go brrr.
            Toggle **Remote API** to hit your FastAPI server instead of running local.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                use_remote = gr.Checkbox(label="Use Remote API", value=DEFAULT_USE_REMOTE, info="Hit FastAPI server instead of local inference.")
                remote_url_input = gr.Textbox(label="Remote API URL", value=REMOTE_API_URL or "http://127.0.0.1:7860", interactive=True, info="Change the remote API base URL (e.g. http://127.0.0.1:7860)")
                api_key_input = gr.Textbox(label="Remote API Key (x-api-key)", value=REMOTE_API_KEY or "", interactive=True, type="password")
                device_md = gr.Markdown(value="")

                model_dd = gr.Dropdown(choices=[], value=None, label="Model", interactive=True)
                conf = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence", info="Min score to keep detections")
                iou = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="IoU", info="NMS IoU threshold")
                imgsz = gr.Slider(320, 1280, value=640, step=32, label="Image size", info="Inference resolution")

                refresh_btn = gr.Button("🔄 Refresh Models")
                run_btn = gr.Button("🚀 Run")
                sample_btn = gr.Dropdown(choices=["None", "sample_wood1.jpg", "sample_wood2.jpg"], value="None", label="Load sample image")
                paste_url = gr.Textbox(label="Paste image URL and press enter", placeholder="https://.../image.jpg", interactive=True)

            with gr.Column(scale=3):
                in_img = gr.Image(type="numpy", label="Input image")
                out_img = gr.Image(type="numpy", label="Detections (annotated)")
                status_md = gr.Markdown(visible=False)
                curl_box = gr.Code(label="cURL (remote API)")

        with gr.Row():
            out_json = gr.JSON(label="Detections JSON")
            out_table = gr.Dataframe(
                headers=["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
                label="Detections Table",
                wrap=True,
                interactive=False,
                row_count=(0, "dynamic"),
            )

        download_file = gr.File(label="Download annotated image")

        # --- Helper actions for UI wiring ---
        def _init_models(initial_remote: bool, remote_url: str, api_key: str):
            # update globals
            global REMOTE_API_URL, REMOTE_API_KEY
            REMOTE_API_URL = (remote_url or REMOTE_api_url_value()) if remote_url else REMOTE_API_URL
            REMOTE_API_KEY = api_key or REMOTE_API_KEY
            choices, md = fetch_models(initial_remote, override_url=remote_url)
            val = choices[0] if choices else None
            return gr.update(choices=choices, value=val), md

        # simple helper to avoid using a variable before declaration
        def REMOTE_api_url_value() -> str:
            return REMOTE_API_URL or "http://127.0.0.1:7860"

        # initial model population
        choices, md_text = fetch_models(DEFAULT_USE_REMOTE)
        model_dd.choices = choices
        model_dd.value = choices[0] if choices else None
        device_md.value = md_text

        # Refresh / toggle functions
        def refresh_models(use_remote_val: bool, remote_url: str, api_key: str):
            choices, md = fetch_models(use_remote_val, override_url=remote_url or None)
            val = choices[0] if choices else None
            return gr.update(choices=choices, value=val), md

        use_remote.change(
            fn=refresh_models,
            inputs=[use_remote, remote_url_input, api_key_input],
            outputs=[model_dd, device_md],
            queue=False,
        )

        refresh_btn.click(
            fn=refresh_models,
            inputs=[use_remote, remote_url_input, api_key_input],
            outputs=[model_dd, device_md],
            queue=False,
        )

        # Sample image loader
        def _load_sample(name: str):
            if not name or name == "None":
                return None
            path = Path("samples") / name
            try:
                if path.exists():
                    return np.array(Image.open(path).convert("RGB"))
            except Exception:
                return None
            return None

        sample_btn.change(fn=_load_sample, inputs=[sample_btn], outputs=[in_img])

        # Paste URL loader
        def _load_url(url: str):
            if not url:
                return None
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
            except Exception:
                return None

        paste_url.submit(fn=_load_url, inputs=[paste_url], outputs=[in_img])

        # Main run handler used by both button and image change
        def _run_and_update(use_remote_val: bool, model_val: str, img: Optional[np.ndarray], conf_val: float, iou_val: float, imgsz_val: int, remote_url_val: str, api_key_val: str):
            # update global remote values from UI
            global REMOTE_API_URL, REMOTE_API_KEY
            REMOTE_API_URL = remote_url_val or REMOTE_API_URL or "http://127.0.0.1:7860"
            REMOTE_API_KEY = api_key_val or REMOTE_API_KEY or ""
            annotated, dets, table, download_path, status = decide_and_run(use_remote_val, model_val, img, conf_val, iou_val, imgsz_val)
            if not status.get("ok", False):
                # any error or message -> show message
                return None, {}, [], None, gr.update(value=status.get("msg", ""), visible=True), gr.update(value=status.get("curl", ""))
            else:
                # set outputs
                out_json_val = dets
                out_table_val = table
                curl_text_val = status.get("curl", "")
                # If a download file exists, set the File component's value to the path.
                return annotated, out_json_val, out_table_val, download_path, gr.update(value=status.get("msg", ""), visible=True), gr.update(value=curl_text_val)

        # wiring for run button
        run_btn.click(
            fn=_run_and_update,
            inputs=[use_remote, model_dd, in_img, conf, iou, imgsz, remote_url_input, api_key_input],
            outputs=[out_img, out_json, out_table, download_file, status_md, curl_box],
            api_name="run_detection",
        )

        # run on image change as well
        in_img.change(
            fn=_run_and_update,
            inputs=[use_remote, model_dd, in_img, conf, iou, imgsz, remote_url_input, api_key_input],
            outputs=[out_img, out_json, out_table, download_file, status_md, curl_box],
        )

    return demo


# --- Entrypoint ---
def main():
    demo = build_ui()
    # server options can be changed here
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()

'''