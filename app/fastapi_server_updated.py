
# fastapi_server.py (updated to match app.py UX changes)
import os
import io
import base64
import traceback
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
from typing import Optional
from inference_updated import list_models, predict, DEVICE

# --- CONFIG ---
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))  # Default to 8000, override via env var
API_KEY = os.environ.get("REMOTE_API_KEY", "mypass123")

app = FastAPI(
    title="YOLO/DETR Inference Server",
    description="Predict endpoint for YOLO/DETR models. Requires x-api-key header if configured."
)

# Allow CORS for local dev and Gradio front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    annotated_image_base64: Optional[str] = None
    detections: list = []
    meta: dict = {}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Return JSON errors for easier frontend debugging
    return JSONResponse(status_code=500, content={"error": str(exc), "trace": traceback.format_exc().splitlines()[-5:]})

@app.post('/predict', response_model=PredictResponse)
async def predict_endpoint(
    model_name: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    # Simple header auth
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail='Invalid API Key')

    models = list_models()
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f'Unknown model {model_name}. Available: {models}')

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)

    # Run prediction (inference.predict is expected to return annotated np.array and detections list)
    try:
        annotated, dets = predict(model_name, image_np, conf, iou, imgsz)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference error: {e}')

    encoded = None
    if annotated is not None:
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format='JPEG')
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')

    meta = {
        'model': model_name,
        'device': DEVICE,
        'imgsz': imgsz,
        'conf': conf,
        'iou': iou,
    }

    return JSONResponse({
        'detections': dets,
        'annotated_image_base64': encoded,
        'meta': meta,
    })

@app.get('/models')
async def models():
    return {'models': list_models(), 'device': DEVICE}

@app.get('/model/{model_name}')
async def model_info(model_name: str):
    models = list_models()
    if model_name not in models:
        raise HTTPException(status_code=404, detail='Unknown model')
    # Provide a small info blob; inference module can be extended to provide more
    info = {
        'name': model_name,
        'description': f'Model {model_name} — available for inference',
    }
    return {'model': info, 'device': DEVICE}

@app.get('/health')
async def health():
    return {'status': 'ok', 'device': DEVICE, 'models_count': len(list_models())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_server_updated:app", host="0.0.0.0", port=SERVER_PORT, reload=False)
'''


# fastapi_server.py (corrected)
import os
import io
import base64
import traceback
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

from inference_updated import list_models, predict, DEVICE

# --- CONFIG ---
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))  # Default to 8000, override via env var
API_KEY = os.environ.get("REMOTE_API_KEY", "mypass123")

app = FastAPI(
    title="YOLO/DETR Inference Server",
    description="Predict endpoint for YOLO/DETR models. Requires x-api-key header if configured."
)

# Allow CORS for local dev and Gradio front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (dev mode)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Response Models ---
class PredictResponse(BaseModel):
    annotated_image_base64: Optional[str] = None
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


# --- Global exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Return JSON errors for easier frontend debugging
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "trace": traceback.format_exc().splitlines()[-5:],
        },
    )


# --- Predict endpoint ---
@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    model_name: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    # --- API key check ---
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    models = list_models()
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model_name}. Available: {models}",
        )

    # --- Read and decode uploaded image ---
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    image_np = np.array(image)

    # --- Run inference ---
    try:
        result = predict(model_name, image_np, conf, iou, imgsz)
        if isinstance(result, tuple) and len(result) == 3:
            annotated, dets, meta = result
        elif isinstance(result, tuple) and len(result) == 2:
            annotated, dets = result
            meta = {}
        else:
            raise RuntimeError("Unexpected predict() return signature")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # --- Encode annotated image ---
    encoded = None
    if annotated is not None:
        try:
            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="JPEG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            encoded = None

    # --- Meta info ---
    meta.update(
        {
            "model": model_name,
            "device": DEVICE,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
        }
    )

    return {
        "detections": dets,
        "annotated_image_base64": encoded,
        "meta": meta,
    }


# --- Utility endpoints ---
@app.get("/models")
async def models():
    return {"models": list_models(), "device": DEVICE}


@app.get("/model/{model_name}")
async def model_info(model_name: str):
    models = list_models()
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Unknown model")
    info = {
        "name": model_name,
        "description": f"Model {model_name} — available for inference",
    }
    return {"model": info, "device": DEVICE}


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "models_count": len(list_models())}


# --- Entrypoint ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=SERVER_PORT, reload=False)
'''