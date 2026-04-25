#!/usr/bin/env python3
import argparse
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL_ID = "allenai/MolmoPoint-8B"

app = FastAPI(title="MolmoPoint-8B API")

_model = None
_processor = None
_model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_load_lock = threading.Lock()
_loader_started = False
_state = {
    "status": "starting",
    "detail": "waiting for model loader",
    "device": "unknown",
    "model_id": DEFAULT_MODEL_ID,
}


class PointRequest(BaseModel):
    image_path: str
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=2048)


def _model_device_name():
    return str(_model_device)


def _serialize_points(points):
    serialized = []
    for point in points:
        serialized.append(
            {
                "object_id": int(point[0]),
                "image_index": int(point[1]),
                "x": float(point[2]),
                "y": float(point[3]),
            }
        )
    return serialized


def _load_model_once():
    global _model, _processor

    with _load_lock:
        if _model is not None and _processor is not None:
            return

        _state["status"] = "loading"
        _state["detail"] = "loading MolmoPoint model into memory"

        _processor = AutoProcessor.from_pretrained(
            _state["model_id"],
            trust_remote_code=True,
            padding_side="left",
        )
        model_kwargs = {
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16

        _model = AutoModelForImageTextToText.from_pretrained(
            _state["model_id"],
            **model_kwargs,
        )
        _model.to(_model_device)
        _model.eval()

        _state["status"] = "ready"
        _state["detail"] = "model loaded"
        _state["device"] = _model_device_name()


def _loader_loop():
    while True:
        try:
            _load_model_once()
            return
        except Exception as exc:
            _state["status"] = "error"
            _state["detail"] = str(exc)
            time.sleep(60)


@app.on_event("startup")
def _startup():
    global _loader_started

    if _loader_started:
        return

    _loader_started = True
    threading.Thread(target=_loader_loop, daemon=True).start()


@app.get("/")
def root():
    return dict(_state)


@app.get("/health")
def health():
    if _state["status"] != "ready":
        raise HTTPException(status_code=503, detail=dict(_state))
    return dict(_state)


@app.post("/point")
def point(request: PointRequest):
    if _state["status"] != "ready":
        raise HTTPException(status_code=503, detail=dict(_state))

    image_path = Path(request.image_path)
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")

    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": request.prompt},
                {"type": "image", "image": image},
            ],
        }
    ]

    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        return_pointing_metadata=True,
    )
    metadata = inputs.pop("metadata")

    inputs = {key: value.to(_model_device) if hasattr(value, "to") else value for key, value in inputs.items()}

    generation_kwargs = {"max_new_tokens": request.max_new_tokens}
    if hasattr(_model, "build_logit_processor_from_inputs"):
        generation_kwargs["logits_processor"] = _model.build_logit_processor_from_inputs(inputs)

    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = _model.generate(**inputs, **generation_kwargs)
        else:
            output = _model.generate(**inputs, **generation_kwargs)

    generated_tokens = output[:, inputs["input_ids"].size(1):]
    generated_text = _processor.post_process_image_text_to_text(
        generated_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    points = _model.extract_image_points(
        generated_text,
        metadata["token_pooling"],
        metadata["subpatch_mapping"],
        metadata["image_sizes"],
    )

    return {
        "model_id": _state["model_id"],
        "device": _state["device"],
        "generated_text": generated_text,
        "points": _serialize_points(points),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MolmoPoint-8B HTTP service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _state["model_id"] = args.model_id
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
