import base64
import numpy as np
from keras.utils import CustomObjectScope
from keras.models import load_model
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.metrics import iou as model_iou, calc_loss, dice_coef


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredRequest(BaseModel):
    img_base64: str


class PredResponse(BaseModel):
    mask: str

class ExceptionResponse(BaseModel):
    message: str

async def run_model(img_arr: np.ndarray[np.float32]):
    with CustomObjectScope({"iou": model_iou, "dice_coef": dice_coef, "dice_loss": calc_loss}):
        model = load_model("src/weights/model.h5")

    img_thres = img_arr.astype(np.float32) / 255.0

    img_thres -= [0.485, 0.456, 0.406]  # MEAN
    img_thres /= [0.229, 0.224, 0.225]  # STD

    x = np.expand_dims(img_thres, axis=0)

    pred = model.predict(x)[0]
    pred = np.squeeze(pred, axis=-1)
    pred = (pred > 0.5).astype(np.uint8)

    pred_bytes = (pred * 255).astype(np.uint8).tobytes()

    base64_pred = base64.b64encode(pred_bytes).decode("utf-8")  # .decode('utf-8') to convert to str

    return base64_pred


@app.post("/segment", response_model=PredResponse)
async def segment(payload: PredRequest):
    try:
        image_dict = payload.model_dump()
        image_data = base64.b64decode(image_dict["img_base64"])

        with Image.open(BytesIO(image_data)) as pil_img:
            img_arr = np.array(pil_img.convert("RGB"), dtype=np.float32)

        if img_arr.shape != (256, 256, 3):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image size. Expected (256,256,3), got{img_arr.shape}",
            )

        pred_res = await run_model(img_arr)

        res = PredResponse(mask=pred_res)
        res_dict = res.model_dump()

        return JSONResponse(status_code=200, content=res_dict)
    except Exception as e:
        print(f"Error in run_model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error during model run",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    res = ExceptionResponse(message=exc.detail)
    res_dict = res.model_dump()
    return JSONResponse(status_code=exc.status_code, content=res_dict)


@app.get("/")
async def root():
    return {"message": "Semantic Segmentation API - Tensorflow backed model."}
