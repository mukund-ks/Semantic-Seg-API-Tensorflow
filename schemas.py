from pydantic import BaseModel

class PredRequest(BaseModel):
    img_base64: str


class PredResponse(BaseModel):
    mask: str

class ExceptionResponse(BaseModel):
    message: str