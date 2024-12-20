from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uuid
from pathlib import Path
from model.gpt_model import generate_caption as gpt_generate
from model.vit_decoder import generate_caption as vit_decoder_generate
from model.vit_lstm import generate_caption as vit_lstm_generate

BASE_DIR = Path(__file__).resolve(strict=True).parent
IMAGEDIR = f"{BASE_DIR}/model/images/"

app = FastAPI()

class TextIn(BaseModel):
  text: str

class PredictionOut(BaseModel):
  captions: str

@app.get("/")
def read_root():
  return {"Model Version" : "1", "Model Task" : "Image Captioning"}

@app.post("/predict/")
async def upload_and_predict(file: UploadFile = File(...)):
    # Generate a unique filename
    file_id = uuid.uuid4()
    file_name = f"{file_id}.jpg"
    file_path = f"{IMAGEDIR}{file_name}"

    # Read and save the uploaded file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Call the prediction function
    captions_gpt = gpt_generate(file_path)
    captions_vit_decoder = vit_decoder_generate(file_path)
    captions_vit_lstm = vit_lstm_generate(file_path)

    # Return the prediction
    return {
            "captions_gpt": captions_gpt,
            "captions_vit_decoder": captions_vit_decoder,
            "captions_vit_lstm": captions_vit_lstm
    }


@app.post("/predict/vit-gpt/")
async def upload_and_predict(file: UploadFile = File(...)):
    # Generate a unique filename
    file_id = uuid.uuid4()
    file_name = f"{file_id}.jpg"
    file_path = f"{IMAGEDIR}{file_name}"

    # Read and save the uploaded file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Call the prediction function
    captions = gpt_generate(file_path)

    # Return the prediction
    return {"captions": captions}


@app.post("/predict/vit-transformer-decoder/")
async def upload_and_predict(file: UploadFile = File(...)):
    # Generate a unique filename
    file_id = uuid.uuid4()
    file_name = f"{file_id}.jpg"
    file_path = f"{IMAGEDIR}{file_name}"

    # Read and save the uploaded file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Call the prediction function
    captions = vit_decoder_generate(file_path)

    # Return the prediction
    return {"captions": captions}

@app.post("/predict/vit-lstm/")
async def upload_and_predict(file: UploadFile = File(...)):
    # Generate a unique filename
    file_id = uuid.uuid4()
    file_name = f"{file_id}.jpg"
    file_path = f"{IMAGEDIR}{file_name}"

    # Read and save the uploaded file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Call the prediction function
    captions = vit_lstm_generate(file_path)

    # Return the prediction
    return {"captions": captions}