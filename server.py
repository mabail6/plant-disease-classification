from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image


import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("saved_models/2/plant_disease_detetion_classifier_2.h5")

CLASS_NAMES = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Mosaic_virus', 'Septoria_leaf_spot', 
'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_healthy', 'Yellow_Leaf_Curl_Virus']

@app.get("/ping")
async def ping():
    return "Hello, I am alive."

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict/")
async def create_upload_file(file: UploadFile):
    image = read_file_as_image(await file.read())
    print(image)
    img_batch = np.expand_dims(image,0)
    print(img_batch)
    prediction = MODEL.predict(img_batch)
    print(prediction)
    pass

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)