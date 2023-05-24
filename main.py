from fastapi import FastAPI,File,UploadFile
import uvicorn
from pydantic import BaseModel
import numpy as np
from PIL import Image
from io import BytesIO
from pickle4 import pickle
import tensorflow as tf

app=FastAPI()

MODEL=tf.keras.models.load_model(r"C:\Users\jaswa\PycharmProjects\fastApiProject\models\1")
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
def img_to_array(data):
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.get("/")
def fun():
    return "Hello jaswanth kumar"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = img_to_array(await file.read())
    image = np.expand_dims(image,0)
    prediction = MODEL.predict(image)
    class_predicted = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])*100

    return {"class": class_predicted,
            "prediction_probability": float(confidence)}


if(__name__=="__main__"):
    uvicorn.run(app)