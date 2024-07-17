from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import pickle

class PredictRequest(BaseModel):
    text: str

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://webparkingiot.6l9.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model('model-analisis.h5')

tokenizer = None

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict(request: PredictRequest):    
    text = request.text
    
    new_sequence = tokenizer.texts_to_sequences([text])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=57)
    prediction = model.predict(new_padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)

    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_sentiment = sentiment_labels[predicted_class[0]]
    return {"prediction": predicted_sentiment}
   
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)