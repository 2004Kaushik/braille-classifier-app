from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import numpy as np
import uvicorn
import io

app = FastAPI()
model = load_model("braille_classifier.keras")

# Modify based on your model's input shape
IMG_SIZE = (64, 64)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_data = preprocess_image(image_bytes)
        prediction = model.predict(input_data)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        return {"class": predicted_class, "confidence": confidence}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Optional: for local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
