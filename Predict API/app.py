from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

app = FastAPI()

# Load the model
model = load_model('model.h5')

# Class labels
label_mapping = {
    0: "HDPE",
    1: "LDPE",
    2: "PET",
    3: "PP",
    4: "PS",
    5: "PVC",
    6: "Other"
}

# Prediction function
def predict_plastic_type(model, img_array, label_mapping, threshold=0.8):
    """
    Function to predict the type of plastic from an image.

    Args:
        model: Loaded TensorFlow model.
        img_array: Processed image array.
        label_mapping: Dictionary for label mapping.
        threshold: Confidence threshold to ensure valid prediction.

    Returns:
        dict: Prediction result containing label, confidence, and all class probabilities.
    """
    predictions = model.predict(img_array)
    confidence = np.max(predictions)  # Highest confidence
    predicted_class_index = np.argmax(predictions)  # Predicted class index

    # If confidence is below threshold or index out of bounds, return "Other"
    predicted_label = label_mapping.get(predicted_class_index, "Other")
    if confidence < threshold or predicted_class_index >= len(label_mapping):
        predicted_label = "Other"

    # Prepare the result
    result = {
        "predicted_class": predicted_label,
        "confidence": float(confidence),
        "all_predictions": {
            label_mapping[i]: float(predictions[0][i]) if i in label_mapping else "N/A"
            for i in range(len(predictions[0]))
        },
    }
    return result

@app.post("/predict_image")
async def predict_image(file: UploadFile):
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Read and process the image
        contents = await file.read()
        img = load_img(BytesIO(contents), target_size=(224, 224))  # Adjust to the model's input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Predict the plastic type
        result = predict_plastic_type(model, img_array, label_mapping)

        # Return prediction result
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)})
