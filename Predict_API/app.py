from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import base64
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, SafetySetting, Tool
from vertexai.preview.generative_models import grounding

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

# Inisialisasi Vertex AI
vertexai.init(project="ecoquest-442306", location="us-central1")

# Fungsi untuk menghasilkan konten dari Vertex AI
def generate(plastic_type):

    # Memilih model generatif
    model = GenerativeModel("gemini-1.5-flash-002")

    # Format prompt dengan variabel plastic_type
    prompt = f"Buat sebuah kreasi daur ulang untuk mengolah plastik {plastic_type} buat dalam paragraf."

    # Memanggil model untuk menghasilkan respons
    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    # Menggabungkan hasil respons ke dalam satu variabel
    hasil = ""
    for response in responses:
        hasil += response.text

    return hasil

# Konfigurasi generasi
generation_config = {
    "max_output_tokens": 784,
    "temperature": 0.2,
    "top_p": 0.8,
}

# Konfigurasi pengaturan keamanan
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

# Fungsi prediksi untuk jenis plastik
def predict_plastic_type(model, img_array, label_mapping, threshold=0.8):
    """
    Function to predict the type of plastic from an image and generate recycling ideas.

    Args:
        model: Loaded TensorFlow model.
        img_array: Processed image array.
        label_mapping: Dictionary for label mapping.
        threshold: Confidence threshold to ensure valid prediction.

    Returns:
        dict: Prediction result containing label, confidence, and all class probabilities.
    """
    # Prediksi model
    predictions = model.predict(img_array)
    confidence = np.max(predictions)  # Confidence tertinggi
    predicted_class_index = np.argmax(predictions)  # Index kelas prediksi

    # Tentukan label prediksi atau "Other" jika confidence rendah atau index invalid
    predicted_label = label_mapping.get(predicted_class_index, "Other")
    if confidence < threshold or predicted_class_index >= len(label_mapping):
        predicted_label = "Other"

    if predicted_label == "Other":
        rekomendasi = "Ini bukan jenis plastik ya."
    else:
        # Menghasilkan ide daur ulang menggunakan Vertex AI
        rekomendasi = generate(predicted_label)

    # Hasil akhir
    result = {
        "Jenis Plastik": predicted_label,
        "confidence": float(confidence*100),
        "rekomendasi": rekomendasi,
    }

    return result


@app.post("/predict_image")
async def predict_image(file: UploadFile):
   
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
