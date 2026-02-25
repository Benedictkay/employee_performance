from fastapi import FastAPI, File, UploadFile
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI()

# Load your trained model here (replace with your actual model loading code)

model = tf.keras.models.load_model('train_model.keras')

# Define class name for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define a function to preprocess the uploaded image
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((32, 32))  # Resize to 32x32
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array

# Define an endpoint to handle image uploads and make predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')  # Ensure it's in RGB format

    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

    # Make a prediction using the model
    predictions = model.predict(preprocessed_image)
    predicted_class = class_names[np.argmax(predictions)]

    return {"predicted_class": predicted_class}
