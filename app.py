import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download

# -----------------------------
# 1. Load model and define config
# -----------------------------
IMG_SIZE = 299

model_path = hf_hub_download(
    repo_id="Janeworld/Janeworld_brain-tumor-mri-model",
    filename="brain_tumor_classifier.keras",
)

model = keras.models.load_model(model_path)

# This must match training order exactly
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# preprocess_input = keras.applications.xception.preprocess_input


# -----------------------------
# 2. Preprocessing function
# -----------------------------
def preprocess_image(image):
    img = tf.convert_to_tensor(image)

    # Handle grayscale 2D image: (H, W)
    if len(img.shape) == 2:
        img = tf.expand_dims(img, axis=-1)

    # Handle grayscale 1-channel image: (H, W, 1)
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)

    # Handle RGBA image: (H, W, 4)
    elif img.shape[-1] == 4:
        img = img[..., :3]

    # Resize to match training
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert to float32
    img = tf.cast(img, tf.float32)

    # Match notebook preprocessing exactly
    img = img / 255.0

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)
    return img


# -----------------------------
# 3. Inference function
# -----------------------------
def predict_brain_tumor(image):
    img = preprocess_image(image)

    preds = model.predict(img, verbose=0)
    probs = preds[0]   # already softmax probabilities

    top_idx = int(np.argmax(probs))
    predicted_class = class_names[top_idx]
    confidence = float(probs[top_idx])

    print("Raw probabilities:", probs)
    print("Predicted index:", top_idx)
    print("Predicted class:", predicted_class)

    prob_dict = {
        class_names[i]: float(probs[i]) for i in range(len(class_names))
    }

    result_text = f"Predicted: {predicted_class} (confidence: {confidence:.2%})"

    disclaimer = (
        "\n This is a research/demo model only.\n"
        "It is NOT a medical device and must NOT be used for real diagnosis yet."
    )

    return result_text + disclaimer, prob_dict


# -----------------------------
# 4. Build Gradio interface
# -----------------------------
title = "Brain Tumor MRI Classifier (Demo)"
description = (
    "Upload a brain MRI slice. The model will predict one of:\n"
    f"{', '.join(class_names)}.\n\n"
    "This is for learning/demo purposes ONLY, not for real medical use."
)

demo = gr.Interface(
    fn=predict_brain_tumor,
    inputs=gr.Image(type="numpy", label="Upload MRI Image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Class probabilities"),
    ],
    title=title,
    description=description,
)

if __name__ == "__main__":
    demo.launch()