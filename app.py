import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# -----------------------------
# 1. Load model and define config
# -----------------------------
IMG_SIZE = 224

# Load your trained model
model = keras.models.load_model("brain_tumor_classifier_tf.keras")

# IMPORTANT: use the same class order you had during training
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
# If your train_ds.class_names was different, paste that exact list here.

# If you used MobileNetV2, we may also want its preprocess_input
preprocess_input = keras.applications.mobilenet_v2.preprocess_input


# -----------------------------
# 2. Preprocessing function
# -----------------------------
def preprocess_image(image):
    """
    Gradio gives a PIL.Image or numpy array.
    We resize, convert to RGB, and scale just like in training.
    """
    # Convert to tf tensor
    img = tf.convert_to_tensor(image, dtype=tf.uint8)
    # Ensure 3 channels (RGB)
    if img.shape[-1] == 4:
        img = img[..., :3]  # drop alpha
    elif img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)

    # Resize
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert to float32
    img = tf.cast(img, tf.float32)

    # If you used keras.preprocessing.image_dataset_from_directory
    # with MobileNetV2 + preprocess_input, we should call it here:
    img = preprocess_input(img)

    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    img = tf.expand_dims(img, axis=0)
    return img


# -----------------------------
# 3. Inference function
# -----------------------------
def predict_brain_tumor(image):
    """
    image: input from Gradio (PIL or np.array)
    Returns: predicted label and class probabilities
    """
    # Preprocess
    img = preprocess_image(image)

    # Model prediction
    preds = model.predict(img)
    probs = tf.nn.softmax(preds[0]).numpy()

    # Get top prediction
    top_idx = int(np.argmax(probs))
    predicted_class = class_names[top_idx]
    confidence = float(probs[top_idx])

    # Build a dictionary of class probabilities for Gradio
    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    # Text output string
    result_text = f"Predicted: {predicted_class} (confidence: {confidence:.2%})"

    # ⚠️ Medical disclaimer
    disclaimer = (
        "\n\n⚠️ This is a research/demo model only.\n"
        "It is NOT a medical device and must NOT be used for real diagnosis."
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
    examples=None,  # you can add example image paths here later
)

if __name__ == "__main__":
    demo.launch(share=True)
