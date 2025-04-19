import onnxruntime as ort
import numpy as np
from PIL import Image
import json

# Bild vorbereiten
img = Image.open("test.jpg").resize((224, 224)).convert("RGB")
img_data = np.array(img).astype(np.float32) / 255.0
img_data = np.expand_dims(img_data, axis=0)  # shape: [1, 224, 224, 3]


# ONNX-Modell laden
session = ort.InferenceSession("efficientnet-lite4-11.onnx")

# Inference
inputs = {"images:0": img_data}
outputs = session.run(["Softmax:0"], inputs)[0]

# Label laden
with open("labels_map.txt", "r") as f:
    labels = json.load(f)

# Ergebnis anzeigen
top = outputs[0].argsort()[-5:][::-1]
for i in top:
    print(f"{labels[str(i)]}: {outputs[0][i]:.4f}")
