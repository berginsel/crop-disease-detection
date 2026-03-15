import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("data/processed/classification_tfds/labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Remedies database
remedies = {

"Pepper__bell___Bacterial_spot": {
"fertilizer": "Balanced NPK fertilizer",
"remedy": "Spray copper-based bactericide",
"prevention": "Avoid overhead watering"
},

"Pepper__bell___healthy": {
"fertilizer": "Organic compost",
"remedy": "No disease detected",
"prevention": "Maintain proper irrigation"
},

"Potato___Early_blight": {
"fertilizer": "Nitrogen-rich fertilizer",
"remedy": "Apply chlorothalonil fungicide",
"prevention": "Rotate crops regularly"
},

"Potato___Late_blight": {
"fertilizer": "Potassium-rich fertilizer",
"remedy": "Spray copper fungicide or mancozeb",
"prevention": "Avoid wet leaves"
},

"Potato___healthy": {
"fertilizer": "Organic compost",
"remedy": "No disease detected",
"prevention": "Maintain soil health"
},

"Tomato_Bacterial_spot": {
"fertilizer": "Balanced fertilizer",
"remedy": "Use copper bactericide spray",
"prevention": "Remove infected leaves"
},

"Tomato_Early_blight": {
"fertilizer": "Nitrogen-balanced fertilizer",
"remedy": "Apply chlorothalonil fungicide",
"prevention": "Crop rotation"
},

"Tomato_Late_blight": {
"fertilizer": "Potassium-rich fertilizer",
"remedy": "Spray mancozeb fungicide",
"prevention": "Avoid excess moisture"
},

"Tomato_Leaf_Mold": {
"fertilizer": "Potassium fertilizer",
"remedy": "Apply copper fungicide",
"prevention": "Improve airflow"
},

"Tomato_Septoria_leaf_spot": {
"fertilizer": "Balanced fertilizer",
"remedy": "Spray chlorothalonil",
"prevention": "Remove infected leaves"
},

"Tomato_Spider_mites_Two_spotted_spider_mite": {
"fertilizer": "Organic compost",
"remedy": "Spray neem oil",
"prevention": "Monitor plants regularly"
},

"Tomato__Target_Spot": {
"fertilizer": "Balanced fertilizer",
"remedy": "Apply fungicide spray",
"prevention": "Improve air circulation"
},

"Tomato__Tomato_YellowLeaf__Curl_Virus": {
"fertilizer": "Potassium fertilizer",
"remedy": "Control whiteflies using neem oil",
"prevention": "Use insect nets"
},

"Tomato__Tomato_mosaic_virus": {
"fertilizer": "Balanced fertilizer",
"remedy": "Remove infected plants",
"prevention": "Disinfect tools"
},

"Tomato_healthy": {
"fertilizer": "Organic compost",
"remedy": "No disease detected",
"prevention": "Maintain proper watering"
}

}

# Load image
img = Image.open("test_leaf.jpg").resize((224,224))
img = np.array(img) / 255.0
img = np.expand_dims(img.astype(np.float32), axis=0)

# Run prediction
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
predicted_class = labels[np.argmax(output)]

# Show result + remedy
disease = predicted_class

print("\nDisease Detected:", disease)

if disease in remedies:
    r = remedies[disease]
    print("Fertilizer:", r["fertilizer"])
    print("Remedy:", r["remedy"])
    print("Prevention:", r["prevention"])
else:
    print("No remedy data available.")