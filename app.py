import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🌱 Crop Disease Detection System")

# remedies database
remedies = {

"Pepper__bell___Bacterial_spot":{
"fertilizer":"Balanced NPK fertilizer",
"remedy":"Spray copper bactericide",
"prevention":"Avoid overhead watering"
},

"Pepper__bell___healthy":{
"fertilizer":"Organic compost",
"remedy":"No disease detected",
"prevention":"Maintain good irrigation"
},

"Potato___Early_blight":{
"fertilizer":"Nitrogen-rich fertilizer",
"remedy":"Apply chlorothalonil fungicide",
"prevention":"Rotate crops"
},

"Potato___Late_blight":{
"fertilizer":"Potassium fertilizer",
"remedy":"Spray mancozeb fungicide",
"prevention":"Avoid wet leaves"
},

"Potato___healthy":{
"fertilizer":"Organic compost",
"remedy":"No disease detected",
"prevention":"Maintain soil health"
},

"Tomato_Bacterial_spot":{
"fertilizer":"Balanced fertilizer",
"remedy":"Copper bactericide spray",
"prevention":"Remove infected leaves"
},

"Tomato_Early_blight":{
"fertilizer":"Nitrogen balanced fertilizer",
"remedy":"Apply chlorothalonil",
"prevention":"Crop rotation"
},

"Tomato_Late_blight":{
"fertilizer":"Potassium fertilizer",
"remedy":"Spray mancozeb fungicide",
"prevention":"Avoid excess moisture"
},

"Tomato_Leaf_Mold":{
"fertilizer":"Potassium fertilizer",
"remedy":"Apply copper fungicide",
"prevention":"Improve airflow"
},

"Tomato_Septoria_leaf_spot":{
"fertilizer":"Balanced fertilizer",
"remedy":"Spray chlorothalonil",
"prevention":"Remove infected leaves"
},

"Tomato_Spider_mites_Two_spotted_spider_mite":{
"fertilizer":"Organic compost",
"remedy":"Spray neem oil",
"prevention":"Monitor plants regularly"
},

"Tomato__Target_Spot":{
"fertilizer":"Balanced fertilizer",
"remedy":"Apply fungicide",
"prevention":"Improve air circulation"
},

"Tomato__Tomato_YellowLeaf__Curl_Virus":{
"fertilizer":"Potassium fertilizer",
"remedy":"Control whiteflies using neem oil",
"prevention":"Use insect nets"
},

"Tomato__Tomato_mosaic_virus":{
"fertilizer":"Balanced fertilizer",
"remedy":"Remove infected plants",
"prevention":"Disinfect tools"
},

"Tomato_healthy":{
"fertilizer":"Organic compost",
"remedy":"No disease detected",
"prevention":"Maintain proper watering"
}

}

uploaded = st.file_uploader("Upload a leaf image")

if uploaded:

    img = Image.open(uploaded).resize((224,224))
    st.image(img, caption="Uploaded Leaf", width=300)

    img = np.array(img)/255.0
    img = np.expand_dims(img.astype(np.float32),0)

    interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    with open("data/processed/classification_tfds/labels.txt") as f:
        labels = [line.strip() for line in f]

    prediction = labels[np.argmax(output)]

    st.success("Disease Detected: " + prediction)

    if prediction in remedies:
        r = remedies[prediction]

        st.subheader("Recommended Fertilizer")
        st.write(r["fertilizer"])

        st.subheader("Remedy")
        st.write(r["remedy"])

        st.subheader("Prevention")
        st.write(r["prevention"])