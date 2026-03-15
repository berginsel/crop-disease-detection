import tensorflow as tf

model_path = "ml/training/classification/outputs/checkpoints/final.keras"

model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("crop_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created: crop_disease_model.tflite")