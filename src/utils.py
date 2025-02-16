from tensorflow.keras.preprocessing import image
import numpy as np

def predict_emotion(img_path, model, class_names):
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    return class_names[class_idx]
