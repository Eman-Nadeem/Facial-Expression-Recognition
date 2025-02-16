# import streamlit as st
# from tensorflow.keras.models import load_model
# from src.utils import predict_emotion

# model = load_model('models/emotion_recognition_model.h5')
# class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# st.title('Emotion Recognition from Facial Images')
# uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
#     predicted_emotion = predict_emotion(uploaded_file, model, class_names)
#     st.write(f"Predicted Emotion: {predicted_emotion}")

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Load the model
model = load_model('models/emotion_recognition_model.h5')
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title('Emotion Recognition from Facial Images')

# File uploader for emotion prediction
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    from src.utils import predict_emotion
    predicted_emotion = predict_emotion(uploaded_file, model, class_names)
    st.write(f"Predicted Emotion: {predicted_emotion}")

# Function for accuracy/loss plots
def plot_training_history(history_file):
    import pickle
    with open(history_file, 'rb') as f:
        history = pickle.load(f)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot Accuracy
    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, acc, 'b', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot Loss
    fig2, ax2 = plt.subplots()
    ax2.plot(epochs, loss, 'b', label='Training Loss')
    ax2.plot(epochs, val_loss, 'r', label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    return fig1, fig2

# Button for model evaluation
if st.button("Evaluate Model"):
    st.subheader("Model Evaluation Metrics")

    # Test data directory and configuration
    test_dir = "data/test"
    batch_size = 64
    target_size = (48, 48)

    # Prepare the test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=False
    )

    # Predict the test data
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # Classification Report
    st.write("Classification Report:")
    classification_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    st.table(pd.DataFrame(classification_rep).transpose())

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot Confusion Matrix as a Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual Labels")
    ax.set_xlabel("Predicted Labels")
    st.pyplot(fig)

    # Normalized Confusion Matrix
    norm_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(norm_conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Normalized Confusion Matrix")
    ax.set_ylabel("Actual Labels")
    ax.set_xlabel("Predicted Labels")
    st.pyplot(fig)

    # Bar chart for Precision, Recall, F1-Score
    metrics_df = pd.DataFrame(classification_rep).transpose().iloc[:-3, :]
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax)
    ax.set_title('Precision, Recall, F1-Score per Class')
    ax.set_xticklabels(class_names, rotation=45)
    st.pyplot(fig)
