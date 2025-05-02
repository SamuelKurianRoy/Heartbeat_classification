import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# --- Model Definition: ResNet + LSTM ---
class ResNetLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_classes=3):
        super(ResNetLSTM, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.resnet(x)
        x = x.view(batch_size, 49, 512)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# --- Preprocessing Functions (from training script) ---
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, 0, -1)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, 0, -1)
    return binary

# --- Streamlit App ---
def main():
    st.title("ECG Image Classifier")

    # File uploader
    uploaded_file = st.file_uploader("Upload ECG image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file).convert('L')
        img_array = np.array(image)

        # Preprocess image
        processed_img = preprocess_image(img_array)
        st.image(processed_img, caption="Preprocessed Image", use_column_width=True)

        # Prepare for model
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        input_tensor = transform(Image.fromarray(processed_img)).unsqueeze(0)

        # Load model
        model = ResNetLSTM(num_classes=3)
        model.load_state_dict(torch.load("best_ecg_model.pth", map_location=torch.device('cpu')))
        model.eval()

        # Prediction
        if st.button("Predict"):
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                prediction = torch.argmax(probabilities).item()

            class_names = ['Heart Attack', 'Normal', 'Abnormal']
            st.write(f"Prediction: {class_names[prediction]}")
            st.write("Probabilities:")
            for i, class_name in enumerate(class_names):
                st.write(f"- {class_name}: {probabilities[i]:.4f}")

if __name__ == "__main__":
    main()
