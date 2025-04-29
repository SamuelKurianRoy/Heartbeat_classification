import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Model Definition: ResNet + LSTM ---
class ResNetLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_classes=3):
        super(ResNetLSTM, self).__init__()

        # Load pre-trained ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # Modify the first layer to accept grayscale images
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove ResNet's fully connected layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # output shape: (B, 512, 7, 7)

        # LSTM on flattened spatial features
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.resnet(x)  # (B, 512, 7, 7)
        x = x.view(x.size(0), 49, 512)  # reshape for LSTM: (B, T=49, 512)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # take the last time step
        x = self.fc(x)
        return x

# --- Load Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize and load weights
model = ResNetLSTM()
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.to(device)
model.eval()

# --- Define class labels ---
class_names = ['Abnormal', 'Heart Attack', 'Normal']

# --- Streamlit UI ---
st.title("🫀 ECG Image Classifier")

uploaded_file = st.file_uploader("Upload a grayscale ECG image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded ECG Image", use_column_width=True)

    # --- Image Preprocessing ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale image
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # --- Prediction ---
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()

    # --- Display result ---
    st.success(f"🧠 Predicted Class: **{prediction}**")
    st.info(f"🔎 Confidence: {confidence:.2%}")
