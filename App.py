import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import io
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="ECG Classifier",
    page_icon="🫀",
    layout="wide"
)

# --- Model Definition: ResNet + LSTM ---
class ResNetLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_classes=3):
        super(ResNetLSTM, self).__init__()

        # Load pre-trained ResNet but modify for grayscale input
        self.resnet = models.resnet18()
        # Change first layer to accept grayscale images (1 channel)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Add LSTM for temporal analysis
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        # Get features from ResNet
        x = self.resnet(x)  # Shape: [batch_size, 512, 7, 7]

        # Reshape for LSTM: treat 7x7 spatial features as a sequence
        x = x.view(batch_size, 49, 512)  # Shape: [batch_size, 49, 512]

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # Shape: [batch_size, 49, hidden_size]

        # Use final hidden state for classification
        out = self.fc(lstm_out[:, -1, :])  # Shape: [batch_size, num_classes]
        return out

# --- Spectrogram Processing Functions ---
def preprocess_image(image):
    """Preprocess ECG image to remove grid lines and noise"""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img = np.array(image.convert('L'))
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError("Image must be PIL Image or numpy array")
    
    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, 0, -1)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, 0, -1)

    return binary

def extract_waveform(binary_img):
    """Extract 1D signal from binary image"""
    height, width = binary_img.shape
    signal = []
    for x in range(width):
        column = binary_img[:, x]
        if np.any(column):
            y = np.argmax(column)
            signal.append(height - y)  # Invert Y to get proper waveform orientation
        else:
            signal.append(signal[-1] if signal else 0)  # Continue last value if no signal
    return np.array(signal)

def get_spectrogram(signal, fs=500):
    """Convert 1D signal to spectrogram image"""
    f, t, Sxx = spectrogram(signal, fs)
    Sxx_log = 10 * np.log10(Sxx + 1e-8)  # Convert to dB with small offset to avoid log(0)
    Sxx_log = np.clip(Sxx_log, a_min=Sxx_log.min(), a_max=Sxx_log.max())
    spectro_img = cv2.normalize(Sxx_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return spectro_img

def process_ecg_image(image):
    """Process ECG image through preprocessing pipeline"""
    # Convert to grayscale if needed
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
        img_array = np.array(image)
    else:
        img_array = image
    
    # Preprocess to remove grid lines
    binary = preprocess_image(img_array)
    
    # Extract 1D waveform
    waveform = extract_waveform(binary)
    
    # Convert to spectrogram
    spectro_img = get_spectrogram(waveform)
    
    return binary, waveform, spectro_img

# --- Streamlit App ---
def main():
    st.title("🫀 ECG Image Classifier")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload ECG Image")
        uploaded_file = st.file_uploader("Choose an ECG image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original ECG Image", use_column_width=True)
            
            # Process image through pipeline
            with st.spinner("Processing image..."):
                binary, waveform, spectro_img = process_ecg_image(image)
            
            # Display processed versions
            st.subheader("Processed Images")
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Binary Image", "Waveform", "Spectrogram"])
            
            with tab1:
                st.image(binary, caption="Binary Image (Grid Removed)", use_column_width=True)
            
            with tab2:
                # Plot waveform
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(waveform)
                ax.set_title("Extracted ECG Waveform")
                ax.set_xlabel("Time (samples)")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)
            
            with tab3:
                st.image(spectro_img, caption="Spectrogram Image", use_column_width=True)
    
    with col2:
        st.header("Prediction")
        
        # Class names
        class_names = ['Abnormal', 'Heart Attack', 'Normal']
        
        # Only show prediction section if an image is uploaded
        if 'spectro_img' in locals():
            # Prepare image for model
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            # Convert to tensor
            input_tensor = transform(spectro_img).unsqueeze(0)
            
            # Get device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model if button clicked
            if st.button("Predict"):
                with st.spinner("Loading model and making prediction..."):
                    try:
                        # Initialize model
                        model = ResNetLSTM(num_classes=len(class_names))
                        
                        # Load weights
                        model.load_state_dict(torch.load("best_ecg_model.pth", map_location=device))
                        model.to(device)
                        model.eval()
                        
                        # Make prediction
                        input_tensor = input_tensor.to(device)
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                            pred_idx = torch.argmax(probs).item()
                            
                        # Display prediction result
                        st.success(f"## Prediction: {class_names[pred_idx]}")
                        
                        # Display confidence for each class
                        st.subheader("Confidence Scores")
                        for i, (cls, prob) in enumerate(zip(class_names, probs.cpu().numpy())):
                            st.progress(float(prob))
                            st.write(f"{cls}: {prob*100:.2f}%")
                            
                    except Exception as e:
                        st.error(f"Error in prediction: {e}")
                        st.info("Make sure you have placed the model file 'best_ecg_model.pth' in the same directory as this app.")
        else:
            st.info("Upload an ECG image to get a prediction.")
    
    # Add information section
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This application uses a ResNet-LSTM neural network to classify ECG images into three categories:
        - **Normal**: Normal heartbeat pattern
        - **Abnormal**: Irregular heartbeat pattern
        - **Heart Attack**: Myocardial infarction pattern
        
        The model was trained on ECG images and processes them through a pipeline that:
        1. Removes grid lines and noise
        2. Extracts the waveform signal
        3. Converts the signal to a spectrogram
        4. Passes the spectrogram through the neural network
        """
    )
    
    # Technical details
    with st.sidebar.expander("Technical Details"):
        st.write("""
        - **Model Architecture**: ResNet18 + LSTM
        - **Input Size**: 224x224 grayscale image
        - **Classes**: 3 (Normal, Abnormal, Heart Attack)
        - **Preprocessing**: Grid removal, signal extraction, spectrogram generation
        """)

if __name__ == "__main__":
    main()