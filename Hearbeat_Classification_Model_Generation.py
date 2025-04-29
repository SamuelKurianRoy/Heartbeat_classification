import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import spectrogram
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ---------- Spectrogram Preprocessing Functions ----------
def preprocess_image(image):
    """Preprocess ECG image to remove grid lines and noise"""
    if isinstance(image, str):  # If path is provided instead of image
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = image  # Use the image directly

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


# ---------- Load and Prepare Data ----------
def load_images_from_folder(folder, label):
    """Load all images from a folder with their labels"""
    images = []
    labels = []

    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist")
        return images, labels

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(label)

    print(f"Loaded {len(images)} images from {folder}")
    return images, labels


# Paths to folders
heart_attack_folder = 'ECG Images of Myocardial Infarction Patients (240x12=2880)'
normal_heartbeat_folder = 'Normal Person ECG Images (284x12=3408)'
abnormal_heartbeat_folder = 'ECG Images of Patient that have abnormal heartbeat (233x12=2796)'

# Load images with labels
heart_attack_images, heart_attack_labels = load_images_from_folder(heart_attack_folder, 'Heart Attack')
normal_heartbeat_images, normal_heartbeat_labels = load_images_from_folder(normal_heartbeat_folder, 'Normal')
abnormal_heartbeat_images, abnormal_heartbeat_labels = load_images_from_folder(abnormal_heartbeat_folder, 'Abnormal')

# Combine datasets
all_images = heart_attack_images + normal_heartbeat_images + abnormal_heartbeat_images
all_labels = heart_attack_labels + normal_heartbeat_labels + abnormal_heartbeat_labels

# Create DataFrame
df = pd.DataFrame({'image': all_images, 'label': all_labels})
print(f"Total dataset size: {len(df)} images")

# Display dataset distribution
print(df['label'].value_counts())

# ---------- PyTorch Data Pipeline ----------
# Image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale
])

# Label encoding
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])
print(f"Classes: {label_encoder.classes_}")

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['encoded_label'], random_state=42)
print(f"Training set: {len(train_df)}, Test set: {len(test_df)}")


class ECGDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image and label
        img = self.dataframe.loc[idx, 'image']
        label = self.dataframe.loc[idx, 'encoded_label']

        # Process image
        binary = preprocess_image(img)
        waveform = extract_waveform(binary)
        spectro_img = get_spectrogram(waveform)

        # Apply transformations
        if self.transform:
            spectro_img = self.transform(spectro_img)

        return spectro_img, label


# Create datasets and dataloaders
train_dataset = ECGDataset(train_df, transform=transform)
test_dataset = ECGDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ---------- Model Definition ----------
class ResNetLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_classes=3):
        super(ResNetLSTM, self).__init__()

        # Load pre-trained ResNet but modify for grayscale input
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")
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


# ---------- Training and Evaluation Functions ----------
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    return accuracy, all_preds, all_labels


# ---------- Main Training Loop ----------
# Initialize model
model = ResNetLSTM(num_classes=len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# Training parameters
num_epochs = 10
best_acc = 0.0

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)

    # Evaluate model
    test_acc, _, _ = evaluate(model, test_loader)

    # Scheduler step
    scheduler.step(test_acc)

    # Print statistics
    print(f"Epoch {epoch + 1}/{num_epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Test Acc: {test_acc:.2f}%")

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_ecg_model.pth')
        print(f"Saved new best model with accuracy: {best_acc:.2f}%")

# Load best model for final evaluation
model.load_state_dict(torch.load('best_ecg_model.pth'))
test_acc, preds, true_labels = evaluate(model, test_loader)
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")


# Display example predictions
def show_sample_predictions(model, dataset, num_samples=5):
    model.eval()
    fig, axs = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        # Get a random sample
        idx = np.random.randint(0, len(dataset))
        img, label = dataset[idx]

        # Get prediction
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            _, pred = output.max(1)
            pred = pred.item()

        # Get original image (before transform)
        orig_img = dataset.dataframe.loc[idx, 'image']

        # Get class names
        true_class = label_encoder.inverse_transform([label])[0]
        pred_class = label_encoder.inverse_transform([pred])[0]

        # Display
        axs[i].imshow(orig_img, cmap='gray')
        axs[i].set_title(f"True: {true_class}, Predicted: {pred_class}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

# Uncomment to show sample predictions
# show_sample_predictions(model, test_dataset)