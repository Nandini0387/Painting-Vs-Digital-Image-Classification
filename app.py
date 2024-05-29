from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import io
import base64
from skimage.feature import hog
from torchvision.models import EfficientNet_B4_Weights  # Use B4 instead of B7

# Define Flask app
app = Flask(__name__)

# Define CombinedEfficientNet with the correct feature size
class CombinedEfficientNet(nn.Module):
    def __init__(self, base_model, additional_features_size):
        super(CombinedEfficientNet, self).__init__()
        self.base_model = base_model
        self.additional_features_size = additional_features_size

        # Stream for additional features (e.g., texture)
        self.additional_stream = nn.Sequential(
            nn.Linear(additional_features_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Combined classifier for final classification
        in_features = base_model.classifier[1].in_features
        self.combined_classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features + 128, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 2),  # Adjust based on your final output classes
        )

    def forward(self, images, additional_features):
        base_output = self.base_model.features(images).mean(dim=[2, 3])
        additional_output = self.additional_stream(additional_features)
        combined_output = torch.cat((base_output, additional_output), dim=1)
        output = self.combined_classifier(combined_output)
        return output

# Define augmentations
augment_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Standardize size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
])

# Function to extract texture features
def extract_texture(image):
    image_np = np.array(image)
    features, _ = hog(image_np, pixels_per_cell=(8, 8), cells_per_block=(1, 1), channel_axis=2, visualize=True)
    return features

# Load the complete model from the saved path
model_path = os.path.join('model', 'B4Model_saved.pth')  # Adjust the path to your saved model
base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)  # Change to EfficientNet B4
expected_additional_feature_size = 7056  # Adjust to your expected feature size
model = CombinedEfficientNet(base_model, expected_additional_feature_size)  # Create the correct instance
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()  # Set to evaluation mode

# Function to classify a single image
def classify_image(model, image):
    def ensure_png_and_resize(image):
        # Convert to RGB if needed
        img = image.convert("RGB")
        # Resize the image
        img = img.resize((224, 224), Image.LANCZOS)
        return img

    resized_image = ensure_png_and_resize(image)
    transformed_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
    ])(resized_image).unsqueeze(0)  # Add batch dimension

    # Extract texture features and add a batch dimension
    texture_features = torch.tensor(extract_texture(resized_image), dtype=torch.float32).unsqueeze(0)

    # Get the prediction from the model
    with torch.no_grad():
        outputs = model(transformed_image, texture_features)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), resized_image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    error = None
    if request.method == 'POST':
        if 'file' not in request.files:
            error = "Please upload the image!"
            return render_template('upload.html', error=error)
        file = request.files['file']
        if file.filename == '':
            error = "Please choose the Image!"
            return render_template('upload.html', error=error)
        if file:
            image = Image.open(file.stream)
            predicted_class, confidence, resized_image = classify_image(model, image)
            class_labels = {0: "digital-image", 1: "painting"}  # Adjust based on your class definitions
            result = {
                "predicted_class": class_labels[predicted_class],
                "confidence": f"{confidence * 100:.2f}%",
            }
            # Convert resized image to base64 string
            buffered = io.BytesIO()
            resized_image.save(buffered, format="PNG")
            img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return render_template('result.html', result=result, img_data=img_data)
    return render_template('upload.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)
