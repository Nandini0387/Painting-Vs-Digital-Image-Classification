# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import EfficientNet_B4_Weights  # Use B4 instead of B7
from torchvision.models import efficientnet_b4
from skimage.feature import hog
from PIL import Image
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  

# Step 1: Define Augmentation and Transformations
augment_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Standardize size
    transforms.RandomHorizontalFlip(),  # Horizontal flip
    transforms.RandomVerticalFlip(),  # Vertical flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
])

# Step 2: Function to extract texture features from an image
def extract_texture(image):
    image_np = np.array(image)
    features, _ = hog(image_np, pixels_per_cell=(8, 8), cells_per_block=(1, 1), channel_axis=2, visualize=True)
    return features

# Step 3: Define function to process data in batches
def process_data_in_batches(dataset, batch_size=64):
    texture_features = []
    for i in range(0, len(dataset), batch_size):
        batch_images = [dataset[j][0] for j in range(i, min(i + batch_size, len(dataset)))]
        batch_texture_features = [extract_texture(
            Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        ) for image in batch_images]
        texture_features.extend(batch_texture_features)
    return texture_features

# Step 4: Load the training dataset in batches and extract texture features
train_path = 'dataset/train'  # Adjust path as needed
train_dataset = datasets.ImageFolder(root=train_path, transform=augment_transforms)

# Process the train dataset in batches to extract texture features
train_texture_features = process_data_in_batches(train_dataset, batch_size=32)


# Custom EfficientNet model with concatenation of image features and additional features
class CombinedEfficientNet(nn.Module):
    def __init__(self, base_model, additional_features_size):
        super(CombinedEfficientNet, self).__init__()
        self.base_model = base_model
        self.additional_features_size = additional_features_size

        # Linear layer for additional features (e.g., texture)
        self.additional_stream = nn.Sequential(
            nn.Linear(additional_features_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Combined classifier for final classification
        in_features = self.base_model.classifier[1].in_features
        self.combined_classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features + 128, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, len(train_dataset.classes)),  # Final output layer for classification
        )

    def forward(self, images, additional_features):
        base_output = self.base_model.features(images).mean(dim=[2, 3])
        additional_output = self.additional_stream(additional_features)
        combined_output = torch.cat((base_output, additional_output), dim=1)
        output = self.combined_classifier(combined_output)
        return output

# Instantiate EfficientNet B4 pretrained model
base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

# Initialize CombinedEfficientNet with additional texture features
additional_features_size = len(train_texture_features[0])
model = CombinedEfficientNet(base_model, additional_features_size)

class CombinedDataset(data.Dataset):
    def __init__(self, dataset, texture_features):
        self.dataset = dataset  # Assign the dataset
        self.texture_features = texture_features  # Correctly assign texture_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        texture = torch.tensor(self.texture_features[idx], dtype=torch.float32)
        return (image, texture), label


# Create a DataLoader with the augmented dataset and extracted texture features
train_loader = data.DataLoader(
    CombinedDataset(train_dataset, train_texture_features),
    batch_size=32,
    shuffle=True
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define a function to train the model with additional metrics like Precision, Recall, F1-Score
def train_model_with_metrics(model, loader, criterion, optimizer, num_epochs, num_classes):
    model.train()  # Set to training mode
    training_losses = []
    training_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        all_labels = []
        all_predictions = []

        for (images, textures), labels in loader:
            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            outputs = model(images, textures)
            loss = criterion(outputs, labels)  # Calculate loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get predictions
            correct_predictions += (predicted == labels).sum().item()  # Correct predictions
            total_samples += labels.size(0)  # Number of samples in this batch

            all_labels.extend(labels.cpu().numpy())  # Collect all labels
            all_predictions.extend(predicted.cpu().numpy())  # Collect all predictions

        # Track the average loss and accuracy for the epoch
        epoch_loss = running_loss / len(loader)
        epoch_accuracy = correct_predictions / total_samples

        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%")

        # Calculate precision, recall, and F1-score
        report = classification_report(all_labels, all_predictions, target_names=loader.dataset.dataset.classes, output_dict=False)
        print("Classification Report:\n", report)

        # Display confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print("Confusion Matrix:\n", cm)

    return training_losses, training_accuracies

# Number of epochs
num_epochs = 10
num_classes = len(train_dataset.classes)

# Train the model with additional metrics and get training losses and accuracies
train_losses, train_accuracies = train_model_with_metrics(model, train_loader, criterion, optimizer, num_epochs, num_classes)

# Plot the training loss over epochs
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

# Plot the training accuracy over epochs
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.legend()
plt.show()

# Step 10: Save the complete model for future use
model_path = "/kaggle/working/B4Model_trained"  # Adjust the path
torch.save(model, model_path)  # Save the complete model
print("Complete model saved to:", model_path)



# Validation dataset path
validation_path = 'dataset/validation'  # Adjust as needed
validation_dataset = datasets.ImageFolder(validation_path, transform=augment_transforms)

# Extract texture features for the validation dataset
validation_texture_features = []
batch_size = 32  # Adjusted batch size to avoid memory overload

# Process data in batches to extract texture features for validation dataset
for i in range(0, len(validation_dataset), batch_size):
    batch_images = [validation_dataset[j][0] for j in range(i, min(i + batch_size, len(validation_dataset)))]
    batch_texture_features = [extract_texture(
      Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    ) for image in batch_images]  # Corrected syntax
    validation_texture_features.extend(batch_texture_features)

# Create a DataLoader for the validation dataset
validation_loader = data.DataLoader(
    CombinedDataset(validation_dataset, validation_texture_features),
    batch_size=32,  # Adjust as needed
    shuffle=True  # No need to shuffle validation data
)

# Define a function to evaluate the model
def evaluate_model(model, loader):
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed during evaluation
        for (images, textures), labels in loader:
            outputs = model(images, textures)  # Forward pass with both images and textures
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
            total += labels.size(0)  # Total samples
            correct += (predicted == labels).sum().item()  # Correct predictions

    accuracy = correct / total  # Calculate accuracy
    return accuracy

# Validate the model with the validation dataset
val_accuracy = evaluate_model(model, validation_loader)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")


# Test dataset path
test_path = 'dataset/test'  # Adjust as needed
batch_size = 32  # Batch size for processing

# Create test dataset and extract texture features
test_dataset = datasets.ImageFolder(test_path, transform=augment_transforms)
test_texture_features = []

# Process the test dataset in batches to avoid memory overload
for i in range(0, len(test_dataset), batch_size):
    batch_images = [test_dataset[j][0] for j in range(i, min(i + batch_size, len(test_dataset)))]
    batch_texture_features = [
        extract_texture(
            Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        ) for image in batch_images
    ]
    test_texture_features.extend(batch_texture_features)

# Create a DataLoader for the test dataset
test_loader = data.DataLoader(
    CombinedDataset(test_dataset, test_texture_features),
    batch_size=32,
    shuffle=True  # No need to shuffle test data
)

# Function to evaluate the model with test dataset
def evaluate_model(model, loader):
    true_labels = []
    pred_labels = []

    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # No gradients needed during evaluation
        for (images, textures), labels in loader:
            outputs = model(images, textures)  # Forward pass with combined data
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
            true_labels.extend(labels.tolist())
            pred_labels.extend(predicted.tolist())

    # Calculate various metrics
    accuracy = accuracy_score(true_labels, pred_labels)  # Corrected NameError
    class_report = classification_report(
        true_labels, pred_labels, target_names=test_dataset.classes, output_dict=True
    )
    precision = class_report["weighted avg"]["precision"]
    recall = class_report["weighted avg"]["recall"]
    f1_score = class_report["weighted avg"]["f1-score"]

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1_score * 100:.2f}%")

    return true_labels, pred_labels

# Evaluate the model with the test dataset
true_labels, pred_labels = evaluate_model(model, test_loader)

# Function to plot the confusion matrix for the test dataset
def plot_confusion_matrix(true_labels, pred_labels, class_names):
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Plot the confusion matrix with the test dataset
plot_confusion_matrix(true_labels, pred_labels, test_dataset.classes)


# Step 10: Save the complete model for future use
model_path = "model/B4Model_saved"  # Adjust the path
torch.save(model, model_path)  # Save the complete model
print("Complete model saved to:", model_path)