# Start of Selection
# ------------------------------------------------------------------------
# Example Training Script for Lung Cancer CT Scan Classification
# This script demonstrates how to train a 2D CNN model using the preprocessed data.
# ------------------------------------------------------------------------

"""
Example Training Script for Lung Cancer CT Scan Classification
This script demonstrates how to train a 2D CNN model using the preprocessed data.
"""

# -------------------- Import Required Libraries --------------------

import torch                                           # Import the main PyTorch library for deep learning
import torch.nn as nn                                  # Import PyTorch's neural network subpackage (layers, modules, etc.)
import torch.optim as optim                            # Import PyTorch's optimization package (optimizers like Adam)
from torch.utils.data import DataLoader                # Import DataLoader utility for batching/shuffling/loading datasets
from data_loader import create_data_loaders            # Custom function to create data loaders for this dataset/project
import numpy as np                                     # Import NumPy library for numerical operations and array handling
from tqdm import tqdm                                  # Progress bar library to make loops visually trackable
import matplotlib.pyplot as plt                        # Matplotlib for plotting training curves (loss, accuracy, etc.)

# -------------------- Define the Neural Network Model --------------------

class SimpleLungCancerCNN(nn.Module):
    """
    Simple CNN model for lung cancer classification.
    """
    def __init__(self, num_classes=5):
        # Call the parent constructor to initialize the nn.Module class
        super(SimpleLungCancerCNN, self).__init__()
        
        # Define a stack of convolutional and pooling layers for feature extraction.
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),      # 2D convolution: input 3 channels, output 64 channels, 3x3 kernel, 1-pixel padding
            nn.BatchNorm2d(64),                              # Batch normalization for 64 feature maps
            nn.ReLU(inplace=True),                           # In-place ReLU non-linearity
            nn.MaxPool2d(2, 2),                              # 2x2 max pooling reduces width and height by half
            nn.Dropout2d(0.25),                              # Drop out 25% of features randomly during training
            
            # Second convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # Input: 64 feature maps, Output: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),   # Input: 128 feature maps, Output: 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Fourth convolutional block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),   # Input: 256 feature maps, Output: 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Classifier "head": Reduce spatial features into class logits for classification.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),                   # Pool any size to a fixed (7,7) output for further processing
            nn.Flatten(),                                   # Flatten to 1D for input to Linear layer
            nn.Linear(512 * 7 * 7, 512),                    # Fully connected layer: all features to 512 units
            nn.BatchNorm1d(512),                            # Normalize activations across batch
            nn.ReLU(inplace=True),                          # Activation function
            nn.Dropout(0.5),                                # Drop out half the neurons' outputs randomly at train time
            nn.Linear(512, 256),                            # Second fully connected layer: 512 to 256 units
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                                # Drop out half again
            nn.Linear(256, num_classes)                     # Final fully connected: units → number of classes (logits)
        )
    
    def forward(self, x):
        # Defines the forward pass for input x through entire network
        x = self.features(x)   # Feature extraction by conv layers and poolings
        x = self.classifier(x) # Flatten and classify features
        return x

# -------------------- Single Epoch Training Function --------------------

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()                          # Set model into training mode (enables dropout, updates BN stats)
    running_loss = 0.0                     # Initialize variable to accumulate total epoch loss
    correct = 0                            # Tracks number of correct predictions
    total = 0                              # Tracks total samples processed

    # Iterate over every batch in the training data loader
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)   # Move input data and labels to CPU or GPU as needed
        optimizer.zero_grad()                                   # Zero all previous gradients before this step

        outputs = model(images)                                 # Forward pass: run images through model to get predictions
        loss = criterion(outputs, labels)                       # Compute loss value against the ground truth
        
        loss.backward()                                         # Backward pass: compute gradients with respect to loss
        optimizer.step()                                        # Update model weights/parameters using optimizer
        
        running_loss += loss.item()                             # Add loss value for this batch to the total
        _, predicted = torch.max(outputs.data, 1)               # Take the class with the highest score as prediction
        total += labels.size(0)                                 # Increase total counter by batch size
        correct += (predicted == labels).sum().item()           # Increase correct counter if prediction matches ground truth
    
    epoch_loss = running_loss / len(train_loader)               # Compute average loss by dividing total by number of batches
    epoch_acc = 100 * correct / total                           # Calculate accuracy as percentage (correct / total)
    return epoch_loss, epoch_acc                                # Return average loss and accuracy

# -------------------- Validation (Evaluation) Function --------------------

def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    Returns:
        Average loss and accuracy
    """
    model.eval()                                            # Set model to evaluation mode (turns off dropout, BN uses running stats)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():                                   # Don't calculate gradients; saves memory and time
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)  # Send to device
            outputs = model(images)                               # Get model predictions
            loss = criterion(outputs, labels)                     # Compute loss
            
            running_loss += loss.item()                           # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)             # Predicted labels
            total += labels.size(0)                               # Count
            correct += (predicted == labels).sum().item()         # Count correct predictions
    
    epoch_loss = running_loss / len(val_loader)                   # Average loss over batches
    epoch_acc = 100 * correct / total                             # Compute percent accuracy
    return epoch_loss, epoch_acc

# -------------------- Full Training Pipeline --------------------

def train_model(data_dir="processed_data", num_epochs=20, batch_size=32, learning_rate=0.001):
    """
    Train the CNN model.
    Args:
        data_dir: Path to processed data directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    # ---- Setup device for computation ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Prefer GPU, fallback to CPU
    print(f"Using device: {device}")

    # ---- Create data loaders for train, val, and test sets ----
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=data_dir,         # The location of the processed dataset 
        batch_size=batch_size,     # Number of samples per batch
        num_workers=0,             # Data loading in main process/thread
        augment=True               # Enable data augmentation
    )
    print(f"Classes: {class_names}")                                 # Print class names
    print(f"Number of classes: {len(class_names)}")                  # Print number of distinct classes

    # ---- Instantiate the CNN model ----
    model = SimpleLungCancerCNN(num_classes=len(class_names))        # Use number of classes from data loader
    model = model.to(device)                                         # Move model to computation device
    # ---- Loss function, optimizer and learning rate scheduler ----
    criterion = nn.CrossEntropyLoss()                                # Use cross-entropy for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Adam optimizer with weight decay regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )                                                               # Scheduler reduces learning rate when validation loss plateaus

    # ---- Initialize dictionary to track loss/accuracy through training ----
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_val_acc = 0.0                                              # Track the best (highest) validation accuracy achieved
    best_model_state = None                                         # This will store the parameters of the best model

    print("\nStarting training...")
    print("=" * 60)
    # ---- Main training loop over epochs ----
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")                    # Print which epoch we're on (1-based for humans)
        print("-" * 60)
        # Train the model for one full epoch, collecting training loss & accuracy
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # Evaluate ("validate") the model on the validation dataset, collecting loss & accuracy
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        # Only reduce LR if the validation loss does not improve for several epochs
        scheduler.step(val_loss)

        # Save loss and accuracy results so we can plot them later
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print metrics for this epoch: summary of how well model did
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")        # Show current LR since it may change

        # If this epoch gave us a new best validation accuracy, save the model weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc                                            # Update best accuracy tracker
            best_model_state = model.state_dict().copy()                      # Save current model weights
            print(f"✓ New best validation accuracy: {best_val_acc:.2f}%")
            torch.save(best_model_state, 'best_model.pth')                    # Also save to file (for safety)

    print("\n" + "=" * 60)
    print("Training completed!")                                              # End of training loop
    print(f"Best validation accuracy: {best_val_acc:.2f}%")                  # Report the highest val acc reached

    # ---- Testing phase: Evaluate the best model on the hold-out test set ----
    if best_model_state:
        model.load_state_dict(best_model_state)                               # Restore best weights
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)     # Compute test metrics
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")          # Print test results

    # ---- Plot the loss/accuracy training history ----
    plot_training_history(history)                                            # Draw and save the curves

    return model, history                                                    # Return trained model and stats

# -------------------- Plotting Function for Training History --------------------

def plot_training_history(history):
    """
    Plot training and validation loss and accuracy.
    """
    # Create a 1-row, 2-column subplot -- one panel for loss, one for accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ---- Plot loss over epochs ----
    ax1.plot(history['train_loss'], label='Train Loss')                      # Training loss per epoch
    ax1.plot(history['val_loss'], label='Val Loss')                          # Validation loss per epoch
    ax1.set_xlabel('Epoch')                                                  # Label x-axis
    ax1.set_ylabel('Loss')                                                   # Label y-axis
    ax1.set_title('Training and Validation Loss')                            # Title for loss panel
    ax1.legend()                                                             # Add legend
    ax1.grid(True)                                                           # Add grid for easier visualization

    # ---- Plot accuracy over epochs ----
    ax2.plot(history['train_acc'], label='Train Acc')                        # Training accuracy per epoch
    ax2.plot(history['val_acc'], label='Val Acc')                            # Validation accuracy per epoch
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # ---- Adjust layout and save the figure ----
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')        # Save the plot as a static PNG file
    print("\nTraining history saved to 'training_history.png'")              # Tell user where to find figure

# -------------------- Script Entry Point --------------------

if __name__ == "__main__":
    # When run as a script (not imported as a module),
    # call the training routine using reasonable default parameters.
    model, history = train_model(
        data_dir="processed_data",           # Path to preprocessed, split dataset
        num_epochs=20,                       # Number of training epochs
        batch_size=32,                       # Mini-batch size
        learning_rate=0.001                  # Initial learning rate for Adam optimizer
    )

# End of Selection
