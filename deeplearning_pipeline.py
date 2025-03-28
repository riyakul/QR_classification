#!/usr/bin/env python
"""
Production-Level Original vs Counterfeit Classifier with a Tkinter GUI
Trains a ResNet18 (transfer learning) on a user-selected dataset,
provides evaluation, and tests single images or entire folders.
Now includes the load_datasets and load_test_dataset function definitions
so that Python recognizes them in the same file.
"""

import os
import threading
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# -------------------------------
# Data & Model Helper Functions
# -------------------------------

def get_transforms(img_height=224, img_width=224):
    """
    Defines separate transforms for training (with augmentation) and validation/test (minimal).
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop((img_height, img_width), scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    return train_transforms, val_transforms

def load_datasets(data_dir, img_height=224, img_width=224, batch_size=16, val_split=0.2):
    """
    Loads an ImageFolder dataset from data_dir, splits into train/validation,
    and returns DataLoaders plus the class names.
    """
    train_transforms, val_transforms = get_transforms(img_height, img_width)
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    
    # Split into train/validation
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply different (non-augmented) transforms to validation set
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.classes

def load_test_dataset(test_dir, img_height=224, img_width=224, batch_size=16):
    """
    Loads an ImageFolder dataset from test_dir (no train/val split) for evaluation or 'Test Folder' usage.
    """
    _, test_transforms = get_transforms(img_height, img_width)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dataset.classes

def build_model(num_classes=2):
    """
    Builds a pre-trained ResNet18, freezing early layers, and replaces the final FC layer with num_classes.
    """
    model = models.resnet18(pretrained=True)
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, log_func=print):
    """
    Trains the model using CrossEntropyLoss. Logs progress via log_func.
    Only the final layer (model.fc) is trained initially.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    train_acc_history, val_acc_history = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        
        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_acc_history.append(val_acc)
        
        log_func(f"Epoch {epoch+1}/{epochs} - "
                 f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} - "
                 f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_acc_history,
        "val_acc": val_acc_history
    }
    return history

def evaluate_model(model, data_loader, device, class_names):
    """
    Evaluates the model on data_loader, prints and returns classification metrics.
    Also displays a confusion matrix (both as a plot and in textual form).
    """
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a textual representation of the confusion matrix
    cm_text = "\nConfusion Matrix (Rows=True, Cols=Predicted):\n"
    for row in cm:
        cm_text += " ".join(map(str, row)) + "\n"
    
    # Plot confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    
    results = (f"Accuracy: {acc*100:.2f}%\n"
               f"Precision: {prec:.2f}\n"
               f"Recall: {rec:.2f}\n"
               f"F1 Score: {f1:.2f}\n\n"
               f"Classification Report:\n{report}"
               f"{cm_text}")
    return results

def test_single_image(model, img_path, device, img_height=224, img_width=224, class_names=None):
    """
    Classifies a single image, returning the predicted class (Original/Counterfeit) and confidence score.
    """
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    if not os.path.exists(img_path):
        return "Image file not found."
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_index = np.argmax(prob)
        pred_class = class_names[pred_index] if class_names else str(pred_index)
        confidence = prob[pred_index]
    
    return f"Predicted Class: {pred_class}\nConfidence: {confidence:.2f}"

# -------------------------------
# Tkinter GUI
# -------------------------------

class ProductionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Original vs Counterfeit Classifier")
        self.dataset_dir = None
        self.model = None
        self.class_names = None
        self.history = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # GUI Elements
        self.select_btn = tk.Button(root, text="Select Training Dataset", command=self.select_dataset)
        self.select_btn.pack(pady=5)
        
        self.train_btn = tk.Button(root, text="Train Model", command=self.start_training, state=tk.DISABLED)
        self.train_btn.pack(pady=5)
        
        self.eval_btn = tk.Button(root, text="Evaluate on Validation", command=self.evaluate_validation, state=tk.DISABLED)
        self.eval_btn.pack(pady=5)
        
        self.test_btn = tk.Button(root, text="Test Single Image", command=self.test_image, state=tk.DISABLED)
        self.test_btn.pack(pady=5)
        
        self.test_folder_btn = tk.Button(root, text="Test Entire Folder", command=self.test_folder, state=tk.DISABLED)
        self.test_folder_btn.pack(pady=5)
        
        self.log_area = scrolledtext.ScrolledText(root, width=80, height=20)
        self.log_area.pack(pady=10)
    
    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
    
    def select_dataset(self):
        self.dataset_dir = filedialog.askdirectory(title="Select Dataset Directory")
        if self.dataset_dir:
            self.log(f"Dataset directory selected: {self.dataset_dir}")
            self.train_btn.config(state=tk.NORMAL)
        else:
            self.log("No dataset directory selected.")
    
    def start_training(self):
        if not self.dataset_dir:
            messagebox.showerror("Error", "Please select a dataset directory first.")
            return
        self.log("Starting training...")
        self.train_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.train_thread)
        thread.start()
    
    def train_thread(self):
        try:
            # Load train/val
            train_loader, val_loader, self.class_names = load_datasets(self.dataset_dir)
            # Build model
            self.model = build_model(num_classes=len(self.class_names)).to(self.device)
            # Train
            self.history = train_model(self.model, train_loader, val_loader, self.device,
                                       epochs=10, lr=0.001, log_func=self.log)
            self.log("Training completed.")
            self.eval_btn.config(state=tk.NORMAL)
            self.test_btn.config(state=tk.NORMAL)
            self.test_folder_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.log(f"Error during training: {str(e)}")
    
    def evaluate_validation(self):
        if self.model is None:
            messagebox.showerror("Error", "Model is not trained yet.")
            return
        # Reload the train/val split to get the same validation loader
        _, val_loader, _ = load_datasets(self.dataset_dir)
        results = evaluate_model(self.model, val_loader, self.device, self.class_names)
        self.log("Validation Evaluation Results:")
        self.log(results)
        messagebox.showinfo("Evaluation Results", results)
    
    def test_image(self):
        if self.model is None:
            messagebox.showerror("Error", "Model is not trained yet.")
            return
        img_path = filedialog.askopenfilename(title="Select Image",
                                              filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if img_path:
            result = test_single_image(self.model, img_path, self.device, class_names=self.class_names)
            self.log("Test Single Image Result:")
            self.log(result)
            messagebox.showinfo("Test Single Image Result", result)
        else:
            self.log("No image selected for testing.")
    
    def test_folder(self):
        """
        Lets you pick a folder with subfolders 'Original' and 'Counterfeit'
        to compute overall accuracy, precision, recall, F1, and confusion matrix.
        """
        if self.model is None:
            messagebox.showerror("Error", "Model is not trained yet.")
            return
        folder_path = filedialog.askdirectory(title="Select Folder to Test")
        if folder_path:
            self.log(f"Testing folder: {folder_path}")
            thread = threading.Thread(target=self.test_folder_thread, args=(folder_path,))
            thread.start()
        else:
            self.log("No folder selected for testing.")
    
    def test_folder_thread(self, folder_path):
        try:
            test_loader, test_classes = load_test_dataset(folder_path)
            # If the user expects exactly the same classes, we can proceed
            # Otherwise, we might mismatch. For simplicity, we assume 2 classes: Original, Counterfeit.
            if set(test_classes) != set(self.class_names):
                self.log("Warning: The test folder's classes differ from the training classes.")
            
            results = evaluate_model(self.model, test_loader, self.device, test_classes)
            self.log("Test Folder Results:")
            self.log(results)
            messagebox.showinfo("Test Folder Results", results)
        except Exception as e:
            self.log(f"Error during test folder evaluation: {str(e)}")

# -------------------------------
# Main
# -------------------------------

if __name__ == '__main__':
    # Use interactive mode for matplotlib so it works with Tkinter
    plt.ion()
    root = tk.Tk()
    app = ProductionApp(root)
    root.mainloop()
