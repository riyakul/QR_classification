#!/usr/bin/env python
"""
Incremental Learning QR Code Classifier with a GUI
Using SGDClassifier for partial_fit so we don't have to retrain from scratch each time.
Now includes functionality to classify either a single image or an entire folder of test images.
For each classified image, the feature scores and interpretation are displayed along with:
    - The model’s predicted label (Original/Counterfeit)
    - An inferred ground truth label if available
"""

import os
import cv2
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

###########################################################################
# Feature Extraction Functions
###########################################################################

def measure_crispness(gray_image):
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return laplacian.var()

def measure_edge_density(gray_image, threshold1=100, threshold2=200):
    edges = cv2.Canny(gray_image, threshold1, threshold2)
    edge_pixels = np.sum(edges > 0)
    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    return edge_pixels / total_pixels

def measure_cdp_contrast(gray_image):
    ret, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(otsu_thresh == 0) == 0 or np.sum(otsu_thresh == 255) == 0:
        return 0.0
    mean_black = np.mean(gray_image[otsu_thresh == 0])
    mean_white = np.mean(gray_image[otsu_thresh == 255])
    return mean_white - mean_black

def measure_high_frequency_ratio(gray_image, cutoff=0.5):
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    total_energy = np.sum(magnitude_spectrum)

    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    max_distance = np.sqrt(crow**2 + ccol**2)
    threshold = cutoff * max_distance
    high_freq_mask = distance > threshold
    high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
    ratio = high_freq_energy / (total_energy + 1e-7)
    return ratio

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f1 = measure_crispness(gray)
    f2 = measure_edge_density(gray)
    f3 = measure_cdp_contrast(gray)
    f4 = measure_high_frequency_ratio(gray, cutoff=0.5)
    return np.array([f1, f2, f3, f4], dtype=np.float32)

def interpret_features(f1, f2, f3, f4):
    """
    Provide a brief interpretation of each feature.
    (Thresholds here are arbitrary—adjust them based on your data.)
    """
    interp = {}
    interp['Crispness'] = "High (sharp image)" if f1 > 100 else "Low (blurred image)"
    interp['Edge Density'] = "Sufficient edges" if f2 > 0.05 else "Few edges"
    interp['Contrast'] = "Good contrast" if f3 > 30 else "Low contrast"
    interp['High Frequency Ratio'] = "Rich in high frequencies" if f4 > 0.5 else "Low high-frequency content"
    return interp

def infer_actual_label(filename):
    """
    Infers the actual label from the filename.
    For example, if the filename contains 'original' or 'true', it returns 'Original (True)'
    and if it contains 'counterfeit' or 'fake', it returns 'Counterfeit (False)'.
    Otherwise returns "Unknown".
    """
    lower = filename.lower()
    if "original" in lower or "true" in lower:
        return "Original (True)"
    elif "counterfeit" in lower or "fake" in lower:
        return "Counterfeit (False)"
    else:
        return "Unknown"

###########################################################################
# Incremental Learning Class
###########################################################################

class QRCodeIncrementalModel:
    """
    Holds the model (SGDClassifier) and scaler (StandardScaler),
    allows partial_fit with new data, and saves/loads from disk.
    """
    def __init__(self, model_path="model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self._classes = np.array([0, 1], dtype=np.int32)  # 0=original, 1=counterfeit

        # Load or initialize
        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            print(f"[INFO] Loaded model from {self.model_path}")
        else:
            self.model = SGDClassifier(
                loss='hinge',  # SVM-style
                learning_rate='optimal',
                random_state=42
            )
            self.scaler = StandardScaler()
            X_init = np.zeros((1, 4), dtype=np.float32)
            y_init = np.array([0], dtype=np.int32)
            self.model.partial_fit(X_init, y_init, classes=self._classes)
            self.scaler.partial_fit(X_init)
            print("[INFO] Created a new model (no existing model file).")

    def partial_fit(self, X, y):
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.partial_fit(X_scaled, y, classes=self._classes)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self):
        data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(data, self.model_path)
        print(f"[INFO] Model saved to {self.model_path}")

###########################################################################
# GUI Implementation
###########################################################################

class QRCodeGUI:
    """
    A tkinter GUI that allows:
      - Selecting new data folders for updating the model.
      - Incrementally updating the model.
      - Classifying individual images.
      - Classifying a folder of test images and displaying feature scores.
      - Displaying classification metrics including a confusion matrix.
    """
    def __init__(self, root, model_path="model.pkl"):
        self.root = root
        self.root.title("Incremental QR Code Classifier")
        # Set a larger display window
        self.root.geometry("1200x800")
        self.model_path = model_path

        # Create or load the incremental model
        self.incremental_model = QRCodeIncrementalModel(model_path=self.model_path)

        self.original_folder = ""
        self.counterfeit_folder = ""
        self.single_image_path = ""
        self.test_folder = ""

        # Build UI
        self._build_ui()

    def _build_ui(self):
        # Header label
        header = tk.Label(self.root, text="Incremental QR Code Classifier", font=("Helvetica", 24, "bold"))
        header.pack(pady=10)

        # Frame for training data folder selection
        folder_frame = tk.Frame(self.root, padx=10, pady=5)
        folder_frame.pack(fill=tk.X)

        tk.Button(folder_frame, text="Select Original Folder", command=self.select_original_folder, width=25).grid(row=0, column=0, padx=5, pady=5)
        self.label_original = tk.Label(folder_frame, text="Original: Not selected", width=50, anchor="w")
        self.label_original.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(folder_frame, text="Select Counterfeit Folder", command=self.select_counterfeit_folder, width=25).grid(row=1, column=0, padx=5, pady=5)
        self.label_counterfeit = tk.Label(folder_frame, text="Counterfeit: Not selected", width=50, anchor="w")
        self.label_counterfeit.grid(row=1, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Update Model with New Data", command=self.update_model, width=30).pack(pady=10)

        # Frame for single image classification
        single_frame = tk.Frame(self.root, padx=10, pady=5)
        single_frame.pack(fill=tk.X)

        tk.Button(single_frame, text="Select Single Image", command=self.select_single_image, width=25).grid(row=0, column=0, padx=5, pady=5)
        self.label_single = tk.Label(single_frame, text="No image selected", width=50, anchor="w")
        self.label_single.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Classify Single Image", command=self.classify_single_image, width=30).pack(pady=5)

        # Frame for test folder classification
        test_folder_frame = tk.Frame(self.root, padx=10, pady=5)
        test_folder_frame.pack(fill=tk.X)

        tk.Button(test_folder_frame, text="Select Test Folder", command=self.select_test_folder, width=25).grid(row=0, column=0, padx=5, pady=5)
        self.label_test = tk.Label(test_folder_frame, text="Test Folder: Not selected", width=50, anchor="w")
        self.label_test.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Classify Test Folder", command=self.classify_test_folder, width=30).pack(pady=5)

        # Frame for metrics and report display
        metrics_frame = tk.Frame(self.root, padx=10, pady=5)
        metrics_frame.pack(fill=tk.X)

        self.text_display = scrolledtext.ScrolledText(self.root, width=100, height=20, state=tk.DISABLED)
        self.text_display.pack(pady=10)

        self.metrics_label = tk.Label(metrics_frame, text="F1 Score (Macro): Not calculated", width=50, anchor="w", font=("Helvetica", 12))
        self.metrics_label.grid(row=0, column=0, padx=5, pady=5)

    def select_original_folder(self):
        folder = filedialog.askdirectory(title="Select Original QR Code Folder")
        if folder:
            self.original_folder = folder
            self.label_original.config(text=f"Original: {folder}")

    def select_counterfeit_folder(self):
        folder = filedialog.askdirectory(title="Select Counterfeit QR Code Folder")
        if folder:
            self.counterfeit_folder = folder
            self.label_counterfeit.config(text=f"Counterfeit: {folder}")

    def select_single_image(self):
        filepath = filedialog.askopenfilename(
            title="Select QR Code Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if filepath:
            self.single_image_path = filepath
            self.label_single.config(text=os.path.basename(filepath))

    def select_test_folder(self):
        folder = filedialog.askdirectory(title="Select Test Images Folder")
        if folder:
            self.test_folder = folder
            self.label_test.config(text=f"Test Folder: {folder}")

    def update_model(self):
        """
        Incrementally train the model with new data from selected folders (original vs. counterfeit).
        """
        if not self.original_folder or not self.counterfeit_folder:
            messagebox.showerror("Error", "Please select both Original and Counterfeit folders.")
            return

        X, y = self._collect_data()
        if len(X) == 0:
            messagebox.showinfo("Info", "No valid images found. Cannot update model.")
            return

        self.incremental_model.partial_fit(X, y)
        self.incremental_model.save()

        messagebox.showinfo("Success", "Model has been updated with new data!")
        self._append_text("Model updated with new data.\n")
        self._update_metrics(X, y)

    def classify_single_image(self):
        """
        Classify one image and display feature scores, interpretation, and classification results.
        """
        if not self.single_image_path:
            messagebox.showerror("Error", "Please select an image to classify.")
            return
        try:
            feats = extract_features(self.single_image_path).reshape(1, -1)
            f1, f2, f3, f4 = feats.flatten()
            pred = self.incremental_model.predict(feats)
            predicted_label = "Original (True)" if pred[0] == 0 else "Counterfeit (False)"
            actual_label = infer_actual_label(os.path.basename(self.single_image_path))
            interp = interpret_features(f1, f2, f3, f4)
            result_text = (
                f"Image: {os.path.basename(self.single_image_path)}\n"
                f"  - Crispness: {f1:.2f} ({interp['Crispness']})\n"
                f"  - Edge Density: {f2:.4f} ({interp['Edge Density']})\n"
                f"  - Contrast: {f3:.2f} ({interp['Contrast']})\n"
                f"  - High Frequency Ratio: {f4:.4f} ({interp['High Frequency Ratio']})\n"
                f"Predicted Label: {predicted_label}\n"
            )
            if actual_label != "Unknown":
                result_text += f"Inferred Actual Label: {actual_label}\n"
            result_text += "\n"
            self._append_text(result_text)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def classify_test_folder(self):
        """
        Classify all images in the selected test folder.
        For each image, display feature scores, interpretation, and predicted label.
        If an inferred actual label can be deduced (i.e., not 'Unknown'), it is displayed.
        """
        if not self.test_folder:
            messagebox.showerror("Error", "Please select a test folder.")
            return

        files = os.listdir(self.test_folder)
        if not files:
            messagebox.showinfo("Info", "No files found in the selected folder.")
            return

        for filename in files:
            full_path = os.path.join(self.test_folder, filename)
            if not os.path.isfile(full_path):
                continue
            try:
                feats = extract_features(full_path).reshape(1, -1)
                f1, f2, f3, f4 = feats.flatten()
                pred = self.incremental_model.predict(feats)
                predicted_label = "Original (True)" if pred[0] == 0 else "Counterfeit (False)"
                actual_label = infer_actual_label(filename)
                interp = interpret_features(f1, f2, f3, f4)
                result_text = (
                    f"Image: {filename}\n"
                    f"  - Crispness: {f1:.2f} ({interp['Crispness']})\n"
                    f"  - Edge Density: {f2:.4f} ({interp['Edge Density']})\n"
                    f"  - Contrast: {f3:.2f} ({interp['Contrast']})\n"
                    f"  - High Frequency Ratio: {f4:.4f} ({interp['High Frequency Ratio']})\n"
                    f"Predicted Label: {predicted_label}\n"
                )
                if actual_label != "Unknown":
                    result_text += f"Inferred Actual Label: {actual_label}\n"
                result_text += "\n"
                self._append_text(result_text)
            except Exception as e:
                self._append_text(f"Error processing {filename}: {e}\n")

    def _collect_data(self):
        """
        Extract features from images in the original folder (label=0) and
        the counterfeit folder (label=1).
        """
        X = []
        y = []
        for filename in os.listdir(self.original_folder):
            full_path = os.path.join(self.original_folder, filename)
            if not os.path.isfile(full_path):
                continue
            try:
                feats = extract_features(full_path)
                X.append(feats)
                y.append(0)
            except:
                pass
        for filename in os.listdir(self.counterfeit_folder):
            full_path = os.path.join(self.counterfeit_folder, filename)
            if not os.path.isfile(full_path):
                continue
            try:
                feats = extract_features(full_path)
                X.append(feats)
                y.append(1)
            except:
                pass

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        return X, y

    def _update_metrics(self, X, y):
        """
        Compute and display classification metrics along with a confusion matrix.
        """
        y_pred = self.incremental_model.predict(X)
        report = classification_report(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        # Create a textual representation of the confusion matrix
        cm_text = "Confusion Matrix (Rows=True, Columns=Predicted):\n"
        for row in cm:
            cm_text += " ".join(map(str, row)) + "\n"
        self.metrics_label.config(text=f"F1 Score (Macro): {f1:.4f}")
        metrics_report = f"Classification Report:\n{report}\nAccuracy: {accuracy:.4f}\n{cm_text}\n"
        self._append_text(metrics_report)
        # Plot confusion matrix as a heatmap
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def _append_text(self, msg):
        self.text_display.config(state=tk.NORMAL)
        self.text_display.insert(tk.END, msg)
        self.text_display.see(tk.END)
        self.text_display.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = QRCodeGUI(root, model_path="model.pkl")
    root.mainloop()

if __name__ == "__main__":
    main()


