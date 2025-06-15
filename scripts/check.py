import os
import cv2
import easyocr
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from PIL import Image
import numpy as np
import sys
import io
import warnings
import logging
import csv
from datetime import datetime
import time

# Completely disable all warnings and logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("easyocr").setLevel(logging.ERROR)

# Define a context manager to suppress stdout/stderr
class SuppressOutput:
    def __init__(self):
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

# Define model constants
MODEL_NAME = "microsoft/layoutlm-base-uncased"

# Define Paths for SROIE dataset - focus on task2_train folder only
task2_folder = r"D:/Fraud Detection/data/SROIE Dataset/SROIE2019/SROIE2019/task2_train/"
output_path = r"D:\Fraud Detection\outputs"
results_path = os.path.join(output_path, "results")

# Ensure output directories exist
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Step 1: OCR Processing with EasyOCR - create reader once
reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'])
    return reader

def perform_ocr(image_path):
    # Use the global reader
    reader = get_ocr_reader()
    result = reader.readtext(image_path)
    
    # Extract bounding boxes and text
    boxes = []
    text = []
    for detection in result:
        boxes.append(detection[0])  # The coordinates of the detected text box
        text.append(detection[1])   # The detected text

    return boxes, text

# Step 2: Layout Analysis (visualize OCR results) - with robust error handling
def visualize_ocr_results(image_path, boxes, text, skip_visualization=False):
    if skip_visualization:
        return
        
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return
            
        for box, t in zip(boxes, text):
            try:
                # Ensure box is properly formatted for np.array
                if not box or len(box) < 4:
                    continue
                    
                pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Safe coordinate extraction
                try:
                    if isinstance(box[0], (list, tuple, np.ndarray)) and len(box[0]) >= 2:
                        x, y = int(box[0][0]), int(box[0][1] - 10)
                    else:
                        x, y = int(pts[0][0][0]), int(pts[0][0][1] - 10)
                    
                    # Make sure coordinates are within image boundaries
                    h, w = image.shape[:2]
                    x = max(0, min(x, w-1))
                    y = max(0, min(y, h-1))
                    
                    # Truncate text if too long
                    display_text = t[:20] + '...' if len(t) > 20 else t
                    cv2.putText(image, display_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    # Skip adding this text annotation
                    pass
            except Exception as e:
                # Skip this box if there's an error
                continue
        
        output_image_path = os.path.join(output_path, os.path.basename(image_path))
        cv2.imwrite(output_image_path, image)
    except Exception as e:
        print(f"Warning: Error visualizing OCR results: {e}")

# Step 3: Prepare data for LayoutLM (Tokenizing)
# Pre-load tokenizer
tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        with SuppressOutput():
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

def prepare_layoutlm_input(image_path, boxes, text):
    # Convert image to PIL format for LayoutLM processing
    pil_image = Image.open(image_path).convert("RGB")
    
    # Convert EasyOCR boxes to LayoutLM format [x0, y0, x1, y1]
    normalized_boxes = []
    for box in boxes:
        try:
            # Calculate min/max coordinates to get the bounding box
            x_coordinates = [point[0] for point in box]
            y_coordinates = [point[1] for point in box]
            
            x0 = int(min(x_coordinates))
            y0 = int(min(y_coordinates))
            x1 = int(max(x_coordinates))
            y1 = int(max(y_coordinates))
            
            normalized_boxes.append([x0, y0, x1, y1])
        except Exception:
            # Add a default box if calculation fails
            normalized_boxes.append([0, 0, 10, 10])
    
    # Get the pre-loaded tokenizer
    tokenizer = get_tokenizer()
    
    # Use a regular tokenization without layout information
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    return encoding, pil_image, text, normalized_boxes

# Step 4: LayoutLM Model Inference - pre-load model
model = None

def get_model():
    global model
    if model is None:
        with SuppressOutput():
            model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    return model

def run_layoutlm_inference(encoding):
    # Get the pre-loaded model
    model = get_model()
    
    # Perform Inference
    with torch.no_grad():
        outputs = model(**encoding)
    
    return outputs

# Step 5: Process the output
def process_layoutlm_output(outputs):
    # Outputs are logits, let's take the first output (assuming batch size is 1)
logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1)
    return predicted_class

# Function to save predictions to CSV
def save_predictions(image_path, text, predicted_classes):
    # Create a unique filename based on the original image and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_filename = f"{base_name}_{timestamp}_predictions.csv"
    csv_path = os.path.join(results_path, csv_filename)
    
    # Map predictions to words - for simplicity, just map each word to a prediction
    results = []
    try:
        # Just use a simple 1:1 mapping as best we can
        pred_list = predicted_classes[0].tolist()
        pred_len = len(pred_list)
        text_len = len(text)
        
        # Ensure number of predictions is at least the number of words
        for i in range(min(text_len, pred_len)):
            results.append({
                "word": text[i],
                "predicted_class": pred_list[i],
                "class_label": "Suspicious" if pred_list[i] == 1 else "Normal"
            })
            
        # If we have fewer predictions than words, assign the most common class to remaining words
        if text_len > pred_len:
            # Find most common class
            from collections import Counter
            common_class = Counter(pred_list).most_common(1)[0][0]
            
            for i in range(pred_len, text_len):
                results.append({
                    "word": text[i],
                    "predicted_class": common_class,
                    "class_label": "Suspicious" if common_class == 1 else "Normal"
                })
    except Exception as e:
        # Last resort fallback - save all words with a general prediction
        print(f"Warning: Error mapping predictions to words: {e}")
        try:
            # See if there's any positive class
            if 1 in predicted_classes.unique().tolist():
                overall_pred = 1
            else:
                overall_pred = 0
        except:
            # If that fails, just default to 0
            overall_pred = 0
            
        for word in text:
            results.append({
                "word": word,
                "predicted_class": overall_pred,
                "class_label": "Suspicious" if overall_pred == 1 else "Normal"
            })
    
    # Write to CSV
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['word', 'predicted_class', 'class_label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return None, 0, 0
    
    # Count suspicious words
    suspicious_count = sum(1 for r in results if r["predicted_class"] == 1)
    total_words = len(results)
    
    return csv_path, suspicious_count, total_words

# Function to get already processed files
def get_processed_files():
    processed_files = set()
    if os.path.exists(results_path):
        for file in os.listdir(results_path):
            if file.endswith('_predictions.csv'):
                base_name = file.split('_')[0]
                processed_files.add(base_name)
    return processed_files

# Process only task2_train folder with optimizations
def process_task2_dataset(max_images=200, skip_visualization=True):
    start_time = time.time()
    processed_count = 0
    
    # Get already processed files
    processed_files = get_processed_files()
    print(f"Found {len(processed_files)} already processed files")
    
    # Check if task2_folder exists
    if not os.path.exists(task2_folder):
        print(f"Task 2 folder {task2_folder} not found.")
        return
    
    # Collect image files only from task2_train folder
    image_files = []
    for root, dirs, files in os.walk(task2_folder):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                # Skip duplicates with (1) in filename
                if "(1)" not in file:
                    # Skip already processed files
                    base_name = os.path.splitext(file)[0]
                    if base_name not in processed_files:
                        image_files.append(os.path.join(root, file))
    
    # Sort for consistent processing
    image_files.sort()
    
    total_files = len(image_files)
    print(f"Found {total_files} unprocessed images in task2_train folder")
    
    # Pre-load models
    print("Pre-loading models...")
    get_ocr_reader()
    get_tokenizer()
    get_model()
    print("Models loaded")
    
    # Process files with a limit
    for i, image_path in enumerate(image_files[:max_images]):
        if processed_count >= max_images:
            break
            
        print(f"Processing [{i+1}/{min(total_files, max_images)}]: {image_path}")
        
        try:
            # Measure time for this image
            img_start_time = time.time()
            
            # Step 1: Perform OCR on the image
            boxes, text = perform_ocr(image_path)

            if not text:
                print(f"No text detected in {image_path}, skipping...")
                continue

            # Step 2: Visualize the OCR results (optional)
            visualize_ocr_results(image_path, boxes, text, skip_visualization)

            # Step 3: Prepare data for LayoutLM
            encoding, pil_image, text, normalized_boxes = prepare_layoutlm_input(image_path, boxes, text)

            # Step 4: Run LayoutLM model inference
            outputs = run_layoutlm_inference(encoding)

            # Step 5: Process the LayoutLM output
            predicted_class = process_layoutlm_output(outputs)
            
            # Step 6: Save predictions to CSV
            csv_path, suspicious_count, total_words = save_predictions(image_path, text, predicted_class)
            
            # Print a summary
            if suspicious_count > 0:
                risk_level = "HIGH" if suspicious_count / total_words > 0.2 else "MEDIUM"
                print(f"Result: {risk_level} RISK - {suspicious_count}/{total_words} suspicious elements")
            else:
                print(f"Result: LOW RISK - No suspicious elements found")
                
            print(f"Saved detailed analysis to: {csv_path}")
            
            # Track processing time
            img_duration = time.time() - img_start_time
            print(f"Image processed in {img_duration:.2f} seconds")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print overall stats
    total_duration = time.time() - start_time
    print(f"\nProcessing complete:")
    print(f"Processed {processed_count} images in {total_duration:.2f} seconds")
    print(f"Average time per image: {total_duration/max(1, processed_count):.2f} seconds")

# Main function to execute the pipeline
def main():
    print(f"Starting Optimized OCR and Layout Analysis Pipeline for task2_train...")
    print(f"Output directory: {results_path}")
    
    # Set the maximum number of images to process (can be adjusted)
    max_images = 200
    
    # Skip visualization to save time
    skip_visualization = True
    
    process_task2_dataset(max_images, skip_visualization)
    print("OCR and Layout Analysis Complete.")

if __name__ == "__main__":
    main()