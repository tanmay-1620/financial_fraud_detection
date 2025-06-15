# Document Fraud Detection System

This application provides a user-friendly interface for detecting potential fraud in document images using OCR, layout analysis, and fraud detection algorithms.

## Features

- Upload and process document images (PNG, JPG, JPEG)
- Extract text using OCR (EasyOCR)
- Analyze document layout using LayoutLM
- Detect potential fraud indicators
- Visualize results with interactive charts
- Detailed analysis report

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Navigate to the project directory:
```bash
cd "Fraud Detection"
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## How to Use

1. Click on "Choose an image file" to upload a document image
2. Wait for the processing to complete
3. View the results:
   - Extracted text from the document
   - Fraud detection indicators
   - Visual representation of results
   - Detailed analysis

## Fraud Detection Indicators

The system checks for several fraud indicators:

1. **Amount Mismatch**: Detects inconsistencies in monetary amounts
2. **Suspicious Patterns**: Identifies unusual patterns in the document
3. **Layout Anomalies**: Detects irregularities in document layout

# 💾 Model Weights

**Note:** Model weights are not included in this repository due to file size limitations.

To use the pretrained model, download weights from:
https://huggingface.co/dslim/bert-base-NER

After downloading, place the files in the following directory:
```
models/models--dslim--bert-base-NER/snapshots/
```
If the directory does not exist, create it as shown above.

**Example:**
```
Fraud Detection/
├── models/
│   └── models--dslim--bert-base-NER/
│       └── snapshots/
│           └── <downloaded model files here>
```

## Screenshots

Take screenshots of:
1. The uploaded document
2. Extracted text results
3. Fraud detection visualization
4. Detailed analysis section

## Note

This is a demonstration version. The fraud detection logic in the `detect_fraud()` function should be customized based on your specific requirements and trained models. 
