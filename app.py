import streamlit as st
import numpy as np
from PIL import Image
import easyocr
import torch
from transformers import LayoutLMForSequenceClassification, LayoutLMTokenizer
import pandas as pd
import plotly.express as px
import re
from datetime import datetime

# Initialize OCR and models
reader = easyocr.Reader(['en'])
model_name = "microsoft/layoutlm-base-uncased"
tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
model = LayoutLMForSequenceClassification.from_pretrained(model_name)

def process_image(image):
    img_array = np.array(image)
    results = reader.readtext(img_array)
    extracted_text = []
    boxes = []
    for (bbox, text, prob) in results:
        if prob > 0.5:
            extracted_text.append(text)
            boxes.append(bbox)
    return extracted_text, boxes

def analyze_layout(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1)
    return predictions

def extract_amounts(text):
    amounts = []
    amount_pattern = r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
    for t in text:
        matches = re.findall(amount_pattern, t)
        for match in matches:
            try:
                clean_amount = float(match.replace('$', '').replace(',', ''))
                amounts.append(clean_amount)
            except ValueError:
                continue
    return amounts

def extract_dates(text):
    dates = []
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{1,2}-\d{1,2}-\d{2,4}',
        r'\d{1,2}\.\d{1,2}\.\d{2,4}'
    ]
    for t in text:
        for pattern in date_patterns:
            matches = re.findall(pattern, t)
            dates.extend(matches)
    return dates

def detect_fraud(text, layout_analysis):
    fraud_indicators = {
        'Amount Mismatch': False,
        'Duplicate Amounts': False,
        'Negative Amount': False,
        'Date Inconsistency': False,
        'Layout Anomaly': False,
        'Missing Key Info': False,
        'Suspicious Pattern': False
    }
    # Amount checks
    amounts = extract_amounts(text)
    if amounts:
        if len(amounts) > 1:
            total = max(amounts)
            line_items = [amt for amt in amounts if amt != total]
            if line_items and abs(sum(line_items) - total) > 5.0:
                fraud_indicators['Amount Mismatch'] = True
            if len(set(amounts)) != len(amounts):
                fraud_indicators['Duplicate Amounts'] = True
        if any(amt < 0 for amt in amounts):
            fraud_indicators['Negative Amount'] = True

    # Date checks
    dates = extract_dates(text)
    if dates:
        try:
            parsed_dates = []
            for date in dates:
                for fmt in ('%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y', '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y'):
                    try:
                        parsed_dates.append(datetime.strptime(date, fmt))
                        break
                    except ValueError:
                        continue
            if len(parsed_dates) > 1 and not all(parsed_dates[i] <= parsed_dates[i+1] for i in range(len(parsed_dates)-1)):
                fraud_indicators['Date Inconsistency'] = True
        except Exception:
            fraud_indicators['Date Inconsistency'] = True

    # Layout check
    if layout_analysis is not None and torch.any(layout_analysis != 0):
        fraud_indicators['Layout Anomaly'] = True

    # Missing info check
    crucial_info = ['total', 'date', 'invoice', 'bill', 'amount', 'grand total', 'balance due']
    text_lower = ' '.join(text).lower()
    missing_info = [info for info in crucial_info if info not in text_lower]
    if len(missing_info) > 4:
        fraud_indicators['Missing Key Info'] = True

    # Suspicious pattern check
    suspicious_patterns = [
        r'\d{16}',  # Credit card numbers
        r'\d{3}-\d{2}-\d{4}',  # SSN
        r'void',
        r'copy',
        r'duplicate'
    ]
    for pattern in suspicious_patterns:
        if any(re.search(pattern, t, re.IGNORECASE) for t in text):
            fraud_indicators['Suspicious Pattern'] = True
            break

    # Only flag as fraud if 2 or more indicators are True
    fraud = sum(fraud_indicators.values()) >= 2
    return fraud, fraud_indicators

def main():
    st.title("Document Fraud Detection System")
    st.write("Upload a document image to check for fraud.")

    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Document', use_column_width=True)

        with st.spinner('Processing document...'):
            extracted_text, boxes = process_image(image)
            layout_analysis = analyze_layout(extracted_text)
            fraud, fraud_indicators = detect_fraud(extracted_text, layout_analysis)

            st.subheader("Extracted Text")
            st.write(extracted_text)

            st.subheader("Fraud Detection Result")
            if fraud:
                st.error("⚠️ Fraud Detected!")
            else:
                st.success("✅ No Fraud Detected.")

            st.subheader("Fraud Indicators (for transparency)")
            fraud_df = pd.DataFrame({
                'Indicator': list(fraud_indicators.keys()),
                'Detected': ['Yes' if v else 'No' for v in fraud_indicators.values()]
            })
            st.dataframe(fraud_df)

            # Optional: Visual bar chart
            fig = px.bar(fraud_df, x='Indicator', y='Detected',
                        color='Detected',
                        color_discrete_map={'Yes': 'red', 'No': 'green'},
                        title='Fraud Detection Indicators')
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()