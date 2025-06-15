import os
import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import warnings
import logging
from collections import Counter
from tqdm import tqdm

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Define paths
RESULTS_PATH = r"D:/Fraud Detection/outputs/results"  # Path with OCR results and predictions
OUTPUT_PATH = r"D:/Fraud Detection/outputs/ner_results"  # Path for saving NER results
MODEL_PATH = r"D:/Fraud Detection/models"  # Path for models
DATA_PATH = r"D:/Fraud Detection/data"  # Path for input data

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Define NER model - using a pre-trained model designed for document NER
MODEL_NAME = "dslim/bert-base-NER"  # Document NER model

class NERProcessor:
    def __init__(self):
        print("Initializing NER Processor...")
        # Load and initialize the NER model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)
        self.model.eval()  # Set model to evaluation mode
        
        # ID to label mapping for this NER model
        self.id2label = self.model.config.id2label
        print(f"Model loaded with {len(self.id2label)} entity labels")
        print("NER Processor initialized successfully!")
    
    def process_text(self, text):
        """Process text through NER model to extract entities"""
        print(f"Processing text with length: {len(text)}")
        
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Run model inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Convert predictions to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_predictions = [self.id2label[prediction.item()] for prediction in predictions[0]]
        
        # Process NER results
        word_level_predictions = []
        current_entity = None
        current_text = ""
        
        # Process token-level predictions to get word-level entities
        for token, prediction in zip(tokens, token_predictions):
            if token.startswith("##"):
                # Continuation of previous token
                if current_entity:
                    current_text += token[2:]  # Remove ## prefix
            else:
                # If we had a previous entity, add it to our list
                if current_entity and current_text:
                    word_level_predictions.append({
                        "entity": current_entity,
                        "text": current_text
                    })
                
                # Start new entity
                if prediction.startswith("B-") or prediction.startswith("I-"):
                    current_entity = prediction[2:]  # Remove B- or I- prefix
                    current_text = token
                else:
                    current_entity = None
                    current_text = ""
        
        # Don't forget the last entity
        if current_entity and current_text:
            word_level_predictions.append({
                "entity": current_entity,
                "text": current_text
            })
        
        # Group by entity type
        entities = {}
        for pred in word_level_predictions:
            entity_type = pred["entity"]
            entity_text = pred["text"]
            
            if entity_type not in entities:
                entities[entity_type] = []
            
            entities[entity_type].append(entity_text)
        
        # Count the frequency of each entity
        entity_counts = {entity: len(items) for entity, items in entities.items()}
        
        print(f"Found {sum(entity_counts.values())} entities across {len(entity_counts)} categories")
        return entities, entity_counts
    
    def process_prediction_files(self):
        """Process all prediction files from OCR results"""
        print(f"Looking for OCR prediction files in {RESULTS_PATH}...")
        prediction_files = [f for f in os.listdir(RESULTS_PATH) if f.endswith('_predictions.csv')]
        
        if not prediction_files:
            print("No prediction files found. Please ensure OCR has been run first.")
            return
        
        print(f"Found {len(prediction_files)} prediction files to process")
        results = []
        
        # Process each file
        for file in tqdm(prediction_files, desc="Processing files"):
            file_path = os.path.join(RESULTS_PATH, file)
            
            # Extract document ID from filename
            doc_id = file.split('_')[0]
            
            try:
                # Read OCR prediction CSV
                df = pd.read_csv(file_path)
                
                # Combine all words into a single text
                if 'word' in df.columns:
                    text = ' '.join(df['word'].astype(str).tolist())
                    
                    # Process through NER
                    entities, entity_counts = self.process_text(text)
                    
                    # Calculate risk metrics
                    # More entities usually indicates a more complex document
                    entity_complexity = len(entities)
                    
                    # Documents with certain entity types might have higher fraud risk
                    high_risk_entities = ['ORG', 'MONEY', 'DATE']
                    risk_entity_count = sum([len(entities.get(e, [])) for e in high_risk_entities])
                    
                    # Create result record
                    result = {
                        'document_id': doc_id,
                        'total_entities': sum(entity_counts.values()),
                        'entity_types': len(entity_counts),
                        'entity_complexity': entity_complexity,
                        'risk_entity_count': risk_entity_count,
                        'entities': entities,
                        'entity_counts': entity_counts
                    }
                    
                    results.append(result)
                else:
                    print(f"Warning: File {file} doesn't have the expected 'word' column")
                    
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        
        # Save consolidated results
        if results:
            self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """Save NER analysis results to CSV"""
        print(f"Saving NER analysis results for {len(results)} documents...")
        
        # Create DataFrame with core metrics
        df = pd.DataFrame([{
            'document_id': r['document_id'],
            'total_entities': r['total_entities'],
            'entity_types': r['entity_types'],
            'entity_complexity': r['entity_complexity'],
            'risk_entity_count': r['risk_entity_count']
        } for r in results])
        
        # Save to CSV
        output_file = os.path.join(OUTPUT_PATH, "ner_analysis_results.csv")
        df.to_csv(output_file, index=False)
        
        # Save detailed entity information for each document
        for result in results:
            doc_id = result['document_id']
            entities = result['entities']
            
            # Create entity records
            entity_records = []
            for entity_type, values in entities.items():
                for value in values:
                    entity_records.append({
                        'entity_type': entity_type,
                        'value': value
                    })
            
            if entity_records:
                # Save to document-specific CSV
                entity_df = pd.DataFrame(entity_records)
                entity_file = os.path.join(OUTPUT_PATH, f"{doc_id}_entities.csv")
                entity_df.to_csv(entity_file, index=False)
        
        print(f"Results saved to {OUTPUT_PATH}")
        print(f"Main results file: {os.path.join(OUTPUT_PATH, 'ner_analysis_results.csv')}")

def main():
    """Main function to run NER processing"""
    print("Starting NER Processing System...")
    processor = NERProcessor()
    
    # Process all OCR prediction files
    results = processor.process_prediction_files()
    
    # Output summary statistics
    if results:
        total_entities = sum(r['total_entities'] for r in results)
        avg_entities = total_entities / len(results)
        max_entities = max(r['total_entities'] for r in results)
        max_doc = next(r['document_id'] for r in results if r['total_entities'] == max_entities)
        
        print("\n===== NER Processing Complete =====")
        print(f"Processed {len(results)} documents")
        print(f"Found {total_entities} total entities (avg: {avg_entities:.1f} per document)")
        print(f"Document with most entities: {max_doc} ({max_entities} entities)")
        print(f"Results saved to: {OUTPUT_PATH}")
    
    print("NER Processing completed!")

if __name__ == "__main__":
    main() 