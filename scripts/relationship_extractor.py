import os
import pandas as pd
import numpy as np
import re
import json
import spacy
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import logging
from pathlib import Path
from collections import defaultdict, Counter

# Suppress unnecessary warnings and logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define paths - ensure these are correct for your environment
BASE_PATH = r"D:/Fraud Detection"
NER_RESULTS_PATH = os.path.join(BASE_PATH, "outputs/ner_results")
RELATIONSHIP_OUTPUT_PATH = os.path.join(BASE_PATH, "outputs/relationship_results")
VISUALIZATION_PATH = os.path.join(BASE_PATH, "outputs/visualizations")

# Ensure directories exist
os.makedirs(RELATIONSHIP_OUTPUT_PATH, exist_ok=True)
os.makedirs(VISUALIZATION_PATH, exist_ok=True)

# Load spaCy model for linguistic analysis
print("Loading NLP models...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model loaded successfully")
except:
    print("Downloading SpaCy model (one-time setup)...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model loaded successfully")

class RelationshipExtractor:
    def __init__(self):
        print("Initializing Relationship Extractor...")
        
        # Define relationship patterns
        self.relationship_patterns = {
            "OWNERSHIP": [
                {"POS": ["NOUN", "PROPN"], "DEP": ["nsubj", "compound"]},
                {"LEMMA": ["own", "possess", "have", "hold", "acquire"]},
                {"POS": ["NOUN", "PROPN"], "DEP": ["dobj", "attr"]}
            ],
            "EMPLOYMENT": [
                {"POS": ["NOUN", "PROPN"], "DEP": ["nsubj", "compound"]},
                {"LEMMA": ["work", "employ", "hire", "contract"]},
                {"LEMMA": ["for", "with", "at", "by"]},
                {"POS": ["NOUN", "PROPN"], "DEP": ["pobj"]}
            ],
            "TRANSACTION": [
                {"POS": ["NOUN", "PROPN"], "DEP": ["nsubj", "compound"]},
                {"LEMMA": ["pay", "transfer", "send", "receive", "deposit", "withdraw"]},
                {"LEMMA": ["to", "from"]},
                {"POS": ["NOUN", "PROPN"], "DEP": ["pobj"]}
            ],
            "LOCATION": [
                {"POS": ["NOUN", "PROPN"], "DEP": ["nsubj", "compound"]},
                {"LEMMA": ["locate", "situate", "base", "headquarter", "live"]},
                {"LEMMA": ["in", "at", "near"]},
                {"POS": ["NOUN", "PROPN"], "DEP": ["pobj"]}
            ]
        }
        
        # Dictionary to store high-confidence relationships (these are rule-based)
        self.high_confidence_patterns = {
            r'(.*) is (employed|hired|working) (at|by|with) (.*)': 'EMPLOYMENT',
            r'(.*) works for (.*)': 'EMPLOYMENT',
            r'(.*) owns (.*)': 'OWNERSHIP', 
            r'(.*) is owned by (.*)': 'OWNERSHIP',
            r'(.*) paid (.*) to (.*)': 'TRANSACTION',
            r'(.*) transferred (.*) to (.*)': 'TRANSACTION',
            r'(.*) is located (in|at) (.*)': 'LOCATION',
            r'(.*) is based (in|at) (.*)': 'LOCATION'
        }
        
        # Entity type mappings for relationships
        self.entity_relation_mapping = {
            ('PER', 'ORG'): ['EMPLOYMENT', 'OWNERSHIP'],
            ('ORG', 'ORG'): ['OWNERSHIP', 'TRANSACTION'],
            ('PER', 'PER'): ['TRANSACTION'],
            ('ORG', 'LOC'): ['LOCATION'],
            ('PER', 'LOC'): ['LOCATION'],
            ('ORG', 'MONEY'): ['TRANSACTION'],
            ('PER', 'MONEY'): ['TRANSACTION']
        }
        
        # Load NER results
        self.ner_results_file = os.path.join(NER_RESULTS_PATH, "ner_analysis_results.csv")
        self.documents = self.load_ner_results()
        print(f"Loaded data for {len(self.documents)} documents")
        
        # Prepare relationship graph
        self.global_graph = nx.DiGraph()
        
        print("Relationship Extractor initialized successfully!")
    
    def load_ner_results(self):
        """Load NER results from both the main CSV and individual entity files"""
        if not os.path.exists(self.ner_results_file):
            print(f"ERROR: NER results file not found at {self.ner_results_file}")
            print("Please run the NER processor first.")
            return []
        
        # Load main results
        main_df = pd.read_csv(self.ner_results_file)
        print(f"Found {len(main_df)} documents with NER analysis")
        
        documents = []
        for _, row in main_df.iterrows():
            doc_id = row['document_id']
            entity_file = os.path.join(NER_RESULTS_PATH, f"{doc_id}_entities.csv")
            
            if os.path.exists(entity_file):
                # Load detailed entity information
                entity_df = pd.read_csv(entity_file)
                
                # Group entities by type
                entities = {}
                for entity_type in entity_df['entity_type'].unique():
                    entities[entity_type] = entity_df[entity_df['entity_type'] == entity_type]['value'].tolist()
                
                # Create document record
                doc_record = {
                    'document_id': doc_id,
                    'total_entities': row['total_entities'],
                    'entity_types': row['entity_types'],
                    'entity_complexity': row['entity_complexity'],
                    'risk_entity_count': row['risk_entity_count'],
                    'entities': entities
                }
                documents.append(doc_record)
        
        return documents
    
    def extract_relationships_from_text(self, text, entities_by_type):
        """Extract relationships from text using linguistic patterns"""
        relationships = []
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Check for rule-based high confidence patterns first
        for pattern, rel_type in self.high_confidence_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Multi-group match (for complex patterns)
                    relationships.append({
                        'type': rel_type,
                        'source': match[0].strip(),
                        'target': match[-1].strip(),
                        'confidence': 'HIGH'
                    })
                elif isinstance(match, str):
                    # Single group match
                    parts = match.split(' ')
                    if len(parts) >= 2:
                        relationships.append({
                            'type': rel_type,
                            'source': parts[0].strip(),
                            'target': ' '.join(parts[1:]).strip(),
                            'confidence': 'HIGH'
                        })
        
        # Extract relationships based on entities and their co-occurrence
        flat_entities = {}
        for ent_type, ents in entities_by_type.items():
            for ent in ents:
                flat_entities[ent.lower()] = ent_type
        
        # Find entity co-occurrences within sentences
        for sent in doc.sents:
            sent_text = sent.text.lower()
            found_entities = []
            
            for entity, entity_type in flat_entities.items():
                if entity.lower() in sent_text:
                    found_entities.append((entity, entity_type))
            
            # If we have at least 2 entities in a sentence, they might be related
            if len(found_entities) >= 2:
                for i in range(len(found_entities)):
                    for j in range(i+1, len(found_entities)):
                        ent1, type1 = found_entities[i]
                        ent2, type2 = found_entities[j]
                        
                        # Check if these entity types can have a relationship
                        if (type1, type2) in self.entity_relation_mapping:
                            possible_rels = self.entity_relation_mapping[(type1, type2)]
                            
                            # Use dependency parsing to infer relationship type
                            # For now, use the first possible relationship type
                            rel_type = possible_rels[0]
                            
                            relationships.append({
                                'type': rel_type,
                                'source': ent1,
                                'target': ent2,
                                'confidence': 'MEDIUM'
                            })
        
        # Filter out duplicate relationships
        unique_relationships = []
        seen = set()
        
        for rel in relationships:
            key = (rel['type'], rel['source'].lower(), rel['target'].lower())
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships
    
    def process_documents(self):
        """Process all documents to extract relationships"""
        print(f"\nProcessing {len(self.documents)} documents for relationship extraction...")
        
        all_relationships = []
        document_relationships = {}
        
        for doc in tqdm(self.documents, desc="Extracting relationships"):
            doc_id = doc['document_id']
            entities = doc['entities']
            
            # We need text to extract relationships - let's reconstruct a simplified version
            text = ""
            for entity_type, values in entities.items():
                for value in values:
                    text += f"{value} is a {entity_type}. "
            
            # Extract relationships from text
            relationships = self.extract_relationships_from_text(text, entities)
            
            document_relationships[doc_id] = relationships
            all_relationships.extend(relationships)
            
            # Also add relationships to the global graph
            for rel in relationships:
                source = rel['source']
                target = rel['target']
                rel_type = rel['type']
                
                # Add nodes and edge to graph
                self.global_graph.add_node(source)
                self.global_graph.add_node(target)
                
                # Add edge with relationship type as attribute
                self.global_graph.add_edge(source, target, type=rel_type)
        
        # Analyze and save relationship data
        self.save_relationships(document_relationships, all_relationships)
        self.visualize_relationship_network()
        
        # Return relationship stats
        rel_types = Counter([rel['type'] for rel in all_relationships])
        return {
            'total_relationships': len(all_relationships),
            'by_type': dict(rel_types),
            'document_count': len(document_relationships)
        }
    
    def save_relationships(self, document_relationships, all_relationships):
        """Save relationship extraction results"""
        print(f"Saving relationship results to {RELATIONSHIP_OUTPUT_PATH}...")
        
        # Save all relationships to a single CSV
        relationships_df = pd.DataFrame(all_relationships)
        all_relationships_file = os.path.join(RELATIONSHIP_OUTPUT_PATH, "all_relationships.csv")
        relationships_df.to_csv(all_relationships_file, index=False)
        
        # Save relationships by document
        for doc_id, relationships in document_relationships.items():
            if relationships:  # Only save if there are relationships
                doc_rel_df = pd.DataFrame(relationships)
                doc_file = os.path.join(RELATIONSHIP_OUTPUT_PATH, f"{doc_id}_relationships.csv")
                doc_rel_df.to_csv(doc_file, index=False)
        
        # Save network data as JSON for visualization
        network_data = {
            'nodes': list(self.global_graph.nodes()),
            'edges': [{'source': u, 'target': v, 'type': d['type']} 
                      for u, v, d in self.global_graph.edges(data=True)]
        }
        
        network_file = os.path.join(RELATIONSHIP_OUTPUT_PATH, "relationship_network.json")
        with open(network_file, 'w') as f:
            json.dump(network_data, f)
    
    def visualize_relationship_network(self):
        """Create visualizations of the relationship network"""
        print("\nGenerating network visualizations...")
        
        if len(self.global_graph) == 0:
            print("No relationships found for visualization.")
            return
        
        # Set up color map for relationship types
        rel_types = set([d['type'] for _, _, d in self.global_graph.edges(data=True)])
        colors = plt.cm.tab10.colors
        color_map = {rel_type: colors[i % len(colors)] for i, rel_type in enumerate(rel_types)}
        
        # Prepare edge colors
        edge_colors = [color_map[self.global_graph.edges[edge]['type']] 
                      for edge in self.global_graph.edges()]
        
        # Create main visualization
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(self.global_graph, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(self.global_graph, pos, alpha=0.8, node_size=500)
        nx.draw_networkx_labels(self.global_graph, pos, font_size=10)
        nx.draw_networkx_edges(self.global_graph, pos, width=2, alpha=0.7, edge_color=edge_colors)
        
        # Add legend
        for rel_type, color in color_map.items():
            plt.plot([0], [0], color=color, label=rel_type, linewidth=3)
        
        plt.legend(title="Relationship Types", loc="upper right")
        plt.title("Entity Relationship Network", size=15)
        plt.axis('off')
        
        # Save visualization
        network_viz_file = os.path.join(VISUALIZATION_PATH, "relationship_network.png")
        plt.savefig(network_viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional stats visualizations
        self.create_stats_visualizations()
        
        print(f"Network visualization saved to {network_viz_file}")
    
    def create_stats_visualizations(self):
        """Create additional visualizations for relationship statistics"""
        # Count relationship types
        rel_types = Counter([d['type'] for _, _, d in self.global_graph.edges(data=True)])
        
        # Relationship types distribution
        plt.figure(figsize=(10, 6))
        plt.bar(rel_types.keys(), rel_types.values(), color=plt.cm.tab10.colors[:len(rel_types)])
        plt.title("Distribution of Relationship Types", size=15)
        plt.xlabel("Relationship Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_PATH, "relationship_types.png"), dpi=300)
        plt.close()
        
        # Node connectivity (degree)
        node_degrees = dict(self.global_graph.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        plt.figure(figsize=(10, 6))
        plt.bar([n[0] for n in top_nodes], [n[1] for n in top_nodes], 
                color=plt.cm.viridis(np.linspace(0, 1, len(top_nodes))))
        plt.title("Top 10 Most Connected Entities", size=15)
        plt.xlabel("Entity")
        plt.ylabel("Number of Connections")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_PATH, "top_connected_entities.png"), dpi=300)
        plt.close()

def main():
    """Main function to run relationship extraction"""
    print("Starting Relationship Extraction System...")
    
    extractor = RelationshipExtractor()
    
    # Process all documents
    results = extractor.process_documents()
    
    # Print summary statistics
    print("\n===== Relationship Extraction Complete =====")
    print(f"Processed {results['document_count']} documents")
    print(f"Found {results['total_relationships']} relationships")
    print("\nRelationship types distribution:")
    for rel_type, count in results['by_type'].items():
        print(f"  - {rel_type}: {count}")
    
    print(f"\nResults saved to: {RELATIONSHIP_OUTPUT_PATH}")
    print(f"Visualizations saved to: {VISUALIZATION_PATH}")
    
    print("\nRelationship Extraction completed!")

if __name__ == "__main__":
    main() 