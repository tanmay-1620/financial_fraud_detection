import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import os
import json
from datetime import datetime
import numpy as np
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Constants - Updated paths to be more flexible
BASE_PATH = r"D:/Fraud Detection"
RELATIONSHIP_OUTPUT_PATH = os.path.join(BASE_PATH, "outputs", "relationship_results")
NER_RESULTS_PATH = os.path.join(BASE_PATH, "outputs", "ner_results")

# Ensure output directories exist
os.makedirs(RELATIONSHIP_OUTPUT_PATH, exist_ok=True)
os.makedirs(NER_RESULTS_PATH, exist_ok=True)

class FraudDetectionDashboard:
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        """Load all necessary data for the dashboard"""
        # Load relationship data
        self.relationships = self._load_relationship_data()
        
        # Load NER results
        self.ner_results = self._load_ner_results()
        
        # Create network graph
        self.G = self._create_network_graph()
        
    def _load_relationship_data(self):
        """Load relationship data from CSV files"""
        relationships = []
        if os.path.exists(RELATIONSHIP_OUTPUT_PATH):
            for file in os.listdir(RELATIONSHIP_OUTPUT_PATH):
                if file.endswith('_relationships.csv'):
                    try:
                        df = pd.read_csv(os.path.join(RELATIONSHIP_OUTPUT_PATH, file))
                        relationships.append(df)
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
        return pd.concat(relationships) if relationships else pd.DataFrame()
    
    def _load_ner_results(self):
        """Load NER results from CSV files"""
        ner_results = []
        if os.path.exists(NER_RESULTS_PATH):
            for file in os.listdir(NER_RESULTS_PATH):
                if file.endswith('_entities.csv'):
                    try:
                        df = pd.read_csv(os.path.join(NER_RESULTS_PATH, file))
                        ner_results.append(df)
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
        return pd.concat(ner_results) if ner_results else pd.DataFrame()
    
    def _create_network_graph(self):
        """Create a network graph from relationships"""
        G = nx.DiGraph()
        
        if not self.relationships.empty:
            for _, row in self.relationships.iterrows():
                G.add_edge(
                    row['source'],
                    row['target'],
                    relationship_type=row['type'],
                    confidence=row['confidence']
                )
        
        return G
    
    def plot_network_graph(self):
        """Create an interactive network graph using plotly"""
        if not self.G.edges():
            st.warning("No relationships found to visualize.")
            return None
        
        # Create node positions using spring layout
        pos = nx.spring_layout(self.G)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in self.G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"{edge[2]['relationship_type']} ({edge[2]['confidence']})")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            mode='lines',
            text=edge_text
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_connections = []
        
        for node in self.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_connections.append(len(list(self.G.neighbors(node))))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=node_connections,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left'
                )
            )
        )
        
        # Create figure with updated layout
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='Relationship Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def get_relationship_stats(self):
        """Calculate and return relationship statistics"""
        if self.relationships.empty:
            return pd.DataFrame()
        
        stats = {
            'Total Relationships': len(self.relationships),
            'Unique Entities': len(set(self.relationships['source'].unique()) | set(self.relationships['target'].unique())),
            'Relationship Types': self.relationships['type'].nunique(),
            'High Confidence Relationships': len(self.relationships[self.relationships['confidence'] == 'HIGH']),
            'Average Relationships per Entity': len(self.relationships) / len(set(self.relationships['source'].unique()) | set(self.relationships['target'].unique()))
        }
        
        return pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    
    def get_entity_risk_scores(self):
        """Calculate risk scores for entities based on their relationships"""
        if self.relationships.empty:
            return pd.DataFrame()
        
        # Calculate risk scores based on relationship types and confidence
        risk_scores = {}
        for _, row in self.relationships.iterrows():
            source = row['source']
            target = row['target']
            rel_type = row['type']
            confidence = row['confidence']
            
            # Assign risk weights based on relationship type
            risk_weights = {
                'TRANSACTION': 2.0,
                'OWNERSHIP': 1.5,
                'EMPLOYMENT': 1.0,
                'LOCATION': 0.5
            }
            
            # Calculate base risk
            base_risk = risk_weights.get(rel_type, 1.0)
            
            # Adjust for confidence
            confidence_multiplier = 1.5 if confidence == 'HIGH' else 1.0
            
            # Update risk scores
            risk_scores[source] = risk_scores.get(source, 0) + (base_risk * confidence_multiplier)
            risk_scores[target] = risk_scores.get(target, 0) + (base_risk * confidence_multiplier)
        
        # Convert to DataFrame
        risk_df = pd.DataFrame(list(risk_scores.items()), columns=['Entity', 'Risk Score'])
        risk_df = risk_df.sort_values('Risk Score', ascending=False)
        
        return risk_df

def main():
    st.title("🔍 Fraud Detection Dashboard")
    
    # Initialize dashboard
    dashboard = FraudDetectionDashboard()
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    view_option = st.sidebar.selectbox(
        "Select View",
        ["Network Graph", "Risk Analysis", "Entity Analysis"]
    )
    
    # Main content
    if view_option == "Network Graph":
        st.header("Relationship Network Visualization")
        fig = dashboard.plot_network_graph()
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        
        # Show relationship statistics
        st.subheader("Relationship Statistics")
        stats_df = dashboard.get_relationship_stats()
        st.dataframe(stats_df, use_container_width=True)
        
    elif view_option == "Risk Analysis":
        st.header("Entity Risk Analysis")
        risk_df = dashboard.get_entity_risk_scores()
        
        if not risk_df.empty:
            # Plot risk scores
            fig = px.bar(risk_df.head(10), 
                        x='Entity', 
                        y='Risk Score',
                        title='Top 10 High-Risk Entities')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed risk scores
            st.subheader("Detailed Risk Scores")
            st.dataframe(risk_df, use_container_width=True)
        else:
            st.warning("No risk analysis data available.")
        
    else:  # Entity Analysis
        st.header("Entity Analysis")
        
        if not dashboard.ner_results.empty:
            # Entity type distribution
            entity_types = dashboard.ner_results['entity_type'].value_counts()
            fig = px.pie(values=entity_types.values, 
                        names=entity_types.index,
                        title='Entity Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show entity details
            st.subheader("Entity Details")
            st.dataframe(dashboard.ner_results, use_container_width=True)
        else:
            st.warning("No entity data available for analysis.")

if __name__ == "__main__":
    main() 