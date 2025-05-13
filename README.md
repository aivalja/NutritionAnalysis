# Fineli Nutritional Dataset Analysis

This project analyzes the Fineli nutritional dataset containing detailed information for approximately 4,000 food items. It creates and analyzes nutritional network graphs where nodes represent food items and edges represent similarities in nutritional content.

## Features

- **Data preprocessing** for handling missing values and inconsistencies
- **Network graph analysis** including:
  - Construction of nutritional similarity networks
  - Analysis of degree, closeness, and betweenness distributions
  - Clustering coefficient calculations
  - Community detection (Girvan-Newman and Louvain algorithms)
  - Identification of influential items (PageRank, HITS)
  - Network robustness analysis through node removal simulations
  - K-core decomposition

## Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/aivalja/NutritionAnalysis.git
cd NutritionAnalysis
pip install -r requirements.txt
```

## Usage

Run the analysis with:

```bash
python3 main.py
```

## Project Overview

This project performs comprehensive network analysis on the Fineli nutritional dataset, where food items are represented as nodes and their nutritional similarities as edges.

### Key Features

- **Data Preprocessing**: Handles missing values and normalizes nutritional data
- **Network Creation**: Generates similarity-based food networks
- **Centrality Analysis**: Calculates and visualizes degree, closeness, and betweenness distributions
- **Community Detection**: Identifies food communities using Louvain algorithm
- **Advanced Network Metrics**:
  - PageRank for influential food identification
  - HITS algorithm for hub and authority detection
  - K-core decomposition for network structure analysis
  - Assortativity coefficient calculation
  - Network robustness simulation

## Visualizations

The program generates various network visualizations including community graphs, centrality distributions, and influence networks.

## Requirements

See requirements.txt for detailed dependencies.

## Note

This project is computationally intensive and may require significant processing time for large datasets.
