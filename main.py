import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess the Fineli dataset
def load_fineli_data(data_dir='.'):
    """
    Loads and preprocesses the Fineli dataset files.

    Args:
        data_dir: Directory containing the CSV files

    Returns:
        Dictionary of preprocessed DataFrames
    """
    # Define file paths
    food_file = os.path.join(data_dir, 'food.csv')
    component_values_file = os.path.join(data_dir, 'component_value.csv')
    component_file = os.path.join(data_dir, 'component.csv')

    # Load the CSV files with correct encoding and separator
    print("Loading data files...")
    food_df = pd.read_csv(food_file, sep=';', encoding='latin1')
    component_values_df = pd.read_csv(component_values_file, sep=';', encoding='latin1')
    component_df = pd.read_csv(component_file, sep=';', encoding='latin1')

    print(f"Loaded {len(food_df)} food items")
    print(f"Loaded {len(component_values_df)} component values")
    print(f"Loaded {len(component_df)} components")

    # Convert decimal values from European format (comma) to standard format (period)
    print("Preprocessing component values...")
    component_values_df['BESTLOC'] = component_values_df['BESTLOC'].astype(str).str.replace(',', '.').astype(float)

    # Check for missing values
    print("Checking for missing values...")
    for df_name, df in [("food", food_df), ("component_values", component_values_df), ("component", component_df)]:
        missing_values = df.isnull().sum().sum()
        print(f"  {df_name}: {missing_values} missing values")

    # Identify potential zero values that might represent missing data
    zero_values = (component_values_df['BESTLOC'] == 0).sum()
    print(f"  Number of zero values in BESTLOC: {zero_values}")

    # Create a pivot table for nutrient analysis
    print("Creating nutrient pivot table...")
    nutrient_pivot = component_values_df.pivot_table(
        values='BESTLOC',
        index='FOODID',
        columns='EUFDNAME',
        aggfunc='first'  # Use first if there are multiple values
    ).reset_index()

    # Merge with food information
    print("Merging with food information...")
    food_nutrients = nutrient_pivot.merge(food_df, on='FOODID', how='left')

    # Check for potential outliers or inconsistent values
    print("Checking for potential outliers...")
    # Get numeric columns (excluding FOODID)
    numeric_columns = food_nutrients.select_dtypes(include=[np.number]).columns.tolist()
    if 'FOODID' in numeric_columns:
        numeric_columns.remove('FOODID')

    # Calculate basic statistics for numeric columns
    stats_df = food_nutrients[numeric_columns].describe().T
    stats_df['missing'] = food_nutrients[numeric_columns].isnull().sum()
    stats_df['zeros'] = (food_nutrients[numeric_columns] == 0).sum()

    print("Preprocessing complete!")

    return {
        'food': food_df,
        'component_values': component_values_df,
        'component': component_df,
        'food_nutrients': food_nutrients,
        'stats': stats_df
    }

def create_nutritional_network(data_dir='.', similarity_threshold=0.85):
    """
    Creates a nutritional network graph where:
    - Nodes represent food items
    - Edges represent similarity based on nutritional content

    Args:
        data_dir: Directory containing the CSV files
        similarity_threshold: Threshold for creating edges between nodes (0-1)

    Returns:
        NetworkX graph of food items
    """
    # Load and preprocess data (using function from previous step)
    data = load_fineli_data(data_dir)
    food_nutrients = data['food_nutrients']

    # Get food item information
    food_ids = food_nutrients['FOODID'].values
    food_names = food_nutrients['FOODNAME'].values

    # Select only nutrient columns (excluding metadata columns)
    nutrient_cols = [col for col in food_nutrients.columns
                    if col not in ['FOODID', 'FOODNAME', 'FOODTYPE', 'PROCESS',
                                  'EDPORT', 'IGCLASS', 'IGCLASSP', 'FUCLASS', 'FUCLASSP']]

    # Fill missing values with 0 (for this initial analysis)
    nutrients_data = food_nutrients[nutrient_cols].fillna(0)

    # Normalize nutritional values using Min-Max scaling
    scaler = MinMaxScaler()
    normalized_nutrients = scaler.fit_transform(nutrients_data)

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(normalized_nutrients)

    # Create a new graph
    G = nx.Graph()

    # Add nodes (food items)
    for i, (food_id, food_name) in enumerate(zip(food_ids, food_names)):
        G.add_node(food_id, name=food_name)

    # Add edges based on similarity threshold
    for i in range(len(food_ids)):
        for j in range(i+1, len(food_ids)):  # Avoid duplicates and self-loops
            similarity = similarity_matrix[i, j]
            if similarity >= similarity_threshold:
                G.add_edge(food_ids[i], food_ids[j], weight=similarity)

    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Using similarity threshold: {similarity_threshold}")

    return {
        'graph': G,
        'food_mapping': dict(zip(food_ids, food_names)),
        'similarity_matrix': similarity_matrix
    }

def visualize_network(graph_data, max_nodes=100):
    """
    Visualizes the nutritional network graph

    Args:
        graph_data: Output from create_nutritional_network
        max_nodes: Maximum number of nodes to display for readability
    """
    G = graph_data['graph']
    food_mapping = graph_data['food_mapping']

    # If the graph is very large, take a subset for visualization
    if G.number_of_nodes() > max_nodes:
        # Get the nodes with the most connections
        sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, d in sorted_nodes[:max_nodes]]
        G_viz = G.subgraph(top_nodes)
        print(f"Visualizing a subset of {len(top_nodes)} most connected nodes")
    else:
        G_viz = G

    # Set up the plot
    plt.figure(figsize=(14, 10))

    # Position nodes using force-directed layout
    pos = nx.spring_layout(G_viz, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G_viz, pos, node_size=100, alpha=0.7)

    # Draw edges with weights affecting thickness
    edge_weights = [G_viz.get_edge_data(u, v)['weight'] * 2 for u, v in G_viz.edges()]
    nx.draw_networkx_edges(G_viz, pos, width=edge_weights, alpha=0.4)

    # Add node labels (food names) with smaller font
    labels = {node: food_mapping[node] for node in G_viz.nodes()}
    nx.draw_networkx_labels(G_viz, pos, labels=labels, font_size=8, font_family='sans-serif')

    plt.title("Nutritional Similarity Network of Food Items")
    plt.axis('off')
    plt.tight_layout()

    # Save the visualization
    plt.savefig('nutritional_network.png', dpi=300, bbox_inches='tight')
    print("Network visualization saved as 'nutritional_network.png'")

    # Show plot
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create the nutritional network
    graph_data = create_nutritional_network("Fineli_Rel20", similarity_threshold=0.99)

    # Visualize the network
    visualize_network(graph_data, max_nodes=1000)

    # Display some basic network metrics
    G = graph_data['graph']
    print("\nNetwork Metrics:")
    print(f"Number of nodes (food items): {G.number_of_nodes()}")
    print(f"Number of edges (similarities): {G.number_of_edges()}")

    # Find the most connected food items
    node_degrees = dict(G.degree())
    top_connected = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\nMost connected food items:")
    for food_id, degree in top_connected:
        print(f"{graph_data['food_mapping'][food_id]}: {degree} connections")
