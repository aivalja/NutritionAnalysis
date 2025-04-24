import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import seaborn as sns
import powerlaw
from collections import Counter


def print_result(label, value, indent=4):
    """Print a formatted result with appropriate indentation."""
    spaces = " " * indent
    print(f"{spaces}{label}: {value}")


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


def calculate_centralities(G, use_approximation=True, sample_size=500, k_betweenness=50):
    """Calculate various centrality measures for the graph."""
    print_result("Calculating centrality measures", "")
    centrality = {}

    # Basic centrality metrics
    centrality['degree'] = nx.degree_centrality(G)

    # More computationally expensive metrics
    if use_approximation and G.number_of_nodes() > 1000:
        print_result("Using approximation for closeness centrality", "", indent=6)
        sampled_nodes = random.sample(list(G.nodes()), min(sample_size, G.number_of_nodes()))
        centrality['closeness'] = {}

        for node in sampled_nodes:
            centrality['closeness'][node] = nx.closeness_centrality(G, u=node)

        print_result("Using approximation for betweenness centrality", "", indent=6)
        centrality['betweenness'] = nx.betweenness_centrality(G, k=k_betweenness, seed=42)
    else:
        print_result("Calculating full closeness centrality", "", indent=6)
        centrality['closeness'] = nx.closeness_centrality(G)

        print_result("Calculating full betweenness centrality", "", indent=6)
        centrality['betweenness'] = nx.betweenness_centrality(G)

    return centrality


def plot_centrality_histograms(centrality_dict, network_name="Network", save_dir="plots"):
    """Plot histograms for different centrality measures."""
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    centrality_types = centrality_dict.keys()

    # Plot individual histograms
    for c_type in centrality_types:
        values = list(centrality_dict[c_type].values())

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{c_type.capitalize()} Centrality Distribution ({network_name})')
        plt.xlabel('Centrality Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        filename = f"{save_dir}/{c_type}_centrality_{network_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename)

        plt.close()
        print_result("Saved histogram", filename, indent=6)

    # Plot combined figure (optional for comparison)
    if len(centrality_types) > 1:
        fig, axes = plt.subplots(1, len(centrality_types), figsize=(5*len(centrality_types), 5))

        for i, c_type in enumerate(centrality_types):
            values = list(centrality_dict[c_type].values())
            axes[i].hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'{c_type.capitalize()} Centrality')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{save_dir}/combined_centrality_{network_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename)

        plt.show()

def analyze_centrality_power_law(centrality_dict, show_plot=True, save_dir="plots"):
    """
    Analyze if the centrality distributions follow a power law.

    Parameters:
    - centrality_dict: Dictionary of centrality measures from calculate_centralities function
    - show_plot: Whether to display plots
    - save_plots: Whether to save plot images
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    results = {}

    for centrality_type, centrality_values in centrality_dict.items():
        # Extract values from the centrality dictionary
        values = list(centrality_values.values())

        # Skip if insufficient data
        if len(values) < 10:
            print_result(f"{centrality_type.capitalize()} distribution analysis", 
                         "Insufficient data for analysis")
            continue

        # Fit power law (need to handle zeros for some centrality measures)
        # Add a small constant to avoid zeros which powerlaw can't handle
        min_non_zero = min([v for v in values if v > 0]) / 10
        adjusted_values = [v if v > 0 else min_non_zero for v in values]

        fit = powerlaw.Fit(adjusted_values, discrete=False)
        alpha = fit.alpha

        # Compare to exponential distribution
        R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)

        print_result(f"{centrality_type.capitalize()} power-law exponent alpha", f"{alpha:.4f}")
        print_result(f"{centrality_type.capitalize()} log-likelihood ratio test", 
                     f"R={R:.4f}, p-value={p:.4f}")

        if p < 0.05:
            if R > 0:
                print_result(f"{centrality_type.capitalize()} distribution fit", 
                             "Follows a power-law distribution (p < 0.05)")
            else:
                print_result(f"{centrality_type.capitalize()} distribution fit", 
                             "Follows an exponential distribution (p < 0.05)")
        else:
            print_result(f"{centrality_type.capitalize()} distribution fit", 
                         "Neither distribution is significantly favored (p >= 0.05)")

        # Plot the distribution
        plt.figure(figsize=(8, 6))

        # Create histogram using numpy
        hist, bin_edges = np.histogram(values, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Filter out empty bins
        non_zero_indices = hist > 0
        hist_filtered = hist[non_zero_indices]
        bin_centers_filtered = bin_centers[non_zero_indices]

        if len(hist_filtered) > 1:  # Make sure we have at least 2 points for log-log plot
            plt.loglog(bin_centers_filtered, hist_filtered, 'o', markersize=6)

            # Add power law fit line for visualization
            x_range = np.logspace(np.log10(min(bin_centers_filtered)), 
                                 np.log10(max(bin_centers_filtered)), 50)
            # Scale factor for visualization (approximate)
            scale = hist_filtered[0] / (bin_centers_filtered[0] ** -alpha)
            plt.loglog(x_range, scale * x_range**-alpha, 'r-', 
                     label=f'Power Law Fit (α={alpha:.2f})')

            plt.xlabel(f'{centrality_type.capitalize()} (log scale)')
            plt.ylabel('Frequency (log scale)')
            plt.title(f'{centrality_type.capitalize()} Distribution (Log-Log Scale)')
            plt.grid(True, alpha=0.3)
            plt.legend()

            filename = f"{save_dir}/{centrality_type}_powerlaw.png"
            plt.savefig(filename)

            if show_plot:
                plt.show()
            else:
                plt.close()
        else:
            print_result(f"{centrality_type.capitalize()} plotting", 
                         "Not enough data points for log-log plot")

        results[centrality_type] = (alpha, R, p)

    return results

# Example usage
if __name__ == "__main__":
    # Create the nutritional network
    graph_data = create_nutritional_network("Fineli_Rel20", similarity_threshold=0.99)

    centrality_measures = calculate_centralities(graph_data['graph'], use_approximation=False)

    plot_centrality_histograms({'degree': centrality_measures['degree'],
                                'closeness': centrality_measures['closeness'],
                                'betweenness': centrality_measures['betweenness']},
                                "Full Network",save_dir="plots/centrality")

    results = analyze_centrality_power_law(centrality_measures, save_dir="plots/powerlaw")
