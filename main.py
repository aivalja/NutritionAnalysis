import random
import os
import pickle
from typing import Dict, Any, Callable, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import louvain_communities


def load_component_names(data_dir="."):
    """
    Loads the English component names mapping from eufdname_EN.csv

    Args:
        data_dir: Directory containing the CSV files

    Returns:
        Dictionary mapping component codes to readable names
    """
    component_names_file = os.path.join(data_dir, "eufdname_EN.csv")

    # Load the CSV with correct encoding and separator
    comp_names_df = pd.read_csv(component_names_file, sep=";", encoding="latin1")

    # Create a dictionary mapping code to readable name
    comp_name_map = dict(zip(comp_names_df.iloc[:, 0], comp_names_df.iloc[:, 1]))

    return comp_name_map


def load_or_calculate(
    file_path: str,
    calculate_func: Callable,
    calculate_args: Optional[List] = None,
    calculate_kwargs: Optional[Dict] = None,
    description: str = "data",
    post_process_func: Optional[Callable] = None,
) -> Any:
    """
    Load data from file if it exists, otherwise calculate and save it.

    Args:
        file_path: Path to the pickle file
        calculate_func: Function to call if data needs to be calculated
        calculate_args: Positional arguments for calculate_func
        calculate_kwargs: Keyword arguments for calculate_func
        description: Description of the data being processed (for logging)
        post_process_func: Optional function to process result before saving

    Returns:
        The loaded or calculated data
    """
    calculate_args = calculate_args or []
    calculate_kwargs = calculate_kwargs or {}

    if os.path.exists(file_path):
        print(f"Loading {description} from {file_path}...")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"{description.capitalize()} loaded.")
    else:
        print(f"Calculating {description}...")
        data = calculate_func(*calculate_args, **calculate_kwargs)

        # Apply post-processing if specified
        if post_process_func:
            data = post_process_func(data)

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        print(f"Saving {description} to {file_path}...")
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"{description.capitalize()} saved.")

    return data


def process_girvan_newman(generator, max_communities=None):
    """
    Convert Girvan-Newman generator to a list of communities.

    Args:
        generator: Girvan-Newman generator
        max_communities: Maximum number of community sets to generate

    Returns:
        List of community sets (each set is a tuple of frozensets)
    """
    # Take a limited number of community divisions
    if max_communities:
        return [next(generator) for _ in range(max_communities)]
    # Convert entire generator to list (may be large!)
    return list(generator)


def print_task_header(task_num, task_name):
    """Print a formatted task header."""
    print(f"\n{'='*50}")
    print(f"Task {task_num}: {task_name}")
    print(f"{'='*50}")


def print_subtask(subtask_letter, subtask_name):
    """Print a formatted subtask header."""
    print(f"\n  Part {subtask_letter.upper()}: {subtask_name}")


def print_result(label, value, indent=4):
    """Print a formatted result with appropriate indentation."""
    spaces = " " * indent
    print(f"{spaces}{label}: {value}")


# Function to load and preprocess the Fineli dataset
def load_fineli_data(data_dir="."):
    """
    Loads and preprocesses the Fineli dataset files.

    Args:
        data_dir: Directory containing the CSV files

    Returns:
        Dictionary of preprocessed DataFrames
    """
    # Define file paths
    food_file = os.path.join(data_dir, "food.csv")
    component_values_file = os.path.join(data_dir, "component_value.csv")
    component_file = os.path.join(data_dir, "component.csv")

    # Load the CSV files with correct encoding and separator
    print("Loading data files...")
    food_df = pd.read_csv(food_file, sep=";", encoding="latin1")
    component_values_df = pd.read_csv(component_values_file, sep=";", encoding="latin1")
    component_df = pd.read_csv(component_file, sep=";", encoding="latin1")

    print(f"Loaded {len(food_df)} food items")
    print(f"Loaded {len(component_values_df)} component values")
    print(f"Loaded {len(component_df)} components")

    # Convert decimal values from European format (comma) to standard format (period)
    print("Preprocessing component values...")
    component_values_df["BESTLOC"] = (
        component_values_df["BESTLOC"].astype(str).str.replace(",", ".").astype(float)
    )

    # Check for missing values
    print("Checking for missing values...")
    for df_name, df in [
        ("food", food_df),
        ("component_values", component_values_df),
        ("component", component_df),
    ]:
        missing_values = df.isnull().sum().sum()
        print(f"  {df_name}: {missing_values} missing values")

    # Identify potential zero values that might represent missing data
    zero_values = (component_values_df["BESTLOC"] == 0).sum()
    print(f"  Number of zero values in BESTLOC: {zero_values}")

    # Create a pivot table for nutrient analysis
    print("Creating nutrient pivot table...")
    nutrient_pivot = component_values_df.pivot_table(
        values="BESTLOC",
        index="FOODID",
        columns="EUFDNAME",
        aggfunc="first",  # Use first if there are multiple values
    ).reset_index()

    # Fill NaN values with 0 in the pivot table
    print("Filling missing nutrient values with 0...")
    nutrient_pivot = nutrient_pivot.fillna(0)

    # Merge with food information
    print("Merging with food information...")
    food_nutrients = nutrient_pivot.merge(food_df, on="FOODID", how="left")

    # Check for potential outliers or inconsistent values
    print("Checking for potential outliers...")
    # Get numeric columns (excluding FOODID)
    numeric_columns = food_nutrients.select_dtypes(include=[np.number]).columns.tolist()
    if "FOODID" in numeric_columns:
        numeric_columns.remove("FOODID")

    # Calculate basic statistics for numeric columns
    stats_df = food_nutrients[numeric_columns].describe().T
    stats_df["missing"] = food_nutrients[numeric_columns].isnull().sum()
    stats_df["zeros"] = (food_nutrients[numeric_columns] == 0).sum()

    print("Preprocessing complete!")

    return {
        "food": food_df,
        "component_values": component_values_df,
        "component": component_df,
        "food_nutrients": food_nutrients,
        "stats": stats_df,
    }


def create_nutritional_network(data_dir=".", similarity_threshold=0.85):
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

    print_task_header(1, "Load the Fineli dataset")

    # Load and preprocess data (using function from previous step)
    data = load_fineli_data(data_dir)
    food_nutrients = data["food_nutrients"]

    # Get food item information
    food_ids = food_nutrients["FOODID"].values
    food_names = food_nutrients["FOODNAME"].values

    # Select only nutrient columns (excluding metadata columns)
    nutrient_cols = [
        col
        for col in food_nutrients.columns
        if col
        not in [
            "FOODID",
            "FOODNAME",
            "FOODTYPE",
            "PROCESS",
            "EDPORT",
            "IGCLASS",
            "IGCLASSP",
            "FUCLASS",
            "FUCLASSP",
        ]
    ]

    # Fill missing values with 0 (for this initial analysis)
    nutrients_data = food_nutrients[nutrient_cols].fillna(0)

    # Normalize nutritional values using Min-Max scaling
    scaler = MinMaxScaler()
    normalized_nutrients = scaler.fit_transform(nutrients_data)

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(normalized_nutrients)

    print_task_header(2, "Generate a nutritional network graph")

    # Create a new graph
    G = nx.Graph()

    # Add nodes (food items)
    for i, (food_id, food_name) in enumerate(zip(food_ids, food_names)):
        G.add_node(food_id, name=food_name)

    # Add edges based on similarity threshold
    for i in range(len(food_ids)):
        for j in range(i + 1, len(food_ids)):
            similarity = similarity_matrix[i, j]
            if similarity >= similarity_threshold:
                G.add_edge(food_ids[i], food_ids[j], weight=similarity)

    print(
        f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )
    print(f"Using similarity threshold: {similarity_threshold}")

    return {
        "graph": G,
        "food_mapping": dict(zip(food_ids, food_names)),
        "similarity_matrix": similarity_matrix,
        "food_nutrients": food_nutrients,
    }


def analyze_community_nutrition(graph_data, communities, data_dir="."):
    """
    Analyzes the nutritional composition of each community with proper nutrient names.

    Args:
        graph_data: Dictionary containing graph, food mapping, and nutritional data
        communities: List of communities (each community is a set of node IDs)
        data_dir: Directory containing the CSV files

    Returns:
        DataFrame containing average nutritional values for each community
    """
    G = graph_data["graph"]
    food_nutrients = graph_data["food_nutrients"]

    # Load component name mappings
    component_names = load_component_names(data_dir)

    # Fill any remaining NaN values with 0 to ensure calculations work correctly
    food_nutrients = food_nutrients.fillna(0)

    # Key nutritional attributes to analyze
    key_nutrients = [
        col
        for col in food_nutrients.columns
        if col
        not in [
            "FOODID",
            "FOODNAME",
            "FOODTYPE",
            "PROCESS",
            "EDPORT",
            "IGCLASS",
            "IGCLASSP",
            "FUCLASS",
            "FUCLASSP",
        ]
        and pd.api.types.is_numeric_dtype(food_nutrients[col])
    ]

    # Create a DataFrame to store community nutrition information
    community_nutrition = []

    # Analyze each community
    for i, community in enumerate(communities):
        # Get nutritional data for foods in this community
        community_foods = food_nutrients[food_nutrients["FOODID"].isin(community)]

        # Skip if no foods match
        if community_foods.empty:
            continue

        # Calculate average nutritional values
        avg_nutrients = community_foods[key_nutrients].mean().to_dict()

        # Add community info
        avg_nutrients["community_id"] = i
        avg_nutrients["size"] = len(community)
        avg_nutrients["food_examples"] = ", ".join(
            community_foods["FOODNAME"].head(3).tolist()
        )

        # Add to results
        community_nutrition.append(avg_nutrients)

    # Convert to DataFrame
    result_df = pd.DataFrame(community_nutrition)

    # Final check for any NaNs that might have slipped through
    result_df = result_df.fillna(0)

    return result_df, component_names



def visualize_network(graph_data, max_nodes=100):
    """
    Visualizes the nutritional network graph

    Args:
        graph_data: Output from create_nutritional_network
        max_nodes: Maximum number of nodes to display for readability
    """
    G = graph_data["graph"]
    food_mapping = graph_data["food_mapping"]

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
    edge_weights = [G_viz.get_edge_data(u, v)["weight"] * 2 for u, v in G_viz.edges()]
    nx.draw_networkx_edges(G_viz, pos, width=edge_weights, alpha=0.4)

    # Add node labels (food names) with smaller font
    labels = {node: food_mapping[node] for node in G_viz.nodes()}
    nx.draw_networkx_labels(
        G_viz, pos, labels=labels, font_size=8, font_family="sans-serif"
    )

    plt.title("Nutritional Similarity Network of Food Items")
    plt.axis("off")
    plt.tight_layout()

    # Save the visualization
    plt.savefig("nutritional_network.png", dpi=300, bbox_inches="tight")
    print("Network visualization saved as 'nutritional_network.png'")

    # Show plot
    plt.show()


def calculate_centralities(
    G, use_approximation=True, sample_size=500, k_betweenness=50
):
    """Calculate various centrality measures for the graph."""
    print_result("Calculating centrality measures", "")
    centrality = {}

    # Basic centrality metrics
    centrality["degree"] = nx.degree_centrality(G)

    # More computationally expensive metrics
    if use_approximation and G.number_of_nodes() > 1000:
        print_result("Using approximation for closeness centrality", "", indent=6)
        sampled_nodes = random.sample(
            list(G.nodes()), min(sample_size, G.number_of_nodes())
        )
        centrality["closeness"] = {}

        for node in sampled_nodes:
            centrality["closeness"][node] = nx.closeness_centrality(G, u=node)

        print_result("Using approximation for betweenness centrality", "", indent=6)
        centrality["betweenness"] = nx.betweenness_centrality(
            G, k=k_betweenness, seed=42
        )
    else:
        print_result("Calculating full closeness centrality", "", indent=6)
        centrality["closeness"] = nx.closeness_centrality(G)

        print_result("Calculating full betweenness centrality", "", indent=6)
        centrality["betweenness"] = nx.betweenness_centrality(G)

    return centrality


def plot_centrality_histograms(
    centrality_dict, network_name="Network", save_dir="plots"
):
    """Plot histograms for different centrality measures."""
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    centrality_types = centrality_dict.keys()

    # Plot individual histograms
    for c_type in centrality_types:
        values = list(centrality_dict[c_type].values())

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        plt.title(f"{c_type.capitalize()} Centrality Distribution ({network_name})")
        plt.xlabel("Centrality Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        filename = f"{save_dir}/{c_type}_centrality_{network_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename)

        plt.close()
        print_result("Saved histogram", filename, indent=6)

    # Plot combined figure (optional for comparison)
    if len(centrality_types) > 1:
        fig, axes = plt.subplots(
            1, len(centrality_types), figsize=(5 * len(centrality_types), 5)
        )

        for i, c_type in enumerate(centrality_types):
            values = list(centrality_dict[c_type].values())
            axes[i].hist(values, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
            axes[i].set_title(f"{c_type.capitalize()} Centrality")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
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
            print_result(
                f"{centrality_type.capitalize()} distribution analysis",
                "Insufficient data for analysis",
            )
            continue

        # Fit power law (need to handle zeros for some centrality measures)
        # Add a small constant to avoid zeros which powerlaw can't handle
        min_non_zero = min([v for v in values if v > 0]) / 10
        adjusted_values = [v if v > 0 else min_non_zero for v in values]

        fit = powerlaw.Fit(adjusted_values, discrete=False)
        alpha = fit.alpha

        # Compare to exponential distribution
        R, p = fit.distribution_compare(
            "power_law", "exponential", normalized_ratio=True
        )

        print_result(
            f"{centrality_type.capitalize()} power-law exponent alpha", f"{alpha:.4f}"
        )
        print_result(
            f"{centrality_type.capitalize()} log-likelihood ratio test",
            f"R={R:.4f}, p-value={p:.4f}",
        )

        if p < 0.05:
            if R > 0:
                print_result(
                    f"{centrality_type.capitalize()} distribution fit",
                    "Follows a power-law distribution (p < 0.05)",
                )
            else:
                print_result(
                    f"{centrality_type.capitalize()} distribution fit",
                    "Follows an exponential distribution (p < 0.05)",
                )
        else:
            print_result(
                f"{centrality_type.capitalize()} distribution fit",
                "Neither distribution is significantly favored (p >= 0.05)",
            )

        # Plot the distribution
        plt.figure(figsize=(8, 6))

        # Create histogram using numpy
        hist, bin_edges = np.histogram(values, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Filter out empty bins
        non_zero_indices = hist > 0
        hist_filtered = hist[non_zero_indices]
        bin_centers_filtered = bin_centers[non_zero_indices]

        if (
            len(hist_filtered) > 1
        ):  # Make sure we have at least 2 points for log-log plot
            plt.loglog(bin_centers_filtered, hist_filtered, "o", markersize=6)

            # Add power law fit line for visualization
            x_range = np.logspace(
                np.log10(min(bin_centers_filtered)),
                np.log10(max(bin_centers_filtered)),
                50,
            )
            # Scale factor for visualization (approximate)
            scale = hist_filtered[0] / (bin_centers_filtered[0] ** -alpha)
            plt.loglog(
                x_range,
                scale * x_range**-alpha,
                "r-",
                label=f"Power Law Fit (α={alpha:.2f})",
            )

            plt.xlabel(f"{centrality_type.capitalize()} (log scale)")
            plt.ylabel("Frequency (log scale)")
            plt.title(f"{centrality_type.capitalize()} Distribution (Log-Log Scale)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            filename = f"{save_dir}/{centrality_type}_powerlaw.png"
            plt.savefig(filename)

            if show_plot:
                plt.show()
            else:
                plt.close()
        else:
            print_result(
                f"{centrality_type.capitalize()} plotting",
                "Not enough data points for log-log plot",
            )

        results[centrality_type] = (alpha, R, p)

    return results


def plot_communities(G, communities, title):
    plt.figure(figsize=(12, 8))

    # Map nodes to their community
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    # Colors for each community
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
    node_colors = [colors[community_map[node]] for node in G.nodes()]

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_colors,
        with_labels=True,
        node_size=100,
        font_size=8,
        edge_color="gray",
        alpha=0.8,
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def calculate_community_stats(G, communities):
    stats = []

    for i, comm in tqdm(enumerate(communities)):
        print(i)
        # Create subgraph for this community
        subgraph = G.subgraph(comm)

        # Basic statistics
        num_nodes = len(subgraph.nodes())
        num_edges = len(subgraph.edges())

        # Handle connected and disconnected graphs differently
        if nx.is_connected(subgraph):
            diameter = nx.diameter(subgraph)
            avg_path_length = nx.average_shortest_path_length(subgraph)
        else:
            # For disconnected graphs, use largest connected component
            largest_cc = max(nx.connected_components(subgraph), key=len)
            largest_subgraph = subgraph.subgraph(largest_cc)
            diameter = nx.diameter(largest_subgraph)
            avg_path_length = nx.average_shortest_path_length(largest_subgraph)
            # Note with asterisk that this is based on largest component
            diameter = f"{diameter}*"
            avg_path_length = f"{avg_path_length:.2f}*"

        # Calculate average degree
        degrees = [d for n, d in subgraph.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0

        stats.append(
            {
                "Community": i + 1,
                "Nodes": num_nodes,
                "Edges": num_edges,
                "Diameter": diameter,
                "Avg Path Length": avg_path_length,
                "Avg Degree": f"{avg_degree:.2f}",
            }
        )

    return pd.DataFrame(stats)


# Example usage
if __name__ == "__main__":
    # Define file paths for storing the data
    DATA_DIR = "tmp"
    FILES = {
        "graph_data": os.path.join(DATA_DIR, "graph_data.pkl"),
        "centrality": os.path.join(DATA_DIR, "centrality_measures.pkl"),
        "gn_communities": os.path.join(DATA_DIR, "gn_communities.pkl"),
        "louvain_communities": os.path.join(DATA_DIR, "louvain_communities.pkl"),
        "clustering": os.path.join(DATA_DIR, "clustering.pkl"),
    }

    # Load or calculate graph data
    graph_data = load_or_calculate(
        FILES["graph_data"],
        create_nutritional_network,
        calculate_args=["Fineli_Rel20"],
        calculate_kwargs={"similarity_threshold": 0.99},
        description="graph data",
    )
    G = graph_data["graph"]

    print_task_header(3, "Visualize and plot the degree distribution")

    # Load or calculate centrality measures
    centrality_measures = load_or_calculate(
        FILES["centrality"],
        calculate_centralities,
        calculate_args=[G],
        calculate_kwargs={"use_approximation": False},
        description="centrality measures",
    )

    plot_centrality_histograms(
        {
            "degree": centrality_measures["degree"],
            "closeness": centrality_measures["closeness"],
            "betweenness": centrality_measures["betweenness"],
        },
        "Full Network",
        save_dir="plots/centrality",
    )

    print_task_header(4, "Provide the script for drawing power law distributions")

    results = analyze_centrality_power_law(
        centrality_measures, save_dir="plots/powerlaw"
    )

    print_task_header(
        5,
        "Utilize the NetworkX clustering function to calculate the clustering coefficient",
    )

    node_clustering_coefficients = load_or_calculate(
        FILES["clustering"],
        nx.clustering,
        calculate_args=[G],
        description="clustering",
    )

    clustering_values = list(node_clustering_coefficients.values())

    plt.figure(figsize=(10, 6))
    # Create the histogram with 10 bins
    plt.hist(clustering_values, bins=10, edgecolor="k", alpha=0.7)

    plt.title("Histogram of Node Clustering Coefficients")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Number of Nodes (Count)")
    plt.grid(axis="y", linestyle="--")

    plt.show()

    print_task_header(6, "Detect communities within the nutritional network")

    # # Load or calculate gn communities
    # gn_communities = load_or_calculate(
    #     FILES["gn_communities"],
    #     girvan_newman,
    #     calculate_args=[G],
    #     description="GN communities",
    #     post_process_func=lambda gen: process_girvan_newman(gen, max_communities=500),
    # )

    # Load or calculate louvain communities
    louvain_communities = load_or_calculate(
        FILES["louvain_communities"],
        louvain_communities,
        calculate_args=[G],
        calculate_kwargs={"weight": "weight"},
        description="Louvain communities",
    )

    # plot_communities(
    #     G, gn_communities, "Communities detected by Girvan-Newman algorithm"
    # )
    plot_communities(
        G, louvain_communities, "Communities detected by Louvain algorithm"
    )

    # print_result(
    #     label="Girvan-Newman Communities Statistics:",
    #     value=calculate_community_stats(G, gn_communities),
    #     indent=6,
    # )

    print_result(
        label="Louvain Communities Statistics:\n",
        value="",
        indent=0,
    )
    print("Louvain Communities Statistics:")
    print(calculate_community_stats(G, louvain_communities))

    community_nutrition, component_names = analyze_community_nutrition(
        graph_data, louvain_communities, "Fineli_Rel20"
    )

    # Display results
    print(f"Found {len(louvain_communities)} communities")

    key_display_nutrients = [
        "ENERC",  # energy
        "PROT",  # protein
        "FAT",  # fat, total
        "CHOAVL",  # carbohydrate, available
        "FIBC",  # fiber, total
    ]

    # Show summary
    print("\nCommunity Nutritional Analysis:")
    print("-" * 80)
    for _, row in community_nutrition.iterrows():
        print(f"Community {int(row['community_id'])} (Size: {int(row['size'])} foods)")
        print(f"Examples: {row['food_examples']}")

        # Display key nutrients with their proper names
        for nutrient_code in key_display_nutrients:
            if nutrient_code in row:
                nutrient_name = component_names.get(nutrient_code, nutrient_code)
                print(f"  {nutrient_name}: {row[nutrient_code]:.2f}")
        print("-" * 80)

