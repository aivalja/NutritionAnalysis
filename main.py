import random
import os
import pickle
from typing import Dict, Any, Callable, List, Optional
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import louvain_communities
import seaborn as sns
from tqdm import tqdm


SHOW_PLOT = False
default_nutrients = ["ENERC", "PROT", "FAT", "CHOAVL", "FIBC", "VITC"]


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


def create_nutritional_network(data_dir=".", similarity_threshold=0.85, weighted=False):
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
            "FAFRE",  # Total fatty acids, reduntant with other fatty acids
        ]
    ]

    # Define nutrient groups and their weights
    nutrient_groups = {
        "macronutrients": {
            "columns": [col for col in nutrient_cols if col in macronutrients],
            "weight": 0.50,  # 50% weight
        },
        "lipid_profile": {
            "columns": [col for col in nutrient_cols if col in lipid_profile],
            "weight": 0.15,  # 15% weight
        },
        "sugar_profile": {
            "columns": [col for col in nutrient_cols if col in sugar_profile],
            "weight": 0.10,  # 10% weight
        },
        "major_minerals": {
            "columns": [col for col in nutrient_cols if col in major_minerals],
            "weight": 0.10,  # 10% weight
        },
        "minor_minerals": {
            "columns": [col for col in nutrient_cols if col in minor_minerals],
            "weight": 0.05,  # 5% weight
        },
        "vitamins": {
            "columns": [col for col in nutrient_cols if col in vitamins],
            "weight": 0.08,  # 8% weight
        },
        "amino_acids": {
            "columns": [col for col in nutrient_cols if col in amino_acids],
            "weight": 0.02,  # 2% weight
        },
    }

    # Initialize scaled and weighted nutrients dataframe
    scaled_nutrients = pd.DataFrame(index=food_nutrients.index)

    # Apply scaling to the entire matrix column-wise at the beginning
    scaler = MinMaxScaler()

    # Fill missing values with 0 and scale all columns at once
    nutrients_data = food_nutrients[nutrient_cols].fillna(0)
    scaled_matrix = scaler.fit_transform(nutrients_data)

    # Convert the scaled matrix back to a DataFrame with the same columns and index
    scaled_nutrients_base = pd.DataFrame(
        scaled_matrix, columns=nutrient_cols, index=food_nutrients.index
    )

    # Create a dictionary to track which nutrients are processed
    processed_nutrients = {}

    # Process each group and its nutrients
    for group_name, group_info in nutrient_groups.items():
        columns = group_info["columns"]
        group_weight = group_info["weight"]

        if not columns:
            print(f"Warning: No columns found for group {group_name}")
            continue

        # Calculate individual nutrient weight
        individual_weight = group_weight / len(columns)

        for col in columns:
            # Ensure no duplicate processing of nutrients
            if col in processed_nutrients:
                print(
                    f"Warning: Nutrient {col} already processed in group {processed_nutrients[col]}"
                )
                continue

            processed_nutrients[col] = group_name

            # Use the pre-scaled values and apply weight
            scaled_nutrients[col] = scaled_nutrients_base[col] * individual_weight

    # Check for unprocessed nutrients and apply a small default weight
    default_weight = 0.001
    unprocessed = [col for col in nutrient_cols if col not in processed_nutrients]
    if unprocessed:
        print(
            f"Warning: {len(unprocessed)} nutrients ({unprocessed}) were not assigned to any group. Applying default weight."
        )
        for col in unprocessed:
            # Use the pre-scaled values and apply default weight
            scaled_nutrients[col] = scaled_nutrients_base[col] * default_weight

    # Calculate cosine similarity on all weighted nutrients at once
    similarity_matrix = cosine_similarity(scaled_nutrients)

    print_task_header(2, "Generate a nutritional network graph")

    # Create a new graph
    G = nx.Graph()

    # Add nodes (food items)
    for i, (food_id, food_name) in enumerate(zip(food_ids, food_names)):
        G.add_node(food_id, name=food_name)

    if weighted:
        # Add edges based on similarity threshold
        for i, food_id_i in enumerate(food_ids):
            for j, food_id_j in enumerate(food_ids[i + 1 :], start=i + 1):
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    G.add_edge(food_id_i, food_id_j)
    else:
        # Add edges based on similarity with weights
        for i, food_id_i in enumerate(food_ids):
            for j, food_id_j in enumerate(food_ids[i + 1 :], start=i + 1):
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    # Scale similarity from [similarity_threshold, 1] to [0.01, 1]
                    scaled_weight = (
                        0.01
                        + (
                            (similarity - similarity_threshold)
                            / (1.0 - similarity_threshold)
                        )
                        * 0.99
                    )
                    G.add_edge(food_id_i, food_id_j, weight=scaled_weight)

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


def analyze_community_nutrition(
    graph_data, communities, key_nutrients=default_nutrients
):
    """
    Analyzes the nutritional composition of each community

    Args:
        graph_data: Dictionary containing graph, food mapping, and nutritional data
        communities: List of communities (each community is a set of node IDs)

    Returns:
        DataFrame containing average nutritional values for each community
    """
    food_nutrients = graph_data["food_nutrients"]

    # Fill any remaining NaN values with 0 to ensure calculations work correctly
    food_nutrients = food_nutrients.fillna(0)

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

    return result_df, component_names


def create_community_summary_table(
    community_nutrition, component_names, key_nutrients=default_nutrients
):
    """
    Creates a summary table of key nutritional values across communities.

    Args:
        community_nutrition: DataFrame with community nutritional data
        component_names: Dictionary mapping component codes to readable names
        key_nutrients: List of nutrient codes to include

    Returns:
        Formatted summary DataFrame
    """

    # Filter nutrients that exist in our datakey_nutrients
    available_nutrients = [n for n in key_nutrients if n in community_nutrition.columns]

    # Create a new DataFrame for the summary
    summary_data = []

    for _, row in community_nutrition.iterrows():
        community_data = {
            "Community": int(row["community_id"]),
            "Size": int(row["size"]),
            "Examples": row["food_examples"],
        }

        # Add nutrient values with proper names
        for nutrient in available_nutrients:
            nutrient_name = component_names.get(nutrient, nutrient)
            community_data[nutrient_name] = row[nutrient]

        summary_data.append(community_data)

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def visualize_community_differences(
    community_nutrition,
    component_names,
    key_nutrients=default_nutrients,
):
    """
    Creates visualizations comparing nutritional differences between communities.

    Args:
        community_nutrition: DataFrame with community nutritional data
        component_names: Dictionary mapping component codes to readable names
        key_nutrients: List of nutrient codes to include
    """

    # Filter to nutrients that exist in our data
    available_nutrients = [n for n in key_nutrients if n in community_nutrition.columns]

    # Get proper names for nutrients
    nutrient_names = [component_names.get(n, n) for n in available_nutrients]

    # Create a copy of the data for plotting
    plot_data = community_nutrition.copy()

    # Normalize the data for each nutrient to range from 0 to 1
    for nutrient in available_nutrients:
        min_val = plot_data[nutrient].min()
        max_val = plot_data[nutrient].max()
        if max_val != min_val:  # Avoid division by zero
            plot_data[nutrient] = (plot_data[nutrient] - min_val) / (max_val - min_val)
        else:
            plot_data[nutrient] = 0  # If all values are the same, set to 0

    # 1. BAR CHART: Compare key nutrients across communities
    plt.figure(figsize=(14, 8))

    # Set up the number of bars and positions
    n_communities = len(plot_data)
    n_nutrients = len(available_nutrients)
    bar_width = 0.8 / n_communities

    # Create positions for each group of bars
    positions = np.arange(n_nutrients)

    # Plot bars for each community
    for i, (_, row) in enumerate(plot_data.iterrows()):
        community_id = int(row["community_id"])
        offset = (i - n_communities / 2) * bar_width + bar_width / 2

        # Get normalized values for this community
        values = [row[nutrient] for nutrient in available_nutrients]

        # Plot bars
        plt.bar(
            positions + offset,
            values,
            width=bar_width,
            label=f"Community {community_id} (n={int(row['size'])})",
        )

    # Set labels and title
    plt.xlabel("Nutrients")
    plt.ylabel("Normalized Average Value (0 to 1)")
    plt.title("Normalized Comparison of Key Nutrients Across Food Communities")
    plt.xticks(positions, nutrient_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()

    # 2. HEATMAP: Show relative nutrient composition
    plt.figure(figsize=(12, 8))

    # Prepare data for heatmap
    heatmap_data = plot_data[available_nutrients].copy()

    # Normalize data (scale each nutrient relative to its maximum across communities)
    for col in heatmap_data.columns:
        max_val = heatmap_data[col].max()
        if max_val > 0:  # Avoid division by zero
            heatmap_data[col] = heatmap_data[col] / max_val

    # Add community labels
    heatmap_data["Community"] = plot_data["community_id"].apply(
        lambda x: f"Community {int(x)}"
    )
    heatmap_data = heatmap_data.set_index("Community")

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        xticklabels=[component_names.get(n, n) for n in available_nutrients],
    )
    plt.title("Relative Nutrient Composition by Community (Normalized to Max Value)")
    plt.tight_layout()
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()

    # 3. RADAR CHART: Profile of each community
    # Only include if number of communities is manageable (≤ 6)
    if n_communities <= 6:
        # Prepare data for radar chart
        radar_data = plot_data[available_nutrients].copy()

        # Normalize data for radar chart
        for col in radar_data.columns:
            max_val = radar_data[col].max()
            if max_val > 0:
                radar_data[col] = radar_data[col] / max_val

        # Number of variables
        N = len(available_nutrients)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Initialize the figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Add variable labels around the chart
        plt.xticks(
            angles[:-1],
            [component_names.get(n, n) for n in available_nutrients],
            color="grey",
            size=10,
        )

        # Plot each community
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            community_id = int(row["community_id"])

            # Get normalized values for this community and close the loop
            values = radar_data.iloc[i].values.tolist()
            values += values[:1]

            # Plot values
            ax.plot(
                angles,
                values,
                linewidth=2,
                label=f"Community {community_id} (n={int(row['size'])})",
            )
            ax.fill(angles, values, alpha=0.1)

        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.title("Nutritional Profile of Food Communities (Normalized)")
        if SHOW_PLOT:
            plt.show()
        else:
            plt.close()


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
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()


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
        if SHOW_PLOT:
            plt.show()
        else:
            plt.close()


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
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()


def calculate_community_stats(G, communities):
    stats = []

    for i, comm in enumerate(communities):
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


def find_top_similar_foods_in_communities(graph_data, communities, top_n=10):
    """
    Identifies the top-N individual food items within each community based on average nutritional similarity
    to other foods in the same community.

    Args:
        graph_data: Dictionary containing graph, food mapping, and similarity matrix
        communities: List of communities (each community is a set of node IDs)
        top_n: Number of top similar foods to return per community

    Returns:
        Dictionary with community ID as key and list of top similar foods as value
    """
    food_ids = list(graph_data["food_mapping"].keys())
    similarity_matrix = graph_data["similarity_matrix"]
    food_mapping = graph_data["food_mapping"]

    # Dictionary to store results
    community_top_foods = {}

    # Process each community
    for comm_id, community in enumerate(communities):
        if len(community) < 2:
            community_top_foods[comm_id] = []
            continue

        # Get indices of foods in this community
        comm_indices = [
            food_ids.index(food_id) for food_id in community if food_id in food_ids
        ]

        # Create list to store foods with their average similarity
        food_avg_similarities = []

        # Calculate average similarity for each food to others in community
        for i in comm_indices:
            food_id = food_ids[i]
            # Get similarities to all other foods in community
            similarities = [similarity_matrix[i][j] for j in comm_indices if j != i]
            avg_similarity = (
                sum(similarities) / len(similarities) if similarities else 0
            )
            food_avg_similarities.append(
                {
                    "food_id": food_id,
                    "food_name": food_mapping[food_id],
                    "avg_similarity": avg_similarity,
                }
            )

        # Sort by average similarity and take top N
        food_avg_similarities.sort(key=lambda x: x["avg_similarity"], reverse=True)
        community_top_foods[comm_id] = food_avg_similarities[:top_n]

    return community_top_foods


def analyze_top_food_characteristics(
    graph_data, community_top_foods, component_names, key_nutrients=default_nutrients
):
    """
    Analyzes characteristics of top similar food items in each community to identify trends.
    Uses normalized values to determine dominant characteristics.

    Args:
        graph_data: Dictionary containing graph, food mapping, and nutritional data
        community_top_foods: Dictionary with community ID and list of top similar foods
        component_names: Dictionary mapping component codes to readable names

    Returns:
        Dictionary with community ID as key and analysis summary as value
    """
    food_nutrients = graph_data["food_nutrients"]

    # Calculate global averages for normalization
    global_avg = food_nutrients[key_nutrients].mean()

    analysis_results = {}

    for comm_id, top_foods in community_top_foods.items():
        if not top_foods:
            analysis_results[comm_id] = {
                "summary": "Community too small for analysis",
                "trends": [],
            }
            continue

        # Get food IDs of top similar foods
        food_ids_top = [food["food_id"] for food in top_foods]

        # Get nutritional data for these foods
        relevant_foods = food_nutrients[food_nutrients["FOODID"].isin(food_ids_top)]

        # Calculate average values for key nutrients
        avg_nutrients = relevant_foods[key_nutrients].mean()

        # Identify dominant characteristics by relative value compared to global average
        nutrient_values = []
        for code in key_nutrients:
            if not pd.isna(avg_nutrients[code]) and global_avg[code] > 0:
                relative_value = avg_nutrients[code] / global_avg[code]
                nutrient_values.append(
                    (
                        component_names.get(code, code),
                        relative_value,
                        avg_nutrients[code],
                    )
                )

        # Sort by relative value (how much higher/lower than global average)
        nutrient_values.sort(key=lambda x: x[1], reverse=True)
        dominant_traits = [
            f"{name}: {value:.2f} ({rel:.2f}x avg)"
            for name, rel, value in nutrient_values[:2]
        ]

        # Look for common patterns in food names (simple keyword analysis)
        food_names = relevant_foods["FOODNAME"].str.lower().tolist()
        common_words = []
        if food_names:
            word_counts = {}
            for name in food_names:
                words = name.split()
                for word in words:
                    if len(word) > 3:  # Ignore very short words
                        word_counts[word] = word_counts.get(word, 0) + 1
            common_words = [
                word
                for word, count in sorted(
                    word_counts.items(), key=lambda x: x[1], reverse=True
                )
                if count > len(food_names) * 0.1
            ][:3]

        analysis_results[comm_id] = {
            "summary": f"Top similar foods characterized by {', '.join(dominant_traits)}",
            "common_words": common_words,
            "trends": [
                f"Average similarity score of top foods: {sum(f['avg_similarity'] for f in top_foods) / len(top_foods):.3f}",
                f"Common terms in names: {', '.join(common_words) if common_words else 'No common terms'}",
            ],
            "foods_count": len(top_foods),
        }

    return analysis_results


def initialize_data(
    data_dir="tmp", dataset="Fineli_Rel20", similarity_threshold=0.80, weighted=False
):
    """Initialize data directory and create file paths for storing data."""
    os.makedirs(data_dir, exist_ok=True)
    files = {
        "graph_data": os.path.join(data_dir, "graph_data.pkl"),
        "centrality": os.path.join(data_dir, "centrality_measures.pkl"),
        "gn_communities": os.path.join(data_dir, "gn_communities.pkl"),
        "louvain_communities": os.path.join(data_dir, "louvain_communities.pkl"),
        "clustering": os.path.join(data_dir, "clustering.pkl"),
    }

    # Load or calculate graph data
    graph_data = load_or_calculate(
        files["graph_data"],
        create_nutritional_network,
        calculate_args=[dataset],
        calculate_kwargs={
            "similarity_threshold": similarity_threshold,
            "weighted": weighted,
        },
        description="graph data",
    )

    return files, graph_data


def analyze_centrality(G, files, show_plot=False):
    """Analyze and visualize centrality measures."""
    print_task_header(3, "Visualize and plot the degree distribution")
    approximate = True

    centrality_measures = load_or_calculate(
        files["centrality"],
        calculate_centralities,
        calculate_args=[G],
        calculate_kwargs={"use_approximation": approximate},
        description="centrality measures",
    )

    if approximate:
        plot_centrality_histograms(
            {
                "degree": centrality_measures["degree"],
                "closeness": centrality_measures["closeness"],
                "betweenness": centrality_measures["betweenness"],
            },
            "Approximation",
            save_dir="plots/centrality",
        )
    else:
        plot_centrality_histograms(
            {
                "degree": centrality_measures["degree"],
                "closeness": centrality_measures["closeness"],
                "betweenness": centrality_measures["betweenness"],
            },
            "full",
            save_dir="plots/centrality",
        )

    print_task_header(4, "Provide the script for drawing power law distributions")
    results = analyze_centrality_power_law(
        centrality_measures, show_plot=show_plot, save_dir="plots/powerlaw"
    )

    return centrality_measures, results


def analyze_clustering(G, files, show_plot=False, output_dir="."):
    """Analyze clustering coefficients in the network."""
    print_task_header(
        5,
        "Utilize the NetworkX clustering function to calculate the clustering coefficient",
    )

    node_clustering_coefficients = load_or_calculate(
        files["clustering"],
        nx.clustering,
        calculate_args=[G],
        description="clustering",
    )

    clustering_values = list(node_clustering_coefficients.values())

    plt.figure(figsize=(10, 6))
    plt.hist(clustering_values, bins=10, edgecolor="k", alpha=0.7)
    plt.title("Histogram of Node Clustering Coefficients")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Number of Nodes (Count)")
    plt.grid(axis="y", linestyle="--")

    filename = f"{output_dir}/clustering_histogram.png"
    plt.savefig(filename)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return node_clustering_coefficients


def detect_communities(G, files, show_plot=False):
    """Detect communities using Girvan-Newman and Louvain algorithms."""
    print_task_header(6, "Detect communities within the nutritional network")

    # Load or calculate Girvan-Newman communities
    gn_communities = load_or_calculate(
        files["gn_communities"],
        girvan_newman,
        calculate_args=[G],
        description="GN communities",
        post_process_func=lambda gen: process_girvan_newman(gen, max_communities=500),
    )

    # Load or calculate Louvain communities
    louvain_comms = load_or_calculate(
        files["louvain_communities"],
        louvain_communities,
        calculate_args=[G],
        calculate_kwargs={"weight": "weight"},
        description="Louvain communities",
    )

    # Visualize communities
    if show_plot:
        plot_communities(
            G, gn_communities, "Communities detected by Girvan-Newman algorithm"
        )
        plot_communities(G, louvain_comms, "Communities detected by Louvain algorithm")

    # Display community statistics
    print_result(
        label="Girvan-Newman Communities Statistics:",
        value=calculate_community_stats(G, gn_communities),
        indent=6,
    )

    print("Louvain Communities Statistics:")
    print(calculate_community_stats(G, louvain_comms))

    return gn_communities, louvain_comms


def analyze_nutritional_composition(
    graph_data, communities, dataset, key_nutrients=default_nutrients
):
    """Analyze the nutritional composition of communities."""
    print_task_header(7, "Analyze Community Nutritional Composition")

    # Load component name mappings
    component_names = load_component_names(dataset)

    community_nutrition = analyze_community_nutrition(
        graph_data, communities, key_nutrients
    )

    # Generate summary table
    print("\nSummary Table of Community Nutritional Differences:")
    summary_table = create_community_summary_table(
        community_nutrition, component_names, key_nutrients
    )
    print(summary_table)

    # Visualize differences
    print("\nGenerating visualizations to compare communities...")
    visualize_community_differences(community_nutrition, component_names, key_nutrients)

    return community_nutrition, component_names


def analyze_top_similar_foods(
    graph_data, communities, component_names, key_nutrients=default_nutrients
):
    """Find and analyze the top similar foods within communities."""
    community_top_foods = find_top_similar_foods_in_communities(graph_data, communities)

    print_task_header(8, "Identify the top-10 most similar food items")
    top_food_analysis = analyze_top_food_characteristics(
        graph_data, community_top_foods, component_names, key_nutrients
    )

    print(f"Found {len(communities)} communities")

    return community_top_foods, top_food_analysis


def display_results(
    community_nutrition, component_names, community_top_foods, top_food_analysis
):
    """Display the analysis results in a readable format."""
    # Display key nutrients for communities
    key_display_nutrients = ["ENERC", "PROT", "FAT", "CHOAVL", "FIBC"]
    print("\nCommunity Nutritional Analysis:")
    print("-" * 80)

    for _, row in community_nutrition.iterrows():
        comm_id = int(row["community_id"])
        print(f"Community {comm_id} (Size: {int(row['size'])} foods)")
        print(f"Examples: {row['food_examples']}")

        for nutrient_code in key_display_nutrients:
            if nutrient_code in row:
                nutrient_name = component_names.get(nutrient_code, nutrient_code)
                print(f"  {nutrient_name}: {row[nutrient_code]:.2f}")
        print("-" * 80)

    # Display top foods analysis
    print("\nTop Similar Foods Analysis Within Communities (Top 10 Foods):")
    print("-" * 80)

    for comm_id, analysis in top_food_analysis.items():
        print(f"Community {comm_id}")
        print(f"  {analysis['summary']}")

        for trend in analysis["trends"]:
            print(f"  {trend}")

        # Show examples
        if comm_id in community_top_foods and community_top_foods[comm_id]:
            print("  Example top similar foods:")
            for food in community_top_foods[comm_id][:3]:
                print(
                    f"    - {food['food_name']} (Avg Similarity: {food['avg_similarity']:.3f})"
                )
        print("-" * 80)


def analyze_nutritional_assortativity(
    graph_data,
    communities=None,
    nutrient_groups=None,
    top_nutrients=10,
    num_bins=5,
    visualize=True,
):
    """
    Analyzes the assortativity of a nutritional network including community structure.

    Args:
        graph_data: Dictionary containing graph and related data from create_nutritional_network()
        communities: Dictionary mapping node IDs to community IDs, or a list of node sets
        nutrient_groups: Dictionary mapping group names to lists of nutrient columns
        top_nutrients: Number of top nutrients to analyze by variance if not using groups
        num_bins: Number of bins to discretize continuous nutrient values
        visualize: Whether to create and show visualizations

    Returns:
        Dictionary with assortativity results
    """

    print("Analyzing assortativity patterns in the nutritional network...")

    # Extract graph and data
    G = graph_data["graph"]
    food_nutrients = graph_data["food_nutrients"]

    # Define nutrient groups if not provided
    if nutrient_groups is None:
        try:
            # Try to use predefined groups from the global namespace
            predefined_groups = {
                "macronutrients": macronutrients,
                "lipid_profile": lipid_profile,
                "sugar_profile": sugar_profile,
                "major_minerals": major_minerals,
                "minor_minerals": minor_minerals,
                "vitamins": vitamins,
                "amino_acids": amino_acids,
            }

            # Check which groups have at least one column in the data
            nutrient_groups = {}
            for group_name, group_list in predefined_groups.items():
                available_nutrients = [
                    col for col in group_list if col in food_nutrients.columns
                ]
                if available_nutrients:
                    nutrient_groups[group_name] = available_nutrients

        except NameError:
            # If predefined groups don't exist, we'll select by variance
            nutrient_groups = {}

    # If no nutrient groups are available or defined, select top nutrients by variance
    if not nutrient_groups:
        print("ERROR: no nutrient groups")
        return

    results = {}

    # 1. Calculate degree assortativity
    print("Calculating degree assortativity...")
    degree_assortativity = nx.degree_assortativity_coefficient(G)
    results["degree_assortativity"] = degree_assortativity

    # 2. Process community data if provided
    if communities:
        print("Processing community structure...")

        # Convert communities to node-to-community dictionary if necessary
        if isinstance(communities, list):
            # Convert list of sets/lists to dictionary
            community_dict = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_dict[node] = i
        elif isinstance(communities, dict):
            community_dict = communities
        else:
            raise ValueError(
                "Communities must be provided as a dictionary or list of node sets"
            )

        # Set community as node attribute
        nx.set_node_attributes(G, community_dict, name="community")

        # Calculate community assortativity
        community_assortativity = nx.attribute_assortativity_coefficient(G, "community")
        results["community_assortativity"] = community_assortativity
        print(f"Community assortativity coefficient: {community_assortativity:.4f}")

        # Generate community-based analysis
        community_nutrients = defaultdict(list)

        # Group nodes by community
        community_nodes = defaultdict(list)
        for node, comm_id in community_dict.items():
            community_nodes[comm_id].append(node)

        # Calculate average nutrient values per community
        community_avg_nutrients = {}

        for comm_id, nodes in community_nodes.items():
            # Get food IDs in this community
            food_ids = [
                node for node in nodes if node in food_nutrients["FOODID"].values
            ]

            if food_ids:
                # Filter nutrients data for this community
                community_foods = food_nutrients[
                    food_nutrients["FOODID"].isin(food_ids)
                ]

                # Calculate average for each nutrient group
                group_averages = {}

                for group_name, nutrient_list in nutrient_groups.items():
                    available_nutrients = [
                        n for n in nutrient_list if n in food_nutrients.columns
                    ]

                    if available_nutrients:
                        # Calculate mean for each nutrient, ignoring NaN values
                        nutrient_means = community_foods[available_nutrients].mean(
                            skipna=True
                        )
                        group_averages[group_name] = nutrient_means.to_dict()

                community_avg_nutrients[comm_id] = group_averages

        results["community_nutrients"] = community_avg_nutrients

    # 3. Calculate attribute assortativity for selected nutrients
    print("Calculating nutrient attribute assortativity...")
    attribute_results = {}

    for group_name, nutrient_list in nutrient_groups.items():
        print(f"Processing {group_name} group...")
        group_results = {}

        for attr in nutrient_list:
            if attr in food_nutrients.columns:
                # Create a mapping of node IDs to nutrient values
                attr_dict = {}

                for node in G.nodes():
                    food_row = food_nutrients[food_nutrients["FOODID"] == node]
                    if not food_row.empty and not pd.isna(food_row[attr].values[0]):
                        attr_dict[node] = float(food_row[attr].values[0])

                if len(attr_dict) > 1:  # Need at least 2 values to create bins
                    # Discretize continuous values into bins
                    values = list(attr_dict.values())
                    if min(values) == max(values):
                        print(f"  Skipping {attr}: All values are identical")
                        continue

                    bins = np.linspace(min(values), max(values), num_bins + 1)

                    # Create binned attribute dictionary
                    binned_attr_dict = {}
                    for node, value in attr_dict.items():
                        bin_index = np.digitize(value, bins)
                        binned_attr_dict[node] = int(bin_index)

                    # Set the node attribute in the graph
                    nx.set_node_attributes(G, binned_attr_dict, name=f"{attr}_bin")

                    # Calculate assortativity for this attribute
                    try:
                        attr_assortativity = nx.attribute_assortativity_coefficient(
                            G, f"{attr}_bin"
                        )
                        group_results[attr] = attr_assortativity
                        print(f"  {attr}: {attr_assortativity:.4f}")
                    except Exception as e:
                        print(
                            f"  Could not calculate assortativity for {attr}: {str(e)}"
                        )

        if group_results:
            attribute_results[group_name] = group_results

    results["attribute_assortativity"] = attribute_results

    # 4. Create visualizations
    if visualize:
        print("\nCreating visualizations...")

        # Create figure for assortativity coefficients
        plt.figure(figsize=(10, 6))

        # Setup bars for degree and community (if available) assortativity
        labels = ["Degree Assortativity"]
        values = [degree_assortativity]

        if communities:
            labels.append("Community Assortativity")
            values.append(community_assortativity)

        plt.bar(labels, values, color=["skyblue", "lightgreen"][: len(labels)])
        plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
        plt.ylim(-1.1, 1.1)
        plt.ylabel("Assortativity Coefficient")
        plt.title("Assortativity in Nutritional Network")

        # Add interpretation text
        for i, (label, value) in enumerate(zip(labels, values)):
            if value > 0:
                plt.text(
                    i,
                    value / 2,
                    "Assortative\n(similar connect\nto similar)",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.8),
                )
            else:
                plt.text(
                    i,
                    value / 2,
                    "Disassortative\n(dissimilar connect\nto dissimilar)",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.8),
                )

        plt.tight_layout()
        plt.show()

        # Create heatmap of assortativity coefficients by nutrient group
        if attribute_results:
            # Prepare data for heatmap
            heatmap_data = []

            for group_name, group_results in attribute_results.items():
                for nutrient, coeff in group_results.items():
                    heatmap_data.append(
                        {
                            "Nutrient Group": group_name,
                            "Nutrient": nutrient,
                            "Assortativity": coeff,
                        }
                    )

            if heatmap_data:
                df_heatmap = pd.DataFrame(heatmap_data)

                # Create pivot table for the heatmap
                pivot_data = df_heatmap.pivot_table(
                    index="Nutrient Group",
                    columns="Nutrient",
                    values="Assortativity",
                    aggfunc="first",
                )

                # Create heatmap
                plt.figure(figsize=(14, 8))
                sns.heatmap(
                    pivot_data,
                    cmap="coolwarm",
                    center=0,
                    annot=True,
                    fmt=".2f",
                    linewidths=0.5,
                    cbar_kws={"label": "Assortativity Coefficient"},
                )
                plt.title("Assortativity by Nutrient")
                plt.tight_layout()
                plt.show()

                # Bar plot showing average assortativity by nutrient group
                group_avg = (
                    df_heatmap.groupby("Nutrient Group")["Assortativity"]
                    .mean()
                    .sort_values()
                )

                plt.figure(figsize=(10, 6))
                colors = ["lightgreen" if x >= 0 else "lightcoral" for x in group_avg]
                bars = plt.bar(group_avg.index, group_avg, color=colors)
                plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
                plt.ylabel("Average Assortativity Coefficient")
                plt.title("Average Nutrient Group Assortativity")
                plt.xticks(rotation=45, ha="right")

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        0.01 if height < 0 else height + 0.01,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom" if height >= 0 else "top",
                    )

                plt.tight_layout()
                plt.show()

        # Community-specific visualizations
        if communities and len(community_avg_nutrients) > 0:
            print("Creating community-based visualizations...")

            # Prepare data for PCA visualization of communities
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Collect nutrient data for all communities
            community_nutrient_data = []

            # Select common nutrients available across communities for comparison
            common_nutrients = set()

            # First, identify common nutrients across all communities
            for comm_id, group_data in community_avg_nutrients.items():
                for group_name, nutrients in group_data.items():
                    if common_nutrients:
                        common_nutrients = common_nutrients.intersection(
                            set(nutrients.keys())
                        )
                    else:
                        common_nutrients = set(nutrients.keys())

            # If we have common nutrients, prepare data for PCA
            if common_nutrients:
                common_nutrients = list(common_nutrients)

                for comm_id, group_data in community_avg_nutrients.items():
                    # Gather all nutrient values for this community
                    nutrient_values = {}

                    for group_name, nutrients in group_data.items():
                        for nutrient in common_nutrients:
                            if nutrient in nutrients:
                                nutrient_values[nutrient] = nutrients[nutrient]

                    if len(nutrient_values) == len(common_nutrients):
                        row = {"community": comm_id}
                        row.update(nutrient_values)
                        community_nutrient_data.append(row)

                if (
                    len(community_nutrient_data) >= 3
                ):  # Need at least 3 communities for meaningful PCA
                    df_community = pd.DataFrame(community_nutrient_data)

                    # Extract features for PCA
                    X = df_community[common_nutrients].values

                    # Standardize the data
                    X_std = StandardScaler().fit_transform(X)

                    # Apply PCA with 2 components
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_std)

                    # Create PCA plot
                    plt.figure(figsize=(10, 8))
                    plt.scatter(
                        X_pca[:, 0],
                        X_pca[:, 1],
                        c=df_community["community"],
                        cmap="viridis",
                        s=100,
                        alpha=0.8,
                    )

                    # Add community labels
                    for i, comm_id in enumerate(df_community["community"]):
                        plt.annotate(
                            f"Community {comm_id}",
                            (X_pca[i, 0], X_pca[i, 1]),
                            xytext=(5, 5),
                            textcoords="offset points",
                        )

                    # Add feature vectors
                    feature_vectors = pca.components_.T * np.sqrt(
                        pca.explained_variance_
                    )

                    # Only show the most important feature vectors
                    importance = np.sum(np.abs(feature_vectors), axis=1)
                    top_idx = np.argsort(importance)[-min(10, len(common_nutrients)) :]

                    for i in top_idx:
                        plt.arrow(
                            0,
                            0,
                            feature_vectors[i, 0],
                            feature_vectors[i, 1],
                            color="r",
                            alpha=0.5,
                            head_width=0.05,
                        )
                        plt.text(
                            feature_vectors[i, 0] * 1.1,
                            feature_vectors[i, 1] * 1.1,
                            common_nutrients[i],
                            color="r",
                            ha="center",
                            va="center",
                        )

                    plt.title("PCA of Community Nutrient Profiles")
                    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
                    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
                    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
                    plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.show()

                    # Create heatmap comparing nutrient profiles across communities
                    community_nutrient_matrix = df_community.set_index("community")[
                        common_nutrients
                    ]

                    # Standardize the data for heatmap
                    community_nutrient_std = pd.DataFrame(
                        StandardScaler().fit_transform(community_nutrient_matrix),
                        index=community_nutrient_matrix.index,
                        columns=community_nutrient_matrix.columns,
                    )

                    plt.figure(figsize=(12, len(community_nutrient_std) * 0.8))
                    sns.heatmap(
                        community_nutrient_std,
                        cmap="coolwarm",
                        center=0,
                        annot=False,
                        xticklabels=1,
                        yticklabels=1,
                    )
                    plt.title("Standardized Nutrient Profiles by Community")
                    plt.tight_layout()
                    plt.show()
            else:
                print("Warning: No common nutrients found across communities.")

    # 5. Print interpretation and summary
    print("\n===== ASSORTATIVITY ANALYSIS SUMMARY =====")
    print(f"Degree assortativity coefficient: {degree_assortativity:.4f}")

    if degree_assortativity > 0.3:
        print("The network shows strong assortative mixing by degree.")
        print(
            "Foods with many connections tend to connect to other foods with many connections."
        )
    elif degree_assortativity > 0:
        print("The network shows weak assortative mixing by degree.")
    elif degree_assortativity < -0.3:
        print("The network shows strong disassortative mixing by degree.")
        print(
            "Foods with many connections tend to connect to foods with few connections."
        )
    elif degree_assortativity < 0:
        print("The network shows weak disassortative mixing by degree.")
    else:
        print("The network shows no clear assortativity pattern by degree.")

    if communities:
        print(
            f"\nCommunity assortativity coefficient: {results['community_assortativity']:.4f}"
        )

        if results["community_assortativity"] > 0.5:
            print("The network shows STRONG community-based assortativity.")
            print("Nodes are strongly connected within their communities.")
        elif results["community_assortativity"] > 0:
            print("The network shows WEAK community-based assortativity.")
        else:
            print("The network shows DISASSORTATIVE community structure.")
            print("Nodes tend to connect to nodes from different communities.")

    if attribute_results:
        print("\nNutrient attribute assortativity summary:")

        # Find the most assortative and disassortative nutrients overall
        all_nutrients = {}
        for group_data in attribute_results.values():
            all_nutrients.update(group_data)

        if all_nutrients:
            max_assort = max(all_nutrients.items(), key=lambda x: x[1])
            min_assort = min(all_nutrients.items(), key=lambda x: x[1])

            print(f"Most assortative nutrient: {max_assort[0]} ({max_assort[1]:.4f})")
            print(
                f"Most disassortative nutrient: {min_assort[0]} ({min_assort[1]:.4f})"
            )

            # Calculate overall average
            avg_assort = np.mean(list(all_nutrients.values()))
            print(f"Overall average assortativity: {avg_assort:.4f}")

            if avg_assort > 0.2:
                print("The nutritional network is generally ASSORTATIVE:")
                print(
                    "Foods with similar nutrient profiles tend to be connected to each other."
                )
            elif avg_assort < -0.2:
                print("The nutritional network is generally DISASSORTATIVE:")
                print(
                    "Foods with different nutrient profiles tend to be connected to each other."
                )
            else:
                print(
                    "The nutritional network shows WEAK mixing patterns by nutrient content."
                )

    return results


def run_nutritional_network_analysis(
    data_dir="tmp",
    dataset="Fineli_Rel20",
    similarity_threshold=0.80,
    show_plot=False,
    weighted=False,
    key_nutrients=default_nutrients,
):
    """Main function to run the complete nutritional network analysis."""
    # Initialize data
    files, graph_data = initialize_data(
        data_dir, dataset, similarity_threshold, weighted
    )
    G = graph_data["graph"]

    # Analyze centrality
    centrality_measures, _ = analyze_centrality(G, files, show_plot)

    # Analyze clustering
    analyze_clustering(G, files, show_plot, data_dir)

    # Detect communities
    gn_communities, louvain_comms = detect_communities(G, files)

    # Analyze nutritional composition
    community_nutrition, component_names = analyze_nutritional_composition(
        graph_data, louvain_comms, dataset, key_nutrients
    )

    # Analyze similar foods
    community_top_foods, top_food_analysis = analyze_top_similar_foods(
        graph_data, louvain_comms, component_names
    )

    # Display results
    display_results(
        community_nutrition, component_names, community_top_foods, top_food_analysis
    )

    # Analyze network assortativity
    assortativity_results = analyze_nutritional_assortativity(
        graph_data=graph_data,
        communities=louvain_comms,
        visualize=True,
    )


    return {
        "graph": G,
        "centrality_measures": centrality_measures,
        "communities": louvain_comms,
        "community_nutrition": community_nutrition,
        "assortativity_results": assortativity_results,
    }


# Key nutritional attributes to analyze
key_nutrients = [
    "ALC",
    "CA",
    "CAROTENS",
    "CHOAVL",
    "CHOLE",
    "ENERC",
    "F18D2CN6",
    "F18D3N3",
    "F20D5N3",
    "F22D6N3",
    "FAFRE",
    "FAMCIS",
    "FAPU",
    "FAPUN3",
    "FAPUN6",
    "FASAT",
    "FAT",
    "FATRN",
    "FE",
    "FIBC",
    "FIBINS",
    "FOL",
    "FRUS",
    "GALS",
    "GLUS",
    "ID",
    "K",
    "LACS",
    "MALS",
    "MG",
    "NACL",
    "NIA",
    "NIAEQ",
    "OA",
    "P",
    "PROT",
    "PSACNCS",
    "RIBF",
    "SE",
    "STARCH",
    "STERT",
    "SUCS",
    "SUGAR",
    "SUGOH",
    "THIA",
    "TRP",
    "VITA",
    "VITB12",
    "VITC",
    "VITD",
    "VITE",
    "VITK",
    "VITPYRID",
    "ZN",
]

# Macronutrients
macronutrients = [
    "ENERC",  # energy, calculated (kJ)
    "FAT",  # fat, total (g)
    "CHOAVL",  # carbohydrate, available (g)
    "PROT",  # protein, total (g)
    "ALC",  # alcohol (g)
    "FIBC",  # fibre, total (g)
    "FIBINS",  # fibre, insoluble (g)
    "PSACNCS",  # polysaccharide, water-soluble non-cellulose (g)
    "STARCH",  # starch, total (g)
    "SUGAR",  # sugars, total (g)
]

# Lipid Profile
lipid_profile = [
    "FASAT",  # fatty acids, total saturated (g)
    "FAMCIS",  # fatty acids, total monounsaturated cis (g)
    "FAPU",  # fatty acids, total polyunsaturated (g)
    "FATRN",  # fatty acids, total trans (g)
    "FAPUN3",  # fatty acids, total n-3 polyunsaturated (g)
    "FAPUN6",  # fatty acids, total n-6 polyunsaturated (g)
    "F18D2CN6",  # fatty acid 18:2 cis, cis n-6 (linoleic acid)
    "F18D3N3",  # fatty acid 18:3 n-3 (alpha-linolenic acid)
    "F20D5N3",  # fatty acid 20:5 n-3 (EPA)
    "F22D6N3",  # fatty acid 22:6 n-3 (DHA)
    "CHOLE",  # cholesterol (mg)
    "STERT",  # sterols, total (mg)
]

# Sugar Profile
sugar_profile = [
    "FRUS",  # fructose (g)
    "GALS",  # galactose (g)
    "GLUS",  # glucose (g)
    "LACS",  # lactose (g)
    "MALS",  # maltose (g)
    "SUCS",  # sucrose (g)
    "SUGOH",  # sugar alcohols (g)
    "OA",  # organic acids, total (g)
]

# Major Minerals
major_minerals = [
    "CA",  # calcium (mg)
    "K",  # potassium (mg)
    "MG",  # magnesium (mg)
    "NACL",  # sodium/salt (mg)
    "P",  # phosphorus (mg)
]

# Minor Minerals
minor_minerals = [
    "FE",  # iron (mg)
    "ID",  # iodine (ug)
    "SE",  # selenium (ug)
    "ZN",  # zinc (mg)
]

# Vitamins
vitamins = [
    "VITA",  # Vitamin A (RAE)
    "THIA",  # Thiamine (B1)
    "RIBF",  # Riboflavin (B2)
    "NIA",  # Niacin
    "NIAEQ",  # Niacin equivalent (NE)
    "VITPYRID",  # Pyridoxine (B6)
    "FOL",  # Folate
    "VITB12",  # Vitamin B12
    "VITC",  # Vitamin C
    "VITD",  # Vitamin D
    "VITE",  # Vitamin E (alpha-tocopherol)
    "VITK",  # Vitamin K
    "CAROTENS",  # Carotenoids
]

# Amino acids
amino_acids = [
    "TRP",  # Tryptophan
]

if __name__ == "__main__":
    # Define constants
    SHOW_PLOT = False
    DATA_DIR = "tmp"
    DATA_DIR_WEIGHTED = "tmp_weighted"
    DATASET = "Fineli_Rel20"
    SIMILARITY_THRESHOLD = 0.8

    # Run analysis with default parameters
    network_analysis = run_nutritional_network_analysis(
        data_dir=DATA_DIR,
        dataset=DATASET,
        similarity_threshold=SIMILARITY_THRESHOLD,
        show_plot=SHOW_PLOT,
        key_nutrients=key_nutrients,
    )

    print_task_header(9, "Repeat 3-8 with weighted graph")
    # Run analysis with weighted parameters
    weighted_network_analysis = run_nutritional_network_analysis(
        data_dir=DATA_DIR_WEIGHTED,
        dataset=DATASET,
        similarity_threshold=SIMILARITY_THRESHOLD,
        show_plot=SHOW_PLOT,
        weighted=True,
        key_nutrients=key_nutrients,
    )
