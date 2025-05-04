import random
import os
import pickle
from typing import Dict, Any, Callable, List, Optional
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import math
import sys
import datetime
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import louvain_communities
import seaborn as sns
from tqdm import tqdm
from adjustText import adjust_text


default_nutrients = ["ENERC", "PROT", "FAT", "CHOAVL", "FIBC", "VITC"]


class Tee:
    """For redirecting print to file"""

    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.file = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure data is written immediately

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


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


def truncate_text(text, max_length, split_char=None, indicator="..."):
    """Truncate text to max_length and add indicator if truncated.

    Args:
        text (str): The input text to truncate.
        max_length (int): The maximum length of the output text.
        indicator (str): The string to append if text is truncated (default: "...").
        split_char (str, optional): Character to split the text at before truncation limit.

    Returns:
        str: Truncated text, possibly with an indicator appended.
    """
    if len(text) <= max_length:
        return text

    if split_char:
        # Look for the split character within the max_length limit
        split_index = text.find(split_char, 0, max_length - len(indicator))
        if split_index != -1:
            return text[:split_index].strip()

    # If no split_char is provided or not found, truncate normally
    truncate_at = max_length - len(indicator)
    return text[:truncate_at] + indicator


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


def create_nutritional_network(
    data_dir=".", similarity_threshold=0.85, weighted=False, k=50, use_topk=True
):
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

    nutrient_cols = key_nutrients

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
            "weight": 0.10,  # 10% weight
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
            f"Warning: {len(unprocessed)} nutrients ({unprocessed}) were not\
              assigned to any group. Applying default weight."
        )
        for col in unprocessed:
            # Use the pre-scaled values and apply default weight
            scaled_nutrients[col] = scaled_nutrients_base[col] * default_weight

    print("\nNutrient Groups and Weights:")
    print("-" * 40)
    for group_name, group_info in nutrient_groups.items():
        columns = group_info["columns"]
        group_weight = group_info["weight"]
        if columns:  # Only print groups that have nutrients
            individual_weight = group_weight / len(columns) if columns else 0
            print(
                f"Group: {group_name.capitalize()} (Total Weight: {group_weight:.2%})"
            )
            for col in columns:
                print(f"  - {col}: {individual_weight:.3%}")
            print("-" * 40)

    # Calculate cosine similarity on all weighted nutrients at once
    similarity_matrix = cosine_similarity(scaled_nutrients)

    print_task_header(2, "Generate a nutritional network graph")

    # Create a new graph
    G = nx.Graph()

    # Add nodes (food items)
    for i, (food_id, food_name) in enumerate(zip(food_ids, food_names)):
        G.add_node(food_id, name=food_name)

    if use_topk:
        # For each node, add edges only to its k most similar foods
        for i, food_id_i in enumerate(food_ids):
            # Get top k similar foods (excluding self-similarity)
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][0:k]

            for j in top_indices:
                food_id_j = food_ids[j]
                # Add this check to prevent self-loops
                if food_id_i != food_id_j:
                    similarity = similarities[j]
                    if similarity >= similarity_threshold:
                        if weighted:
                            G.add_edge(food_id_i, food_id_j, weight=similarity)
                        else:
                            G.add_edge(food_id_i, food_id_j)

    else:
        if not weighted:
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


def analyze_community_nutrition(graph_data, communities, nutrients=None):
    """
    Analyzes the nutritional composition of each community

    Args:
        graph_data: Dictionary containing graph, food mapping, and nutritional data
        communities: List of communities (each community is a set of node IDs)

    Returns:
        DataFrame containing average nutritional values for each community
    """
    if nutrients is None:
        nutrients = default_nutrients

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
        avg_nutrients = community_foods[nutrients].mean().to_dict()

        # Add community info
        avg_nutrients["community_id"] = i
        avg_nutrients["size"] = len(community)
        avg_nutrients["food_examples"] = "\n ".join(
            community_foods["FOODNAME"].head(3).tolist()
        )

        # Add to results
        community_nutrition.append(avg_nutrients)

    # Convert to DataFrame
    result_df = pd.DataFrame(community_nutrition)

    return result_df


def create_community_summary_table(
    community_nutrition, component_names, nutrients=None
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
    if nutrients is None:
        nutrients = default_nutrients

    # Filter nutrients that exist in our datakey_nutrients
    available_nutrients = [n for n in nutrients if n in community_nutrition.columns]

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
    nutrients=None,
    output_dir=".",
    show_plot=True,
):
    """
    Creates visualizations comparing nutritional differences between communities.

    Args:
        community_nutrition: DataFrame with community nutritional data
        component_names: Dictionary mapping component codes to readable names
        key_nutrients: List of nutrient codes to include
    """
    if nutrients is None:
        nutrients = default_nutrients

    # Filter to nutrients that exist in our data
    available_nutrients = [n for n in nutrients if n in community_nutrition.columns]

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

    # Set up the number of bars and positions
    n_communities = len(plot_data)

    # 1. HEATMAP: Show relative nutrient composition
    plt.figure(figsize=(14, 10))

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
    filename = f"{output_dir}/relative_nutrient_heatmap.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()

    # 2. RADAR CHART: Profile of each community
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
        variable_count = len(available_nutrients)

        # Compute angle for each axis
        angles = [n / float(variable_count) * 2 * np.pi for n in range(variable_count)]
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

        filename = f"{output_dir}/nutritional_profile_of_communities.png"
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()


def visualize_network(graph_data, max_nodes=100, output_dir=".", show_plot=True):
    """
    Visualizes the nutritional network graph

    Args:
        graph_data: Output from create_nutritional_network
        max_nodes: Maximum number of nodes to display for readability
    """
    filename = f"{output_dir}/network.png"
    if os.path.isfile(filename):
        return

    G = graph_data["graph"]
    food_mapping = graph_data["food_mapping"]

    # If the graph is very large, take a subset for visualization
    if G.number_of_nodes() > max_nodes:
        # Get a random subset of nodes
        all_nodes = list(G.nodes())
        top_nodes = random.sample(all_nodes, min(max_nodes, len(all_nodes)))
        G_viz = G.subgraph(top_nodes)
        print(f"Visualizing a random subset of {len(top_nodes)} nodes")
    else:
        G_viz = G

    # Set up the plot
    plt.figure(figsize=(14, 10))

    # Position nodes
    pos = nx.kamada_kawai_layout(G_viz)

    # Draw nodes
    nx.draw_networkx_nodes(G_viz, pos, node_size=50, alpha=0.8)

    # Draw edges with weights affecting thickness (use 1 if no weight assigned)
    edge_weights = [
        G_viz.get_edge_data(u, v).get("weight", 1) * 2 for u, v in G_viz.edges()
    ]
    nx.draw_networkx_edges(G_viz, pos, width=edge_weights, alpha=0.4, edge_color="gray")

    plt.title("Nutritional Similarity Network of Food Items")
    plt.axis("off")
    plt.tight_layout()

    # Save the visualization
    plt.savefig(filename, bbox_inches="tight", dpi=300)

    # Show plot
    if show_plot:
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

    def weight_distance(u, v, d):
        """Calculate distance from weight"""
        if "weight" in d:
            return 1.0 / d["weight"]

        return 1.0

    # More computationally expensive metrics
    if use_approximation and G.number_of_nodes() > 1000:
        print_result("Using approximation for closeness centrality", "", indent=6)
        sampled_nodes = random.sample(
            list(G.nodes()), min(sample_size, G.number_of_nodes())
        )
        centrality["closeness"] = {}

        for node in sampled_nodes:
            centrality["closeness"][node] = nx.closeness_centrality(
                G, u=node, distance=weight_distance
            )

        print_result("Using approximation for betweenness centrality", "", indent=6)
        centrality["betweenness"] = nx.betweenness_centrality(
            G, k=k_betweenness, seed=42, weight="weight"
        )
    else:
        print_result("Calculating full closeness centrality", "", indent=6)
        centrality["closeness"] = nx.closeness_centrality(G, distance=weight_distance)

        print_result("Calculating full betweenness centrality", "", indent=6)
        centrality["betweenness"] = nx.betweenness_centrality(
            G, seed=42, weight="weight"
        )

    return centrality


def plot_centrality_histograms(
    centrality_dict, network_name="Network", output_dir=".", show_plot=True
):
    """Plot histograms for different centrality measures."""
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

        filename = f"{output_dir}/{c_type}_centrality_{network_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches="tight", dpi=300)

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
        filename = f"{output_dir}/combined_centrality_{network_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()


def analyze_centrality_power_law(centrality_dict, show_plot=True, output_dir="."):
    """
    Analyze if the centrality distributions follow a power law.

    Parameters:
    - centrality_dict: Dictionary of centrality measures from calculate_centralities function
    - show_plot: Whether to display plots
    - save_plots: Whether to save plot images
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
        # Check if there are any positive values
        positive_values = [v for v in values if v > 0]
        if not positive_values:
            print_result(
                f"{centrality_type.capitalize()} distribution analysis",
                "No positive values found for analysis",
            )
            continue

        # Now safely find minimum non-zero value
        min_non_zero = min(positive_values) / 10
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

            filename = f"{output_dir}/{centrality_type}_powerlaw.png"
            plt.savefig(filename, bbox_inches="tight", dpi=300)

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


def plot_communities(
    G,
    communities,
    title,
    output_dir=".",
    food_items=None,
    max_labels_per_community=2,
    show_plot=True,
):
    """Plot communities with optional food labels for specific nodes, preventing label overlap"""
    filename = f"{output_dir}/communities.png"
    if os.path.isfile(filename):
        return

    plt.figure(figsize=(12, 8))

    # Convert the set/dictionary to a list before sampling
    samples = []
    for comm in communities:
        comm_list = list(comm)  # Convert to list

        # Get degrees of nodes in this community
        node_degrees = [(node, G.degree(node)) for node in comm_list]

        # Sort by degree (highest first)
        sorted_nodes = [
            node
            for node, degree in sorted(node_degrees, key=lambda x: x[1], reverse=True)
        ]

        # Take 10% of nodes from each community, minimum 5 nodes if available
        num_to_select = min(
            len(comm_list), max(5, int(math.ceil(len(comm_list) * 0.20)))
        )
        samples.extend(sorted_nodes[:num_to_select])
    G_sampled = G.subgraph(samples)

    # Map nodes to their community
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    # Colors for each community
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
    node_colors = [colors[community_map[node]] for node in G_sampled.nodes()]

    # Position nodes using spring layout
    pos = nx.kamada_kawai_layout(G_sampled)

    # Create food_id to food_name mapping from the dictionary structure
    food_mapping = {}
    if food_items:
        for index in food_items:
            for food in food_items[index]:
                food_mapping[food["food_id"]] = food["food_name"]

    # Create a dictionary of labels, limiting the number per community
    node_labels = {}
    if food_mapping:
        # Group sampled nodes by community
        community_nodes = {}
        for node in G_sampled.nodes():
            if node in community_map:
                comm_idx = community_map[node]
                if comm_idx not in community_nodes:
                    community_nodes[comm_idx] = []
                community_nodes[comm_idx].append(node)

        # For each community, add labels for up to max_labels_per_community food nodes
        for comm_idx, nodes in community_nodes.items():
            label_count = 0
            for node in nodes:
                if node in food_mapping and label_count < max_labels_per_community:
                    node_labels[node] = truncate_text(
                        food_mapping[node], 20, split_char=","
                    )
                    label_count += 1

    # Draw the graph
    nx.draw_networkx(
        G_sampled,
        pos=pos,
        node_color=node_colors,
        with_labels=False,  # Don't show all labels
        node_size=20,
        edge_color="gray",
        alpha=0.8,
        width=0.2,
    )

    # Create text objects for labels with adjust_text instead of draw_networkx_labels
    texts = []
    for node, label in node_labels.items():
        x, y = pos[node]
        texts.append(
            plt.text(
                x, y, label, fontsize=8, fontweight="bold", ha="center", va="center"
            )
        )

    # Adjust text positions to avoid overlaps
    if texts:
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    filename = f"{output_dir}/communities.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def calculate_community_stats(G, communities):
    """Calculate community stats"""
    stats = []

    for i, comm in tqdm(enumerate(communities)):
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
    Identifies the top-N individual food items within each community based on
    average nutritional similarity to other foods in the same community.

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
    graph_data, community_top_foods, component_names, nutrients=None
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
    if nutrients is None:
        nutrients = default_nutrients
    food_nutrients = graph_data["food_nutrients"]

    # Calculate global averages for normalization

    global_avg = food_nutrients[nutrients].mean()
    global_std = food_nutrients[nutrients].std()

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
        avg_nutrients = relevant_foods[nutrients].mean()

        # Identify dominant characteristics using z-scores
        nutrient_values = []
        for code in nutrients:
            if not pd.isna(avg_nutrients[code]) and global_std[code] > 0:
                # Calculate z-score: how many standard deviations from the mean
                z_score = (avg_nutrients[code] - global_avg[code]) / global_std[code]
                nutrient_values.append(
                    (
                        component_names.get(code, code),
                        z_score,
                        avg_nutrients[code],
                    )
                )

        # Sort by absolute z-score
        nutrient_values.sort(key=lambda x: abs(x[1]), reverse=True)
        # # Sort by positive z-score
        # nutrient_values.sort(key=lambda x: x[1], reverse=True)
        dominant_traits = [
            f"{name}: {value:.2f} (z-score: {rel:.2f})"
            for name, rel, value in nutrient_values[:3]  # Showing top 3
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
    data_dir=".",
    dataset="Fineli_Rel20",
    similarity_threshold=0.80,
    weighted=False,
    k=20,
):
    """Initialize data directory and create file paths for storing data."""
    os.makedirs(data_dir, exist_ok=True)
    files = {
        "graph_data": os.path.join(data_dir, "graph_data.pkl"),
        "centrality": os.path.join(data_dir, "centrality_measures.pkl"),
        "gn_communities": os.path.join(data_dir, "gn_communities.pkl"),
        "louvain_communities": os.path.join(data_dir, "louvain_communities.pkl"),
        "clustering": os.path.join(data_dir, "clustering.pkl"),
        "assortativity": os.path.join(data_dir, "assortativity.pkl"),
    }

    # Load or calculate graph data
    graph_data = load_or_calculate(
        files["graph_data"],
        create_nutritional_network,
        calculate_args=[dataset],
        calculate_kwargs={
            "similarity_threshold": similarity_threshold,
            "weighted": weighted,
            "k": k,
            "use_topk": True,
        },
        description="graph data",
    )

    return files, graph_data


def analyze_centrality(G, files, show_plot=False, output_dir=".", weighted=True):
    """Analyze and visualize centrality measures."""
    print_task_header(3, "Visualize and plot the degree distribution")
    approximate = False

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
            output_dir=f"{output_dir}/centrality",
            show_plot=show_plot,
        )
    else:
        plot_centrality_histograms(
            {
                "degree": centrality_measures["degree"],
                "closeness": centrality_measures["closeness"],
                "betweenness": centrality_measures["betweenness"],
            },
            "full",
            output_dir=f"{output_dir}/centrality",
            show_plot=show_plot,
        )

    print_task_header(4, "Provide the script for drawing power law distributions")
    results = analyze_centrality_power_law(
        centrality_measures,
        show_plot=show_plot,
        output_dir=f"{output_dir}/powerlaw",
    )

    return centrality_measures, results


def analyze_clustering(G, files, show_plot=False, output_dir=".", weighted=True):
    """Analyze clustering coefficients in the network."""
    print_task_header(
        5,
        "Utilize the NetworkX clustering function to calculate the clustering coefficient",
    )

    node_clustering_coefficients = load_or_calculate(
        files["clustering"],
        nx.clustering,
        calculate_args=[G],
        calculate_kwargs={"weight": ("weight" if weighted else None)},
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
    plt.savefig(filename, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return node_clustering_coefficients


def detect_communities(G, files):
    """Detect communities using Girvan-Newman and Louvain algorithms."""
    print_task_header(6, "Detect communities within the nutritional network")

    # Load or calculate Louvain communities
    louvain_comms = load_or_calculate(
        files["louvain_communities"],
        louvain_communities,
        calculate_args=[G],
        calculate_kwargs={"weight": "weight"},
        description="Louvain communities",
    )

    return louvain_comms


def analyze_nutritional_composition(
    graph_data, communities, dataset, nutrients=None, output_dir=".", show_plot=True
):
    """Analyze the nutritional composition of communities."""
    if nutrients is None:
        nutrients = default_nutrients
    print_task_header(7, "Analyze Community Nutritional Composition")

    # Load component name mappings
    component_names = load_component_names(dataset)

    community_nutrition = analyze_community_nutrition(
        graph_data, communities, nutrients
    )

    # Generate summary table
    print("\nSummary Table of Community Nutritional Differences:")
    summary_table = create_community_summary_table(
        community_nutrition, component_names, nutrients
    )
    print(summary_table)

    # Visualize differences
    print("\nGenerating visualizations to compare communities...")
    visualize_community_differences(
        community_nutrition, component_names, nutrients, output_dir, show_plot=show_plot
    )

    return community_nutrition, component_names


def analyze_top_similar_foods(graph_data, communities, component_names, nutrients=None):
    """Find and analyze the top similar foods within communities."""
    if nutrients is None:
        nutrients = default_nutrients
    community_top_foods = find_top_similar_foods_in_communities(graph_data, communities)

    print_task_header(8, "Identify the top-10 most similar food items")
    top_food_analysis = analyze_top_food_characteristics(
        graph_data, community_top_foods, component_names, nutrients
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
        print(f"Examples:\n {row['food_examples']}")
        print(f"Base nutrients:")

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


def pagerank(G, dampening=0.85, output_dir=".", show_plot=True):
    # Calculate PageRank scores
    pagerank_scores = nx.pagerank(G, alpha=dampening, weight="weight")

    # Get top 10 using readable names
    sorted_items = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]
    top_10 = [(G.nodes[node_id]["name"], score) for node_id, score in sorted_items]

    # Create a DataFrame for visualization
    df = pd.DataFrame(top_10, columns=["Food Item", "PageRank Score"])

    # Create bar chart visualization
    plt.figure(figsize=(10, 6))
    plt.barh(df["Food Item"], df["PageRank Score"], color="skyblue")
    plt.xlabel("PageRank Score")
    plt.ylabel("Food Item")
    plt.title("Top 10 Influential Food Items by PageRank Score")
    plt.tight_layout()
    filename = f"{output_dir}/pagerank_score.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()

    # Create network visualization of top 10 items
    top_10_nodes = [item[0] for item in sorted_items[:10]]  # Get IDs of top 10 nodes
    subgraph = G.subgraph(top_10_nodes)

    # Create figure with explicit axes
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, seed=42, k=0.5)  # Increase 'k' for more spacing

    # Create a mapping of node IDs to food names for labels
    node_labels = {node: G.nodes[node]["name"] for node in subgraph.nodes()}

    # Create color map
    node_colors = [pagerank_scores[node] for node in subgraph.nodes()]
    cmap = plt.cm.Reds

    # Draw the network (without labels initially)
    nx.draw_networkx_nodes(
        subgraph,
        pos=pos,
        node_color=node_colors,
        cmap=cmap,
        ax=ax,
        node_size=60,
        alpha=0.8,
    )

    nx.draw_networkx_edges(subgraph, pos=pos, width=1.5, edge_color="gray", ax=ax)

    # Add labels manually and collect them for adjustment
    texts = []
    for node, label in node_labels.items():
        x, y = pos[node]
        text = ax.text(x, y, label, fontsize=10, ha="center", va="center")
        texts.append(text)

    # Adjust text positions to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

    # Add colorbar with explicit axes reference
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="PageRank Score")

    plt.title("Network of Top 10 Influential Food Items")
    ax.set_axis_off()  # Turn off axis
    plt.tight_layout()
    filename = f"{output_dir}/pagerank_network.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_network_assortativity(network_data, attribute_names=None, n_bins=5):
    """
    Analyzes the assortativity of a nutritional network based on various attributes.

    Args:
        network_data: The output dictionary from create_nutritional_network
        attribute_names: List of nutrient names to analyze for assortativity
                         If None, will use a default set of important nutrients
        n_bins: Number of bins to discretize continuous nutrient values

    Returns:
        Dictionary containing assortativity coefficients and visualization
    """
    G = network_data["graph"]
    food_nutrients = network_data["food_nutrients"]

    # Default to key macronutrients if none specified
    if attribute_names is None:
        attribute_names = [
            "ENERC",
            "PROT",
            "FAT",
            "CHOAVL",
            "FIBC",
        ]  # Energy, Protein, Fat, Carbs, Fiber

    # Ensure all attribute names are valid
    valid_attrs = [attr for attr in attribute_names if attr in food_nutrients.columns]
    if len(valid_attrs) < len(attribute_names):
        missing = set(attribute_names) - set(valid_attrs)
        print(f"Warning: The following attributes were not found: {missing}")

    results = {}

    # Calculate assortativity for each attribute
    for attr in tqdm(valid_attrs):
        # Get attribute values for all nodes
        attr_values = {}
        for node in G.nodes():
            # Find the attribute value for this food item
            node_data = food_nutrients[food_nutrients["FOODID"] == node]
            if not node_data.empty:
                attr_values[node] = node_data[attr].values[0]

        # Add values as node attributes in the graph
        nx.set_node_attributes(G, attr_values, name=attr)

        # Calculate numeric assortativity
        numeric_assortativity = nx.numeric_assortativity_coefficient(G, attr)

        # Discretize the values into bins for categorical assortativity
        values = np.array(list(attr_values.values()))
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            bins = np.linspace(np.min(valid_values), np.max(valid_values), n_bins + 1)
            binned_values = {}
            for node, value in attr_values.items():
                if not np.isnan(value):
                    bin_idx = np.digitize(value, bins) - 1
                    binned_values[node] = f"Bin {bin_idx+1}"
                else:
                    binned_values[node] = "Unknown"

            nx.set_node_attributes(G, binned_values, name=f"{attr}_bin")
            categorical_assortativity = nx.attribute_assortativity_coefficient(
                G, f"{attr}_bin"
            )
        else:
            categorical_assortativity = np.nan

        results[attr] = {
            "numeric_assortativity": numeric_assortativity,
            "categorical_assortativity": categorical_assortativity,
        }

    return results


def plot_assortativity(results, show_plot=False, output_dir="."):
    """Plot assortativity"""
    # Create visualization
    fig1, ax = plt.subplots(figsize=(12, 6))

    attrs = list(results.keys())
    numeric_values = [results[attr]["numeric_assortativity"] for attr in attrs]
    categorical_values = [results[attr]["categorical_assortativity"] for attr in attrs]

    x = np.arange(len(attrs))
    width = 0.35

    rects1 = ax.bar(x - width / 2, numeric_values, width, label="Numeric")
    rects2 = ax.bar(x + width / 2, categorical_values, width, label="Categorical")

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    autolabel(rects1)
    autolabel(rects2)

    ax.set_ylabel("Assortativity Coefficient")
    ax.set_title("Nutritional Network Assortativity by Attribute")
    ax.set_xticks(x)
    ax.set_xticklabels(attrs, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    filename = f"{output_dir}/nutrient_assortativity_bar_chart.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()

    plt.show()

    # Add interpretation
    print("\nAssortativity Coefficient Analysis:")
    print("-----------------------------------")
    print("Coefficient values range from -1 to 1:")
    print(
        "  * Positive values indicate that nodes tend to connect to other nodes with similar attribute values"
    )
    print("  * Values close to 0 indicate no assortativity (random connections)")
    print(
        "  * Negative values indicate that nodes tend to connect to nodes with dissimilar values (disassortativity)"
    )
    print("\nResults:")

    for attr in attrs:
        num_val = results[attr]["numeric_assortativity"]
        cat_val = results[attr]["categorical_assortativity"]

        print(f"\n{attr}:")
        print(f"  Numeric assortativity: {num_val:.4f}")
        print(f"  Categorical assortativity: {cat_val:.4f}")

        # Interpret the values
        def interpret(val):
            if pd.isna(val):
                return "Could not be calculated (insufficient data)"
            elif val > 0.3:
                return "Strong tendency for similar foods to connect"
            elif val > 0.1:
                return "Moderate tendency for similar foods to connect"
            elif val > -0.1:
                return "Little to no assortativity pattern"
            elif val > -0.3:
                return "Moderate tendency for dissimilar foods to connect"
            else:
                return "Strong tendency for dissimilar foods to connect"

        print(f"  Interpretation: {interpret(num_val)}")

    # Create a correlation heatmap for nutrients vs assortativity
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    # Prepare data for the heatmap
    assortativity_data = {
        "Nutrient": attrs,
        "Numeric Assortativity": numeric_values,
        "Categorical Assortativity": categorical_values,
    }
    assortativity_df = pd.DataFrame(assortativity_data).set_index("Nutrient")

    # Plot the heatmap
    sns.heatmap(
        assortativity_df,
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Assortativity Coefficient"},
        ax=ax2,
    )
    plt.title("Nutrient Assortativity Heatmap")
    plt.tight_layout()
    filename = f"{output_dir}/nutrient_assortativity_heatmap.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        "results": results,
        "attributes": attrs,
        "assortativity_df": assortativity_df,
    }


def k_core_analyzis(G, output_dir="."):
    """K-core analyzis"""
    # Compute the core number for each node
    core_numbers = nx.core_number(G)

    # Get the distribution of core numbers
    core_distribution = Counter(core_numbers.values())
    max_core = max(core_numbers.values())

    print(f"Maximum core number: {max_core}")
    print(f"Distribution of core numbers: {dict(core_distribution)}")

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(core_distribution.keys(), core_distribution.values())
    plt.xlabel("Core Number")
    plt.ylabel("Number of Nodes")
    plt.title("Distribution of Core Numbers")
    filename = f"{output_dir}/core_distribution.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)

    # Select representative k values (e.g., min, 25%, 50%, 75%, max)
    k_values = [
        1,
        max(2, max_core // 4),
        max(3, max_core // 2),
        max(4, 3 * max_core // 4),
        max_core,
    ]

    for k in k_values:
        # Extract the k-core subgraph
        k_core = nx.k_core(G, k=k)

        print(
            f"k={k} core has {len(k_core.nodes())} nodes and {len(k_core.edges())} edges"
        )

        # Only visualize if the subgraph is small enough
        if len(k_core.nodes()) < 5000:
            plt.figure(figsize=(8, 8))
            pos = nx.spring_layout(k_core, seed=42)
            nx.draw(k_core, pos, node_size=30, with_labels=False)
            plt.title(f"k-core (k={k})")
            filename = f"{output_dir}/k_core_{k}.png"
            plt.savefig(filename, bbox_inches="tight", dpi=300)

    # Group nodes by their core number
    nodes_by_core = {}
    for node, core in core_numbers.items():
        if core not in nodes_by_core:
            nodes_by_core[core] = []
        nodes_by_core[core].append(node)

    # Analyze properties for each core
    core_properties = {}
    for k in sorted(nodes_by_core.keys()):
        nodes = nodes_by_core[k]
        if not nodes:
            continue

        # Calculate average degree
        avg_degree = sum(dict(G.degree(nodes)).values()) / len(nodes)

        # Calculate average clustering coefficient (for smaller cores)
        if len(nodes) < 1000:
            subgraph = G.subgraph(nodes)
            avg_clustering = nx.average_clustering(subgraph)
        else:
            # For large cores, sample nodes to calculate clustering
            sample_size = min(1000, len(nodes))
            sampled_nodes = np.random.choice(nodes, sample_size, replace=False)
            avg_clustering = nx.average_clustering(G.subgraph(sampled_nodes))

        core_properties[k] = {
            "node_count": len(nodes),
            "avg_degree": avg_degree,
            "avg_clustering": avg_clustering,
        }

    # Plot core properties
    plt.figure(figsize=(15, 5))
    ks = sorted(core_properties.keys())

    plt.subplot(1, 3, 1)
    plt.plot(ks, [core_properties[k]["node_count"] for k in ks])
    plt.xlabel("Core Number")
    plt.ylabel("Number of Nodes")
    plt.title("Node Count vs Core Number")

    plt.subplot(1, 3, 2)
    plt.plot(ks, [core_properties[k]["avg_degree"] for k in ks])
    plt.xlabel("Core Number")
    plt.ylabel("Average Degree")
    plt.title("Average Degree vs Core Number")

    plt.subplot(1, 3, 3)
    plt.plot(ks, [core_properties[k]["avg_clustering"] for k in ks])
    plt.xlabel("Core Number")
    plt.ylabel("Average Clustering")
    plt.title("Average Clustering vs Core Number")

    plt.tight_layout()
    filename = f"{output_dir}/core_properties.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)


def run_nutritional_network_analysis(
    output_dir="tmp",
    dataset="Fineli_Rel20",
    similarity_threshold=0.80,
    show_plot=False,
    weighted=False,
    nutrients=None,
    k=20,
):
    """Main function to run the complete nutritional network analysis."""
    if nutrients is None:
        nutrients = default_nutrients

    # Initialize data
    files, graph_data = initialize_data(
        output_dir, dataset, similarity_threshold, weighted, k
    )

    G = graph_data["graph"]
    food_mapping = graph_data["food_mapping"]  # This maps food IDs to food names

    # Create a new graph with food names as node IDs
    G_renamed = nx.relabel_nodes(G, food_mapping)

    filename = f'{output_dir}/graph_{similarity_threshold}{"_weighted" if weighted else ""}.gexf'

    if not os.path.isfile(filename):
        nx.write_gexf(G_renamed, filename)
        print("Saved gexf to", filename)
    else:
        print("File found, not saving")

    filename = f'{output_dir}/graph_{similarity_threshold}{"_weighted" if weighted else ""}.graphml'
    if not os.path.isfile(filename):
        nx.write_graphml(G_renamed, filename)
        print("Saved graphml to", filename)
    else:
        print("File found, not saving")

    visualize_network(
        graph_data, max_nodes=300, output_dir=output_dir, show_plot=show_plot
    )

    # Analyze centrality
    centrality_measures, _ = analyze_centrality(G, files, show_plot, output_dir)

    # Analyze clustering
    analyze_clustering(G, files, show_plot, output_dir, weighted=weighted)

    # Detect communities
    louvain_comms = detect_communities(G, files)

    # Analyze nutritional composition
    community_nutrition, component_names = analyze_nutritional_composition(
        graph_data, louvain_comms, dataset, nutrients, output_dir, show_plot=show_plot
    )

    # Analyze similar foods
    community_top_foods, top_food_analysis = analyze_top_similar_foods(
        graph_data, louvain_comms, component_names
    )

    # Visualize communities
    # plot_communities(
    #     G, gn_communities, "Communities detected by Girvan-Newman algorithm"
    # )
    plot_communities(
        G,
        louvain_comms,
        "Communities detected by Louvain algorithm",
        output_dir,
        show_plot=show_plot,
        food_items=community_top_foods,
    )

    # Display community statistics
    # print_result(
    #     label="Girvan-Newman Communities Statistics:",
    #     value=calculate_community_stats(G, gn_communities),
    #     indent=6,
    # )

    # print("Louvain Communities Statistics:")
    # print(calculate_community_stats(G, louvain_comms))

    # Display results
    display_results(
        community_nutrition, component_names, community_top_foods, top_food_analysis
    )

    pagerank(G, output_dir=output_dir, show_plot=show_plot)

    assortativity_results = load_or_calculate(
        files["assortativity"],
        analyze_network_assortativity,
        calculate_args=[graph_data],
        calculate_kwargs={"attribute_names": nutrients},
        description="Assortativity results",
    )

    plot_assortativity(
        assortativity_results,
        show_plot=show_plot,
        output_dir=output_dir,
    )

    hits_analyzis(graph_data, output_dir=output_dir, show_plot=show_plot)

    k_core_analyzis(G, output_dir=output_dir)

    return {
        "graph": G,
        "centrality_measures": centrality_measures,
        "communities": louvain_comms,
        "community_nutrition": community_nutrition,
        "assortativity_results": assortativity_results,
    }


def hits_analyzis(graph_data, output_dir=".", show_plot=True):
    """HITS analyzis"""
    G = graph_data["graph"]
    food_mapping = graph_data["food_mapping"]

    # Apply the HITS algorithm to graph G
    hub_scores, authority_scores = nx.hits(G, max_iter=100, normalized=True)

    hub_df = pd.DataFrame(
        {
            "node": [food_mapping[node_id] for node_id in hub_scores.keys()],
            "hub_score": list(hub_scores.values()),
        }
    )
    auth_df = pd.DataFrame(
        {
            "node": [food_mapping[node_id] for node_id in authority_scores.keys()],
            "authority_score": list(authority_scores.values()),
        }
    )

    # Sort by scores in descending order
    hub_df = hub_df.sort_values("hub_score", ascending=False)
    auth_df = auth_df.sort_values("authority_score", ascending=False)

    # Print top 10 hubs and authorities
    print("Top 10 Hubs:")
    print(hub_df.head(10))
    print("\nTop 10 Authorities:")
    print(auth_df.head(10))

    # Visualize top hubs and authorities
    def visualize_top_nodes(G, scores, title, top_n=10):
        # Get top N nodes by score
        top_nodes = dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        # Create subgraph with only the top nodes
        subgraph = G.subgraph(top_nodes.keys())

        # Set up visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, seed=42)  # For reproducible layout

        # Add node labels (food names) with smaller font
        labels = {node: food_mapping[node] for node in subgraph.nodes()}
        # nx.draw_networkx_labels(
        #     subgraph, pos, labels=labels, font_size=8, font_family="sans-serif"
        # )

        # Draw the network
        nx.draw_networkx(
            subgraph,
            pos=pos,
            node_color="skyblue",
            with_labels=True,
            font_size=10,
            labels=labels,
            arrows=True,
            node_size=100,
            edge_color="gray",
            alpha=0.8,
            width=0.2,
        )

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()

        filename = f"{output_dir}/HITS_top_nodes.png"
        plt.savefig(filename, bbox_inches="tight", dpi=300)

        if show_plot:
            plt.show()
        else:
            plt.close()

    # Visualize top hubs
    visualize_top_nodes(G, hub_scores, "Top Hubs in Nutritional Network")

    # Visualize top authorities
    visualize_top_nodes(G, authority_scores, "Top Authorities in Nutritional Network")

    # Combined visualization of both hubs and authorities
    def visualize_combined(G, hub_scores, authority_scores, top_n=5):
        # Get top nodes
        top_hubs = dict(
            sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        top_auths = dict(
            sorted(authority_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        # Combine unique nodes
        nodes = set(list(top_hubs.keys()) + list(top_auths.keys()))
        subgraph = G.subgraph(nodes)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, seed=42)

        # Draw different types of nodes
        hub_only = [n for n in top_hubs if n not in top_auths]
        auth_only = [n for n in top_auths if n not in top_hubs]
        both = [n for n in top_hubs if n in top_auths]

        # Add node labels (food names) with smaller font
        labels = {node: food_mapping[node] for node in subgraph.nodes()}
        # nx.draw_networkx_labels(
        #     subgraph, pos, labels=labels, font_size=8, font_family="sans-serif"
        # )

        # Draw the nodes with different colors
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=hub_only,
            node_color="blue",
            node_size=100,
            label="Hubs",
            alpha=0.8,
        )
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=auth_only,
            node_color="red",
            node_size=100,
            label="Authorities",
            alpha=0.8,
        )
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=both,
            node_color="purple",
            node_size=200,
            label="Hub and authority",
            alpha=0.8,
        )

        # Draw edges and labels
        nx.draw_networkx_edges(
            subgraph, pos, arrows=True, alpha=0.8, edge_color="gray", width=0.2
        )
        nx.draw_networkx_labels(subgraph, pos, labels=labels)

        plt.title("Top Hubs and Authorities in Nutritional Network")
        plt.legend()
        plt.axis("off")
        plt.tight_layout()

        filename = f"{output_dir}/HITS_combined.png"
        plt.savefig(filename, bbox_inches="tight", dpi=300)

        if show_plot:
            plt.show()
        else:
            plt.close()

    # Visualize combined
    visualize_combined(G, hub_scores, authority_scores)


if 0:
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

elif 1:
    # Macronutrients
    key_nutrients = [
        "ENERC",  # energy, calculated (kJ)
        "FAT",  # fat, total (g)
        "CHOAVL",  # carbohydrate, available (g)
        "PROT",  # protein, total (g)
        "ALC",  # alcohol (g)
        "FIBC",  # fibre, total (g)
        "STARCH",  # starch, total (g)
        "SUGAR",  # sugars, total (g)
        "FASAT",  # fatty acids, total saturated (g)
        "FAPU",  # fatty acids, total polyunsaturated (g)
        "FATRN",  # fatty acids, total trans (g)
        "F20D5N3",  # fatty acid 20:5 n-3 (EPA)
        "F22D6N3",  # fatty acid 22:6 n-3 (DHA)
        "CHOLE",  # cholesterol (mg)
        "STERT",  # sterols, total (mg)
        "LACS",  # lactose (g)
        "SUGOH",  # sugar alcohols (g)
        "CA",  # calcium (mg)
        "K",  # potassium (mg)
        "MG",  # magnesium (mg)
        "NACL",  # sodium/salt (mg)
        "P",  # phosphorus (mg)
        "FE",  # iron (mg)
        "ID",  # iodine (ug)
        "VITA",  # Vitamin A (RAE)
        "THIA",  # Thiamine (B1)
        "FOL",  # Folate
        "VITB12",  # Vitamin B12
        "VITC",  # Vitamin C
        "VITK",  # Vitamin K
        "CAROTENS",  # Carotenoids
    ]
    # Macronutrients
    macronutrients = [
        "ENERC",  # energy, calculated (kJ)
        "FAT",  # fat, total (g)
        "CHOAVL",  # carbohydrate, available (g)
        "PROT",  # protein, total (g)
        "ALC",  # alcohol (g)
        "FIBC",  # fibre, total (g)
        "STARCH",  # starch, total (g)
        "SUGAR",  # sugars, total (g)
    ]

    # Lipid Profile
    lipid_profile = [
        "FASAT",  # fatty acids, total saturated (g)
        "FAPU",  # fatty acids, total polyunsaturated (g)
        "FATRN",  # fatty acids, total trans (g)
        "F20D5N3",  # fatty acid 20:5 n-3 (EPA)
        "F22D6N3",  # fatty acid 22:6 n-3 (DHA)
        "CHOLE",  # cholesterol (mg)
        "STERT",  # sterols, total (mg)
    ]

    # Sugar Profile
    sugar_profile = [
        "LACS",  # lactose (g)
        "SUGOH",  # sugar alcohols (g)
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
    ]

    # Vitamins
    vitamins = [
        "VITA",  # Vitamin A (RAE)
        "THIA",  # Thiamine (B1)
        "FOL",  # Folate
        "VITB12",  # Vitamin B12
        "VITC",  # Vitamin C
        "VITK",  # Vitamin K
        "CAROTENS",  # Carotenoids
    ]

else:
    key_nutrients = [
        "ENERC",  # energy, calculated (kJ)
        "FAT",  # fat, total (g)
        "PROT",  # protein, total (g)
        "ALC",  # alcohol (g)
        "FIBC",  # fibre, total (g)
        "SUGAR",  # sugars, total (g)
        "FASAT",  # fatty acids, total saturated (g)
        "NACL",  # sodium/salt (mg)
        "CHOAVL",  # carbohydrate, available (g)
    ]


if __name__ == "__main__":
    # Define constants
    SHOW_PLOT = False
    SIMILARITY_THRESHOLD = 0.5
    OUTPUT_DIR = f"tmp_{SIMILARITY_THRESHOLD}"
    OUTPUT_DIR_WEIGHTED = f"tmp_weighted_{SIMILARITY_THRESHOLD}"
    DATASET = "Fineli_Rel20"
    K = 20

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"output_log_{timestamp}.txt"

    sys.stdout = Tee(log_filename, "w")

    print_task_header(0, "Program parameters:")
    print_result("Similarity threshord", SIMILARITY_THRESHOLD)
    print_result("Top K nodes", K)
    print_result("Output dir", OUTPUT_DIR)
    print_result("Output dir (weighted)", OUTPUT_DIR_WEIGHTED)
    print_result("Log file", log_filename)
    print("")  # For prettier output

    # Run analysis with default parameters
    network_analysis = run_nutritional_network_analysis(
        output_dir=OUTPUT_DIR,
        dataset=DATASET,
        similarity_threshold=SIMILARITY_THRESHOLD,
        show_plot=SHOW_PLOT,
        nutrients=key_nutrients,
        k=K,
    )

    print_task_header(9, "Repeat 3-8 with weighted graph")
    # Run analysis with weighted parameters
    weighted_network_analysis = run_nutritional_network_analysis(
        output_dir=OUTPUT_DIR_WEIGHTED,
        dataset=DATASET,
        similarity_threshold=SIMILARITY_THRESHOLD,
        show_plot=SHOW_PLOT,
        weighted=True,
        nutrients=key_nutrients,
        k=K,
    )

    sys.stdout.close()
