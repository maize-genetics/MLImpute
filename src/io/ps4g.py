import pandas as pd
import numpy as np
import logging


def convert_ps4g(ps4g_file, weight, collapse):
    """
    This function converts PS4G file into a multihot encoded matrix with optional weighting and collapsing.

    Args:
        data (str): PS4G file
        weight (str): Weighting strategy, can be 'global', 'read', or 'unweighted'.
        collapse (bool): If True, collapses the gamete sets into a single row per position.

    Returns:
        numpy.ndarray: A multihot encoded matrix representing the PS4G data.
    """

    ps4g_data = load_ps4g_file(ps4g_file)
    metadata, gamete_data = extract_metadata(ps4g_file)

    input_matrix = create_multihot_matrix(ps4g_data, gamete_data, weight, collapse)
    logging.info(f"Converted PS4G file {ps4g_file} with weight '{weight}' and collapse={collapse}.")
    return input_matrix




def load_ps4g_file(ps4g_file):
    """
    Load a PS4G file and return the DataFrame.

    Args:
        ps4g_file (str): Path to the PS4G file.

    Returns:
        pd.DataFrame: DataFrame containing the PS4G data.
    """
    ps4g = pd.read_csv(ps4g_file, delimiter="\t", comment="#")
    ps4g['gameteSet'] = ps4g['gameteSet'].apply(lambda x: list(map(int, x.split(','))))
    return ps4g

def extract_metadata(ps4g_file):
    """
    Extract metadata from the PS4G file.

    Args:
        ps4g_file (str): Path to the PS4G file.

    Returns:
        dict: A dictionary containing metadata such as sample name, filename, command, and total reads.
        list: A list of dictionaries containing gamete data with gamete name, index, read count, and weight.
    """
    metadata = {
        "sample_name": None,
        "filename1": None,
        "command": None,
        "total_reads": None
    }
    gamete_data = []

    with open(ps4g_file, 'r') as file:
        comments = [line.strip() for line in file if line.startswith('#')]

    for line in comments:
        metadata["sample_name"] = ps4g_file.split('-')[1].split('_')[0]
        if line.startswith("##filename1="):
            metadata["filename1"] = line.split("=")[1]
        elif line.startswith("#Command:"):
            metadata["command"] = line.replace("#Command: ", "")
        elif line.startswith("#TotalUniqueCounts:"):
            metadata["total_reads"] = int(line.split(":")[1])
        elif line.startswith("#") and ":" in line and "\t" in line:
            # Example line: "#B73:0\t1\t10730006"
            line = line[1:]  # Remove leading "#"
            gamete_full, idx, count = line.split("\t")
            gamete_name = gamete_full.split(":")[0]
            gamete_data.append({
                "gamete": gamete_name,
                "gamete_index": int(idx),
                "read_count": int(count),
                "weight": int(count)/metadata["total_reads"]
            })

    return metadata, gamete_data

def create_multihot_matrix(ps4g, gamete_data, weight, collapse):
    """
    Create a multihot encoded matrix from the PS4G data.

    Args:
        ps4g (pd.DataFrame): DataFrame containing the PS4G data.
        gamete_data (list): List of dictionaries containing gamete data.
        weight (str): Weighting strategy.
        collapse (bool): If True, collapses the gamete sets.

    Returns:
        np.ndarray: A multihot encoded matrix.
    """
    # Get number of unique gametes
    gamete_indices = [entry["gamete_index"] for entry in gamete_data]
    num_classes = len(gamete_indices)

    # Map position to index
    unique_positions = ps4g['pos'].unique()
    pos_to_idx = {pos: i for i, pos in enumerate(unique_positions)}

    if collapse:
        logging.info("Collapsing")
        X_multihot, collapsed_df = collapse_ps4g(num_classes, ps4g, unique_positions)

    else:
        logging.info("not collapsed")
        X_multihot = np.zeros((len(ps4g), num_classes), dtype=np.float32)

        for i, indices in enumerate(ps4g['gameteSet']):
            X_multihot[i, indices] = 1  # vectorized assignment

    if weight == "read":
        input_matrix = process_read_weight_mode(X_multihot, collapsed_df, num_classes, pos_to_idx, ps4g, unique_positions)

    elif weight == "global":
        input_matrix = process_global_weight_mode(X_multihot, gamete_data, num_classes)

    else:
        logging.info("unweighted")
        input_matrix = X_multihot

    return input_matrix


def process_global_weight_mode(X_multihot, gamete_data, num_classes):
    """
    Process the global weight mode for the multihot encoded matrix.
    Args:
        X_multihot (np.ndarray): The multihot encoded matrix.
        gamete_data (list): List of dictionaries containing gamete data.
        num_classes (int): Number of unique gametes.
    Returns:
        np.ndarray: The input matrix with global weights applied.
    """
    logging.info("global weight")
    # Map index → weight
    index_to_weight = {entry["gamete_index"]: entry["weight"] for entry in gamete_data}
    # Create aligned weight vector for all columns
    global_weights = np.array([index_to_weight.get(i, 0.0) for i in range(num_classes)], dtype=np.float32)
    input_matrix = X_multihot * global_weights
    return input_matrix


def process_read_weight_mode(X_multihot, collapsed_df, num_classes, pos_to_idx, ps4g, unique_positions):
    """
    Process the read weight mode for the multihot encoded matrix.
    Args:
        X_multihot (np.ndarray): The multihot encoded matrix.
        collapsed_df (pd.DataFrame): DataFrame containing collapsed PS4G data.
        num_classes (int): Number of unique gametes.
        pos_to_idx (dict): Mapping from position to index.
        ps4g (pd.DataFrame): DataFrame containing the PS4G data.
        unique_positions (np.ndarray): Array of unique positions.
    Returns:
        np.ndarray: The input matrix with read weights applied.
    """

    logging.info("read count")
    # Initialize output matrix
    count_matrix = np.zeros((len(unique_positions), num_classes), dtype=np.float32)
    # Accumulate counts per position per gamete
    for _, row in ps4g.iterrows():
        pos_idx = pos_to_idx[row['pos']]
        for gamete in row['gameteSet']:
            count_matrix[pos_idx, gamete] += row['count']
    input_matrix = np.empty(X_multihot.shape, dtype=np.float32)
    for i in range(len(X_multihot)):
        input_matrix[i] = X_multihot[i] * count_matrix[i] / collapsed_df['count'][i]
    return input_matrix


def collapse_ps4g(num_classes, ps4g, unique_positions):
    """ Collapse the PS4G DataFrame by position and aggregate gamete sets.
    Args:
        num_classes (int): Number of unique gametes.
        ps4g (pd.DataFrame): DataFrame containing the PS4G data.
        unique_positions (np.ndarray): Array of unique positions.
    Returns:
        np.ndarray: A multihot encoded matrix with collapsed gamete sets.
        pd.DataFrame: DataFrame containing collapsed PS4G data with aggregated counts.
    """
    collapsed_df = ps4g.groupby('pos').agg({
        'gameteSet': lambda x: sorted(set().union(*x)),
        'count': 'sum'
    }).reset_index()
    collapsed_df = collapsed_df.set_index('pos').loc[unique_positions].reset_index()
    X_multihot = np.zeros((len(collapsed_df), num_classes), dtype=np.float32)
    for i, indices in enumerate(collapsed_df['gameteSet']):
        X_multihot[i, indices] = 1  # vectorized assignment
    return X_multihot, collapsed_df