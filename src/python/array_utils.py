"""
Utility functions for converting NumPy arrays to JSON format for Tauri frontend.
"""
import numpy as np
import json
import base64
from typing import Dict, Any, Union


def numpy_to_json(arr: np.ndarray) -> Dict[str, Any]:
    """
    Convert a NumPy array to a JSON-serializable format.
    
    Args:
        arr: NumPy array to convert
        
    Returns:
        Dictionary containing base64-encoded data, shape, and dtype
    """
    return {
        "data": base64.b64encode(arr.tobytes()).decode('utf-8'),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype)
    }


def create_visualization_response(
    matrix: np.ndarray, 
    row_labels: Union[list, np.ndarray] = None,
    col_labels: Union[list, np.ndarray] = None,
    metadata: Dict[str, Any] = None
) -> str:
    """
    Create a complete JSON response for visualization data.
    
    Args:
        matrix: 2D NumPy array containing the main data
        row_labels: Optional labels for rows
        col_labels: Optional labels for columns  
        metadata: Optional additional metadata
        
    Returns:
        JSON string ready for Tauri frontend
    """
    response = {
        "status": "success",
        "matrix": numpy_to_json(matrix)
    }
    
    if row_labels is not None:
        if isinstance(row_labels, np.ndarray):
            row_labels = row_labels.tolist()
        response["row_labels"] = row_labels
        
    if col_labels is not None:
        if isinstance(col_labels, np.ndarray):
            col_labels = col_labels.tolist()
        response["col_labels"] = col_labels
        
    if metadata is not None:
        response["metadata"] = metadata
        
    return json.dumps(response)


def create_error_response(error_message: str) -> str:
    """
    Create a standardized error response.
    
    Args:
        error_message: Description of the error
        
    Returns:
        JSON string with error information
    """
    response = {
        "status": "error",
        "error": error_message
    }
    return json.dumps(response)


def generate_parent_path_data(num_positions: int = 20, num_samples: int = 8, seed: int = 42) -> Dict[str, Any]:
    """
    Generate sample data for parent path visualization.
    
    Creates a matrix where:
    - Columns represent genomic positions
    - Rows represent unique sample IDs
    - Two paths show parent1 and parent2 trajectories through samples
    - Highlighted cells show the parent+position combinations
    
    Args:
        num_positions: Number of genomic positions (columns)
        num_samples: Number of unique sample IDs (rows)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with matrix, labels, and path data
    """
    np.random.seed(seed)
    
    # Create sample IDs that will be our row labels
    sample_ids = [f"Sample_{i:03d}" for i in range(num_samples)]
    
    # Create position labels (columns)
    position_labels = [f"Pos_{i}" for i in range(num_positions)]
    
    # Create a random matrix for background data
    matrix = np.random.rand(num_samples, num_positions).astype(np.float32)
    
    # Generate parent paths - each parent follows a path through samples
    parent1_path = []
    parent2_path = []
    
    for pos in range(num_positions):
        # Each parent can be at any sample at each position
        parent1_sample_idx = np.random.randint(0, num_samples)
        parent2_sample_idx = np.random.randint(0, num_samples)
        
        parent1_path.append({
            'position': position_labels[pos],
            'sample': sample_ids[parent1_sample_idx],
            'row_idx': parent1_sample_idx,
            'col_idx': pos
        })
        
        parent2_path.append({
            'position': position_labels[pos], 
            'sample': sample_ids[parent2_sample_idx],
            'row_idx': parent2_sample_idx,
            'col_idx': pos
        })
        
        # Enhance matrix values at parent locations to make paths visible
        matrix[parent1_sample_idx, pos] = 0.8 + 0.2 * np.random.random()
        matrix[parent2_sample_idx, pos] = 0.1 + 0.2 * np.random.random()
    
    return {
        'matrix': matrix,
        'row_labels': sample_ids,
        'col_labels': position_labels,
        'parent1_path': parent1_path,
        'parent2_path': parent2_path,
        'metadata': {
            'type': 'parent_paths',
            'description': f'Parent paths through {num_samples} samples across {num_positions} positions',
            'generated': True,
            'seed': seed
        }
    }


if __name__ == "__main__":
    # Example usage for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample visualization data")
    parser.add_argument("--rows", type=int, default=8, help="Number of unique samples (rows)")
    parser.add_argument("--cols", type=int, default=20, help="Number of positions (columns)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="parents", choices=["parents", "basic"], 
                       help="Visualization mode: 'parents' for parent paths, 'basic' for simple matrix")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "parents":
            # Generate parent path data
            data = generate_parent_path_data(args.cols, args.rows, args.seed)
            
            # Create highlights for both parent paths
            highlights = []
            for path_point in data['parent1_path']:
                highlights.append({
                    'row': path_point['sample'],
                    'col': path_point['position'],
                    'parent': 'parent1'
                })
            for path_point in data['parent2_path']:
                highlights.append({
                    'row': path_point['sample'], 
                    'col': path_point['position'],
                    'parent': 'parent2'
                })
            
            # Add path data to metadata
            data['metadata']['parent1_path'] = data['parent1_path']
            data['metadata']['parent2_path'] = data['parent2_path']
            data['metadata']['highlights'] = highlights
            
            result = create_visualization_response(
                data['matrix'], 
                data['row_labels'], 
                data['col_labels'], 
                data['metadata']
            )
        else:
            # Generate basic sample data (original behavior)
            np.random.seed(args.seed)
            matrix = np.random.rand(args.rows, args.cols).astype(np.float32)
            row_labels = [f"Sample_{i}" for i in range(args.rows)]
            col_labels = [f"Pos_{i}" for i in range(args.cols)]
            
            metadata = {
                "generated": True,
                "seed": args.seed,
                "shape_description": f"{args.rows} samples × {args.cols} positions"
            }
            
            result = create_visualization_response(matrix, row_labels, col_labels, metadata)
        
        print(result)
        
    except Exception as e:
        print(create_error_response(str(e)))