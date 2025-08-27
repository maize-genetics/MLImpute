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




