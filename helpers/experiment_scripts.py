import os
import json 
import numpy as np

def convert_to_serializable(obj):
    """
    Recursively convert non-serializable objects (like NumPy arrays) to serializable types.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to list
    elif hasattr(obj, 'tolist'):  # If the object has a 'tolist' method (like some tensor types)
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj  # Directly serializable types
    else:
        return str(obj)  # Convert unknown objects to their string representation

def append_to_json(file_path, new_data):
    """
    Appends a dictionary to a JSON file. If the file doesn't exist, it creates a new one.
    
    Args:
        file_path (str): Path to the JSON file.
        new_data (dict): Dictionary to append.
    """
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing content
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                # If the file is empty or contains invalid JSON, initialize as an empty list
                data = []
    else:
        # If the file does not exist, initialize as an empty list
        data = []

    # Convert new_data and its contents to serializable types
    new_data = convert_to_serializable(new_data)

    # Append the new data (ensure the file is a list)
    if isinstance(data, list):
        data.append(new_data)
    else:
        raise ValueError("Expected the JSON file to contain a list")

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def load_json(file_path):
    """
    Loads a JSON file and returns its content.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        dict or list: The content of the JSON file.
        None: If the file does not exist or is empty.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError:
                print("Error: The file contains invalid JSON.")
                return None
    else:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
