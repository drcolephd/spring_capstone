import pickle
import os
import argparse

FILENAME = "muaps_model.pkl"

def load_pickle(filename):
    """Load existing pickle file or return an empty dictionary if it doesn't exist."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            try:
                data = pickle.load(f)
                if not isinstance(data, dict):
                    print("⚠️ Warning: Existing file is not a dictionary. Resetting.")
                    return {}
                return data
            except (pickle.UnpicklingError, EOFError):
                print("⚠️ Warning: Pickle file corrupted. Resetting.")
                return {}
    return {}

def save_pickle(filename, data):
    """Save dictionary to pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Successfully saved updates to {filename}")

def modify_variable(var_name, var_value, data):
    """Modify or add a variable in the dictionary."""
    data[var_name] = var_value
    print(f"🔹 Set `{var_name}` to: {var_value}")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify or append a variable to muaps_model.pkl")
    parser.add_argument("var_name", type=str, help="Variable name to modify or append")
    parser.add_argument("var_value", type=str, help="New value for the variable")
    args = parser.parse_args()

    # Load existing data
    muaps_data = load_pickle(FILENAME)

    # Try to convert var_value to a Python object (int, float, list, dict, etc.)
    try:
        new_value = eval(args.var_value)  # Convert string input to Python type (careful with eval!)
    except:
        new_value = args.var_value  # Keep as string if conversion fails

    # Modify the variable
    muaps_data = modify_variable(args.var_name, new_value, muaps_data)

    # Save back to the pickle file
    save_pickle(FILENAME, muaps_data)
