import json
from datetime import datetime
from pathlib import Path

# src/utils/metadata.py

import json
from datetime import datetime
from pathlib import Path

def save_model_metadata(
    model_name,
    mae,
    feature_names,
    hyperparameters,
    path="models/model_metadata.json"
):
    version_path = f"models/{model_name}_version.txt"

    # Increment version
    if Path(version_path).exists():
        version = int(Path(version_path).read_text()) + 1
    else:
        version = 1

    Path(version_path).write_text(str(version))

    metadata = {
        "model_name": model_name,
        "version": version,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mae": mae,
        "hyperparameters": hyperparameters,
        "feature_names": feature_names,
    }

    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)



    
def load_model_metadata(path="models/model_metadata.json"):
    """Load metadata for display in Streamlit."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
