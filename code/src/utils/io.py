from pathlib import Path
from src.config import PROCESSED_DIR

def save_df(df, name: str):
    path = PROCESSED_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


import shutil
import os
from pathlib import Path

def save_df(df, filename, output_dir):
    """
    Saves a dataframe to a local temp file first, then moves it to the target directory.
    This prevents TimeoutErrors when saving to Google Drive.
    """
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    
    # 1. Define a temporary local path (e.g., in your /tmp folder or local project folder)
    # We use .resolve() to get the absolute path
    temp_filename = f"temp_{filename}.csv"
    temp_path = Path.cwd() / temp_filename 

    full_target_path = output_dir / f"{filename}.csv"

    print(f"   Writing temporarily to local disk: {temp_path}...")
    
    try:
        # 2. Heavy lifting: Write the file locally (Fast & Stable)
        df.to_csv(temp_path, index=False)
        
        # 3. Create the destination folder if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"   Moving file to Google Drive: {full_target_path}...")
        
        # 4. Move the finished file to Google Drive
        # shutil.move handles cross-filesystem moves (Local -> Cloud) correctly
        shutil.move(str(temp_path), str(full_target_path))
        
        print("   Save complete.")

    except Exception as e:
        print(f"   ERROR: Could not save file. Exception: {e}")
        # Clean up temp file if it exists and failed
        if temp_path.exists():
            os.remove(temp_path)
        raise e