import os
import json
import shutil
import subprocess

# ======================================================================
# 1. PROJECT SETUP
# ======================================================================
project_path = os.path.abspath(".")   # current folder = VirtualTryOnWebApp
print(f"Setting up project at: {project_path}")

folders = [
    # Existing folders
    "data/raw",
    "data/processed",
    "data/landmarks",
    "models",
    "reports",
    "notebooks",
    "accessories/glasses/classic",
    "accessories/glasses/premium",
    "accessories/glasses/luxury",
    "accessories/glasses/sunglasses",
    "outputs/demo_images",
    "outputs/user_uploads",
    
    # NEW folders for FastAPI architecture
    "app",
    "core_logic",
    "training_pipeline"
]

for folder in folders:
    os.makedirs(os.path.join(project_path, folder), exist_ok=True)

print("Project folders created successfully!")

# ======================================================================
# 2. CREATE __init__.py FILES
# ======================================================================
init_files = [
    "app/__init__.py",
    "core_logic/__init__.py",
    "training_pipeline/__init__.py"
]

for init_file in init_files:
    init_path = os.path.join(project_path, init_file)
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write('"""\nPackage initialization\n"""\n')
        print(f"Created {init_file}")

# ======================================================================
# 3. CREATE METADATA.JSON
# ======================================================================
metadata_path = os.path.join(project_path, "accessories", "metadata.json")
if not os.path.exists(metadata_path):
    # Create sample metadata structure
    sample_metadata = {
        "classic_001": {
            "name": "Classic Round",
            "category": "classic",
            "suitable_for": ["oval", "square"],
            "score": 0.95,
            "path": "accessories/glasses/classic/001.png",
            "description": "Timeless round frames",
            "price": 99.99
        }
    }
    with open(metadata_path, "w") as f:
        json.dump(sample_metadata, f, indent=2)
    print("Created accessories/metadata.json with sample data")

# ======================================================================
# 4. KAGGLE API SETUP
# ======================================================================
kaggle_dir = os.path.join(project_path, ".kaggle") 
os.makedirs(kaggle_dir, exist_ok=True)

kaggle_json_source = r"C:\Users\PADELL09\Downloads\kaggle.json"
kaggle_json_dest = os.path.join(kaggle_dir, "kaggle.json")

if not os.path.exists(kaggle_json_dest):
    if os.path.exists(kaggle_json_source):
        shutil.copy(kaggle_json_source, kaggle_json_dest)
        print("Kaggle API credentials copied")
    else:
        print("Kaggle credentials not found at:", kaggle_json_source)
else:
    print("Kaggle API already configured")

os.environ["KAGGLE_CONFIG_DIR"] = kaggle_dir

# ======================================================================
# 5. SUMMARY
# ======================================================================
print("\n" + "="*60)
print("PROJECT SETUP COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Install requirements: pip install -r requirements.txt")
print("2. Add your app code to app/ directory")
print("3. Train your model: python training_pipeline/04_run_training.py")
print("4. Run API: python -m app.main")
print("="*60)