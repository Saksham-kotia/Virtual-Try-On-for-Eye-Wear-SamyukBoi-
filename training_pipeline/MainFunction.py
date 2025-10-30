import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from training_pipeline.CompletePipeline import CompleteCNNPipeline

def main():
    """
    Complete CNN workflow for face shape classification
    """
    
    # Configuration
    project_path = PROJECT_ROOT
    dataset_name = "utkface-new.zip" 
    dataset_dir = os.path.join(project_path, "data", "raw")
    dataset_zip_path = os.path.join(dataset_dir, dataset_name)
    
    # Check if dataset exists
    if not os.path.exists(celeba_zip_path):
        print(f"Error: Dataset not found at {celeba_zip_path}")
        print("Please download 'celeba-dataset.zip' and place it in the 'data/raw' folder.")
        return

    # =================================================================
    # 1. KAGGLE DOWNLOAD LOGIC (MOVED FROM DataLoading)
    # =================================================================
    
    # Check if dataset exists, if not, download it
    if not os.path.exists(dataset_zip_path):
        print(f"Error: Dataset not found at {dataset_zip_path}")
        print("Attempting to download 'utkface-new' from Kaggle...")
        
        try:
            # Ensure the directory exists
            os.makedirs(dataset_dir, exist_ok=True)
            
            os.system(f'kaggle datasets download -d jangedoo/utkface-new -p {dataset_dir} --force --unzip')
            
            print("Downloading 'utkface-new' as a zip file...")
            os.system(f'kaggle datasets download -d jangedoo/utkface-new -p {dataset_dir} --force')
            
            if not os.path.exists(dataset_zip_path):
                raise FileNotFoundError("Kaggle download failed. Is your kaggle.json API key set up?")
            print("Download successful.")
            
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please ensure your Kaggle API key is correctly set up in '.kaggle/kaggle.json'")
            return

    # Initialize CNN pipeline
    cnn_pipeline = CompleteCNNPipeline(project_path)
    
    # Run complete pipeline
    print("Starting CNN-based Face Shape Classification")
    print("This will:")
    print("1. Extract and preprocess images in batches")
    print("2. Train a CNN model for face shape classification")
    print("3. Evaluate model performance")
    print("4. Test landmark integration for overlay engine")
    print("5. Save everything for production use")
    
    # Start with smaller sample for development
    trained_model, results = cnn_pipeline.run_complete_pipeline(
        zip_path=dataset_zip_path,
        sample_size=1000  # Start small, scale up
    )
    
    # Plot training results
    trained_model.plot_training_history()
    
    print("\nCNN Pipeline Complete!")
    print("   Next steps:")
    print("   1. Integrate with recommendation engine")
    print("   2. Connect to overlay system")
    print("   3. Build web API endpoints")

# Run the pipeline
if __name__ == "__main__":
    main()

print("Complete CNN-based Face Shape Classification Pipeline Ready!")
print("This gives you proper TensorFlow/Keras CNN training!")