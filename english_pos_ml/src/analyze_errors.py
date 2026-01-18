import pandas as pd
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data_loader

def analyze_errors():
    """
    Analyze prediction errors and save to CSV
    """
    print("Loading data...")
    try:
        df = data_loader.load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Loading model...")
    try:
        pipeline = joblib.load(os.path.join(config.MODELS_DIR, 'pipeline.joblib'))
    except Exception as e:
        try:
             # Fallback to older model structure if pipeline.joblib doesn't exist
            print("Pipeline not found, trying to load components separately...")
            # This part is tricky if the pipeline structure changed. 
            # Ideally we should use the same training logic to get the split.
            # But let's just assume we want to evaluate on the whole dataset or 
            # re-split. To be consistent with evaluation, let's use the train_test_split from train.py logic
            # OR better, just use the helper in train.py if importable.
            # However, for simplicity here, let's just re-run the split logic.
            pass
        except Exception as e2:
            print(f"Error loading model: {e}")
            return

    # To ensure we look at the TEST set errors specifically (or validation),
    # we should replicate the split logic.
    from src.train import train_model
    # We can't easily get the *exact* user split unless we set random state.
    # config.RANDOM_STATE should handle this.
    
    from sklearn.model_selection import train_test_split
    
    # Rare class filtering (duplicating logic from train.py)
    class_counts = df['pos'].value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    if rare_classes:
        df = df[~df['pos'].isin(rare_classes)]
        
    X = df['spell']
    y = df['pos']
    
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        stratify=y, 
        random_state=config.RANDOM_STATE
    )
    
    print(f"Predicting on {len(X_test)} test samples...")
    y_pred = pipeline.predict(X_test)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Word': X_test,
        'True_Label': y_test,
        'Predicted_Label': y_pred
    })
    
    # Filter for errors
    errors_df = results_df[results_df['True_Label'] != results_df['Predicted_Label']]
    
    output_file = os.path.join(config.OUTPUTS_DIR, 'error_analysis.csv')
    errors_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\nFound {len(errors_df)} errors out of {len(X_test)} samples.")
    print(f"Error Rate: {len(errors_df)/len(X_test):.2%}")
    print(f"Errors saved to: {output_file}")
    
    # Summary of errors by True Label
    print("\nError Count by True Label:")
    print(errors_df['True_Label'].value_counts())
    
    # Summary of specific confusion (True -> Pred)
    print("\nTop 10 Confusion Patterns:")
    confusion = errors_df.groupby(['True_Label', 'Predicted_Label']).size().sort_values(ascending=False).head(10)
    print(confusion)

if __name__ == "__main__":
    analyze_errors()
