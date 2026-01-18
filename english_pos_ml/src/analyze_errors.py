import pandas as pd
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data_loader

def analyze_errors(pipeline=None, X_test=None, y_test=None):
    """
    Analyze prediction errors and save to CSV
    
    Args:
        pipeline: Trained model pipeline (optional)
        X_test: Test features (optional)
        y_test: Test labels (optional)
    """
    if pipeline is not None and X_test is not None and y_test is not None:
        # Use provided data and model
        pass
    else:
        # Load data and model from scratch if not provided
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
            print(f"Error loading model: {e}")
            return

        # Rare class filtering (duplicating logic from train.py)
        class_counts = df['pos'].value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        if rare_classes:
            df = df[~df['pos'].isin(rare_classes)]
            
        X = df['spell']
        y = df['pos']
        
        from sklearn.model_selection import train_test_split
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
