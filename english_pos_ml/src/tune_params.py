import pandas as pd
import sys
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.feature_engineering import LinguisticFeatureExtractor

def tune_parameters():
    """
    Perform GridSearchCV to find optimal hyperparameters.
    """
    print("Loading data...")
    df = pd.read_csv(config.DATA_FILE)
    
    # Preprocessing (same as train.py)
    # Check if necessary columns exist
    if 'word' in df.columns and 'spell' not in df.columns:
        df = df.rename(columns={'word': 'spell'})
    if 'pos' in df.columns and 'part_of_speech' in df.columns:
         # Use pos if available, but checking for variations
        pass 
        
    # Standardize column names using config
    for unified_name, potential_names in config.COLUMN_MAPPING.items():
        for name in potential_names:
            if name in df.columns:
                df = df.rename(columns={name: unified_name})
                break
    
    # Filter rare classes
    class_counts = df['pos'].value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    if rare_classes:
        df = df[~df['pos'].isin(rare_classes)]
    
    X = df['spell']
    y = df['pos']
    
    # Split data (use training set for tuning)
    X_train, _, y_train, _ = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        stratify=y, 
        random_state=config.RANDOM_STATE
    )
    
    # Define Pipeline
    # Using LogisticRegression for tuning as it is faster and supports probabilities which might be useful later,
    # but we can try tuning LinearSVC or even the VotingClassifier itself (too complex/slow).
    # Let's tune individual components or a representative one. The plan mentioned tuning SVC/LogReg/RF.
    # Let's try to tune a simple pipeline first to find best features params (ngram) and C.
    
    feature_union = FeatureUnion([
        ('tfidf', TfidfVectorizer(analyzer='char')),           
        ('linguistic', LinguisticFeatureExtractor()),   
    ])
    
    pipeline = Pipeline([
        ('features', feature_union),
        ('clf', LinearSVC(random_state=config.RANDOM_STATE, dual='auto'))
    ])
    
    # Parameter Grid
    param_grid = {
        'features__tfidf__ngram_range': [(1, 3), (2, 5), (3, 6)],
        'clf__C': [0.1, 1, 10],
    }
    
    print("Starting GridSearchCV (this may take a while)...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        verbose=1, 
        n_jobs=-1,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n" + "="*30)
    print("Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print("="*30)
    
    return grid_search.best_params_

if __name__ == "__main__":
    tune_parameters()
