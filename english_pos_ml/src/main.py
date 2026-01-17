import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data_loader, train, evaluate

def main():
    """
    英単語品詞分類プロジェクトのメイン処理
    
    処理の流れ:
    1. データの読み込みと前処理
    2. モデルの学習
    3. モデルの評価
    4. 学習済みモデルとベクトライザの保存
    """
    print("=== English POS Classification Project ===")
    
    # ===========================
    # 1. データの読み込み
    # ===========================
    print("\n--- Loading Data ---")
    try:
        df = data_loader.load_data()
    except FileNotFoundError as e:
        # データファイルが見つからない場合のエラーハンドリング
        print(f"Error: {e}")
        print(f"Please place input CSV file at: {config.DATA_FILE}")
        return
    except Exception as e:
        # その他の予期しないエラー
        print(f"Unexpected error loading data: {e}")
        return
    
    # ===========================
    # 2. モデルの学習
    # ===========================
    print("\n--- Training Model ---")
    # train_model()は学習済みパイプライン、テスト特徴量、テスト正解ラベルを返す
    pipeline, X_test, y_test = train.train_model(df)
    
    # ===========================
    # 3. モデルの評価
    # ===========================
    print("\n--- Evaluating Model ---")
    # テストデータでモデルを評価し、結果をファイルに保存
    evaluate.evaluate_model(pipeline, X_test, y_test)
    
    # ===========================
    # 4. 学習済みモデルの保存
    # ===========================
    print("\n--- Saving Artifacts ---")
    # モデル保存ディレクトリが存在しない場合は作成
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # ハイブリッドパイプライン全体を保存
    # RuleBasedClassifier + MLパイプライン(FeatureUnion + LinearSVC)
    # 注意: 新しいハイブリッド構造では、パイプライン全体を1つのファイルとして保存
    
    # パイプライン全体の保存（推奨）
    pipeline_file = os.path.join(config.MODELS_DIR, 'pipeline.joblib')
    joblib.dump(pipeline, pipeline_file)
    print(f"Complete hybrid pipeline saved to {pipeline_file}")
    
    # 後方互換性のため、内部のMLパイプラインと分類器も保存
    ml_pipeline = pipeline.ml_classifier
    joblib.dump(ml_pipeline, config.VECTORIZER_FILE)
    print(f"ML pipeline saved to {config.VECTORIZER_FILE}")
    
    classifier = ml_pipeline['classifier']
    joblib.dump(classifier, config.MODEL_FILE)
    print(f"Classifier saved to {config.MODEL_FILE}")
    
    print("\n=== Process Completed Successfully ===")

if __name__ == "__main__":
    # このファイルが直接実行された場合のみmain()を実行
    main()
