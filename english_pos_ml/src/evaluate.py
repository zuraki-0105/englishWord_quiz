import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def evaluate_model(model, X_test, y_test):
    """
    学習済みモデルを評価し、メトリクスと混同行列を保存する関数
    
    処理の流れ:
    1. テストデータで予測を実行
    2. 評価指標の計算(Accuracy, Macro F1)
    3. 詳細な分類レポートの生成
    4. 結果をファイルに保存
    5. 混同行列の生成と保存
    
    Args:
        model (Pipeline): 学習済みのモデルパイプライン
        X_test (pd.Series): テスト用の特徴量(単語のスペル)
        y_test (pd.Series): テスト用の正解ラベル(品詞)
    
    Returns:
        None (結果はファイルに保存される)
    """
    print("Evaluating model...")
    
    # テストデータで予測を実行
    y_pred = model.predict(X_test)
    
    # ===========================
    # 評価指標の計算
    # ===========================
    # Accuracy(正解率): 全体のうち正しく分類できた割合
    acc = accuracy_score(y_test, y_pred)
    
    # Macro F1スコア: 各クラスのF1スコアの平均
    # クラス不均衡がある場合でも、各クラスを平等に評価できる
    # average='macro': 各クラスのF1を単純平均(サンプル数を考慮しない)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    
    # ===========================
    # 詳細な分類レポートの生成
    # ===========================
    # 各クラスごとのprecision, recall, f1-score, supportを含むレポート
    report = classification_report(y_test, y_pred)
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    
    # ===========================
    # メトリクスの保存
    # ===========================
    # 評価結果をテキストファイルに保存
    with open(config.METRICS_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Metrics saved to {config.METRICS_FILE}")
    
    # ===========================
    # 混同行列の生成と保存
    # ===========================
    # 混同行列: 実際のクラスと予測されたクラスの組み合わせをカウント
    # labels=model.classes_: モデルが学習したクラスの順序で行列を生成
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    # DataFrameに変換してCSVとして保存(行と列にクラス名を付与)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    cm_df.to_csv(config.CONFUSION_MATRIX_FILE)
    print(f"Confusion matrix saved to {config.CONFUSION_MATRIX_FILE}")
