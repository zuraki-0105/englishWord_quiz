import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import label_binarize
import os
import sys

# Windowsでの日本語文字化け対策
plt.rcParams['font.family'] = 'MS Gothic'

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
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    cm_df.to_csv(config.CONFUSION_MATRIX_FILE)
    print(f"Confusion matrix saved to {config.CONFUSION_MATRIX_FILE}")

    # ===========================
    # PR曲線・ROC曲線の生成と保存
    # ===========================
    try:
        plot_pr_roc_curves(model, X_test, y_test)
    except Exception as e:
        print(f"Error plotting curves: {e}")

def plot_pr_roc_curves(model, X_test, y_test):
    """
    PR曲線とROC曲線を作成・保存する
    
    Args:
        model: 学習済みモデル（predict_probaを実装していること）
        X_test: テスト特徴量
        y_test: テスト正解ラベル
    """
    print("Generating PR and ROC curves...")
    
    # 確率の取得
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        print("Model does not support predict_proba. Skipping curve generation.")
        return

    # クラスの準備
    classes = model.classes_
    n_classes = len(classes)
    
    # ラベルをBinarize (One-vs-Restのため)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # 出力ディレクトリ
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    
    # ---------------------------
    # PR Curve
    # ---------------------------
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{classes[i]} (AUC = {pr_auc:.2f})')
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True)
    
    pr_path = os.path.join(config.OUTPUTS_DIR, "pr_curve.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"PR curve saved to {pr_path}")

    # ---------------------------
    # ROC Curve
    # ---------------------------
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    roc_path = os.path.join(config.OUTPUTS_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved to {roc_path}")

