import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def train_model(df):
    """
    提供されたDataFrameを使用してモデルを学習する関数
    
    処理の流れ:
    1. サンプル数が少ないクラスのフィルタリング(stratified splitに必要)
    2. データの分割(訓練用とテスト用)
    3. パイプラインの構築(TF-IDF + 分類器)
    4. モデルの学習
    
    Args:
        df (pd.DataFrame): 学習用データフレーム('spell'と'pos'カラムを含む)
    
    Returns:
        tuple: (学習済みパイプライン, テスト特徴量, テスト正解ラベル)
            - pipeline (Pipeline): TF-IDF + 分類器の学習済みパイプライン
            - X_test (pd.Series): テスト用の特徴量(単語のスペル)
            - y_test (pd.Series): テスト用の正解ラベル(品詞)
    """
    print("Splitting data...")
    
    # ===========================
    # 少数クラスのフィルタリング
    # ===========================
    # stratified split(層化分割)を行うには、各クラスに最低2サンプル必要
    # サンプル数が1のクラスがあると、train_test_splitでエラーが発生するため事前に除外
    class_counts = df['pos'].value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    
    if rare_classes:
        print(f"Warning: Dropping {len(rare_classes)} class(es) with < 2 samples: {rare_classes}")
        # 少数クラスに属するサンプルを除外
        df = df[~df['pos'].isin(rare_classes)]
        print(f"Remaining samples: {len(df)}")
    
    # 特徴量(X)とラベル(y)の分離
    X = df['spell']  # 単語のスペル
    y = df['pos']    # 品詞ラベル
    
    # ===========================
    # データの分割
    # ===========================
    # stratify=y: 各クラスの比率を訓練データとテストデータで同じにする
    # random_state: 再現性を確保するための乱数シード
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE,      # テストデータの割合(0.2 = 20%)
        stratify=y,                       # クラス比率を保持
        random_state=config.RANDOM_STATE  # 乱数シード
    )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # ===========================
    # パイプラインの構築
    # ===========================
    print("Initializing pipeline...")
    
    # TF-IDF Vectorizer: 文字n-gramを特徴量に変換
    # analyzer='char': 文字レベルで解析(単語レベルではない)
    # ngram_range: 2文字から5文字までの連続した文字列を特徴量として抽出
    #   例: "running" → "ru", "un", "nn", "ni", "ing", "run", "unn", "nni", "ning", ...
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=config.NGRAM_RANGE
    )
    
    # ===========================
    # 分類器の選択
    # ===========================
    # LinearSVC: 線形サポートベクターマシン
    # - テキスト分類タスクで高い性能を発揮
    # - 高次元のスパースな特徴量(TF-IDF)に適している
    # - LogisticRegressionも候補だが、LinearSVCの方が一般的に高精度
    # 
    # 注意: LinearSVCはpredict_proba()をサポートしていない
    # 確率が必要な場合は、LogisticRegressionまたはCalibratedClassifierCVを使用
    classifier = LinearSVC(
        class_weight='balanced',          # クラス不均衡に対応(少数クラスの重みを増やす)
        random_state=config.RANDOM_STATE, # 再現性の確保
        dual='auto'                       # 最新のscikit-learnの警告を抑制
    )
    
    # パイプラインの作成: ベクトル化 → 分類を一連の流れとして定義
    pipeline = Pipeline([
        ('vectorizer', vectorizer),  # ステップ1: テキストをTF-IDF特徴量に変換
        ('classifier', classifier)   # ステップ2: 特徴量から品詞を分類
    ])
    
    # ===========================
    # モデルの学習
    # ===========================
    print("Training model...")
    # パイプライン全体を学習(ベクトル化と分類を同時に実行)
    pipeline.fit(X_train, y_train)
    print("Training completed.")
    
    return pipeline, X_test, y_test
