import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.feature_engineering import LinguisticFeatureExtractor
from src.rule_based_classifier import RuleBasedClassifier

def train_model(df):
    """
    提供されたDataFrameを使用してモデルを学習する関数
    
    処理の流れ:
    1. サンプル数が少ないクラスのフィルタリング(stratified splitに必要)
    2. データの分割(訓練用とテスト用)
    3. パイプラインの構築(TF-IDF + 言語的特徴量 + 分類器)
    4. モデルの学習
    
    Args:
        df (pd.DataFrame): 学習用データフレーム('spell'と'pos'カラムを含む)
    
    Returns:
        tuple: (学習済みパイプライン, テスト特徴量, テスト正解ラベル)
            - pipeline (Pipeline): TF-IDF + 言語的特徴量 + 分類器の学習済みパイプライン
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
    print("Initializing pipeline with enhanced features...")
    
    # TF-IDF Vectorizer: 文字n-gramを特徴量に変換
    # analyzer='char': 文字レベルで解析(単語レベルではない)
    # ngram_range: 2文字から5文字までの連続した文字列を特徴量として抽出
    #   例: "running" → "ru", "un", "nn", "ni", "ing", "run", "unn", "nni", "ning", ...
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=config.NGRAM_RANGE
    )
    
    # 言語的特徴量抽出器
    # 接尾辞・接頭辞パターン、単語長、文字比率などを抽出
    linguistic_extractor = LinguisticFeatureExtractor()
    
    # ===========================
    # FeatureUnionで特徴量を結合
    # ===========================
    # 複数の特徴抽出器を並列に実行し、結果を結合
    # 1. TF-IDF文字n-gram (スパース行列)
    # 2. 言語的特徴量 (密行列)
    feature_union = FeatureUnion([
        ('tfidf', tfidf_vectorizer),           # TF-IDF特徴量
        ('linguistic', linguistic_extractor),   # 言語的特徴量
    ])
    
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
        dual='auto',                      # 最新のscikit-learnの警告を抑制
        max_iter=2000                     # 特徴量が増えたため反復回数を増加
    )
    
    # パイプラインの作成: 特徴量抽出 → 分類を一連の流れとして定義
    ml_pipeline = Pipeline([
        ('features', feature_union),  # ステップ1: TF-IDF + 言語的特徴量を抽出・結合
        ('classifier', classifier)    # ステップ2: 特徴量から品詞を分類
    ])
    
    # ===========================
    # モデルの学習
    # ===========================
    print("Training model with TF-IDF + Linguistic features...")
    # パイプライン全体を学習(特徴抽出と分類を同時に実行)
    ml_pipeline.fit(X_train, y_train)
    print("Training completed.")
    
    # ===========================
    # ハイブリッド予測パイプラインの構築
    # ===========================
    print("Building hybrid prediction pipeline with rule-based classifier...")
    # 学習済みMLパイプラインをRuleBasedClassifierでラップ
    # これにより、助動詞はルールベースで、その他はMLモデルで予測される
    hybrid_pipeline = RuleBasedClassifier(
        ml_classifier=ml_pipeline,
        auxiliary_verbs=config.AUXILIARY_VERBS
    )
    
    # RuleBasedClassifierのfit()を呼び出してclasses_を設定
    # （実際の学習は既に完了しているため、形式的な呼び出し）
    hybrid_pipeline.fit(X_train, y_train)
    print("Hybrid pipeline ready.")
    
    return hybrid_pipeline, X_test, y_test
