import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.feature_engineering import LinguisticFeatureExtractor
from src.rule_based_classifier import RuleBasedClassifier

def train_model(df):
    """
    提供されたDataFrameを使用して複数のアンサンブルモデルを比較学習する関数
    
    Args:
        df (pd.DataFrame): 学習用データフレーム
    
    Returns:
        tuple: (学習済みVotingパイプライン, テスト特徴量, テスト正解ラベル)
        ※メインの戻り値はこれまで通りVotingモデルとしますが、
        内部で比較結果を表示します。
    """
    print("Splitting data...")
    
    # 少数クラスのフィルタリング
    class_counts = df['pos'].value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    
    if rare_classes:
        print(f"Warning: Dropping {len(rare_classes)} class(es) with < 2 samples: {rare_classes}")
        df = df[~df['pos'].isin(rare_classes)]
        print(f"Remaining samples: {len(df)}")
    
    X = df['spell']
    y = df['pos']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE,      
        stratify=y,                       
        random_state=config.RANDOM_STATE  
    )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # 共通の特徴量抽出パイプライン
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=config.NGRAM_RANGE
    )
    
    linguistic_extractor = LinguisticFeatureExtractor()
    
    feature_union = FeatureUnion([
        ('tfidf', tfidf_vectorizer),           
        ('linguistic', linguistic_extractor),   
    ])
    
    # ===========================
    # モデル定義
    # ===========================
    
    # ベースモデル群
    # LinearSVCは確率出力を持たないので、CalibratedClassifierCVでラップする
    svc = LinearSVC(class_weight='balanced', random_state=config.RANDOM_STATE, dual='auto', max_iter=3000)
    clf_svc = CalibratedClassifierCV(svc, cv=5)
    
    clf_lr = LogisticRegression(class_weight='balanced', random_state=config.RANDOM_STATE, max_iter=2000)
    clf_rf = RandomForestClassifier(class_weight='balanced', random_state=config.RANDOM_STATE, n_estimators=200)
    
    # 追加モデル
    clf_nb = MultinomialNB()
    clf_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=config.RANDOM_STATE)
    
    # SGDClassifierは確率出力のためにloss='log_loss'またはCalibratedClassifierCVが必要
    sgd = SGDClassifier(loss='log_loss', class_weight='balanced', random_state=config.RANDOM_STATE)
    clf_sgd = sgd
    
    clf_knn = KNeighborsClassifier(n_neighbors=5)
    
    # Extra Treesも追加（多様性のため）
    clf_et = ExtraTreesClassifier(class_weight='balanced', random_state=config.RANDOM_STATE, n_estimators=200)
    
    # Voting (多数決) -> soft voting
    voting_clf = VotingClassifier(
        estimators=[
            ('svc', clf_svc), 
            ('lr', clf_lr), 
            ('rf', clf_rf),
            ('nb', clf_nb),
            ('mlp', clf_mlp),
            ('sgd', clf_sgd),
            ('knn', clf_knn),
            ('et', clf_et)
        ],
        voting='soft'
    )
    
    final_pipeline = Pipeline([('features', feature_union), ('classifier', voting_clf)])
    
    print("\nTraining Ensemble (Voting)...")
    final_pipeline.fit(X_train, y_train)
    
    score = final_pipeline.score(X_test, y_test)
    print(f"Ensemble Accuracy: {score:.4f}")
    
    # ===========================
    # ハイブリッド予測パイプラインの構築
    # ===========================
    # 最も良かったモデル（またはVoting）を最終的な成果物として返す
    # ここではユーザーの混乱を防ぐため、とりあえずVotingを返すか、Bestを返すか。
    # レポート目的もあるので、Bestを返してmetrics.txtに反映させるのが良さそう。
    
    print("Building hybrid prediction pipeline...")
    hybrid_pipeline = RuleBasedClassifier(
        ml_classifier=final_pipeline,
        auxiliary_verbs=config.AUXILIARY_VERBS
    )
    
    # RuleBasedClassifierのfit
    hybrid_pipeline.fit(X_train, y_train)
    
    return hybrid_pipeline, X_test, y_test
