import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV
import sys
import os

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: LightGBM not found. Comparison will skip Boosting.")

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
    
    models = {}
    
    # 1. Voting (多数決) -> soft votingに変更して確率を出力可能にする
    voting_clf = VotingClassifier(
        estimators=[('svc', clf_svc), ('lr', clf_lr), ('rf', clf_rf)],
        voting='soft'
    )
    models['Voting'] = Pipeline([('features', feature_union), ('classifier', voting_clf)])
    
    # 2. Stacking (スタッキング)
    # final_estimator (メタ学習器) が各モデルの予測を統合する
    stacking_clf = StackingClassifier(
        estimators=[('svc', clf_svc), ('lr', clf_lr), ('rf', clf_rf)],
        final_estimator=LogisticRegression(random_state=config.RANDOM_STATE),
        cv=5
    )
    models['Stacking'] = Pipeline([('features', feature_union), ('classifier', stacking_clf)])
    
    # 3. Boosting (LightGBM)
    if HAS_LGBM:
        # LightGBMは日本語ラベルをそのまま扱えない等の場合があるが、
        # Pipeline内であればscikit-learnラッパーがいい感じに処理するか、
        # あるいは数値エンコードが必要な場合がある。
        # 今回はPipeline内でFeatureUnion後の疎行列/密行列を受け取るため、
        # そのままLGBMClassifierに渡す。
        lgbm_clf = lgb.LGBMClassifier(
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1,
            verbose=-1
        )
        models['Boosting(LightGBM)'] = Pipeline([('features', feature_union), ('classifier', lgbm_clf)])
    
    # ===========================
    # 比較学習と評価
    # ===========================
    print("\n=== Ensemble Comparison ===")
    best_name = 'Voting'
    best_score = 0
    best_pipeline = None
    
    results = []
    
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        print(f"  Accuracy: {score:.4f}")
        results.append((name, score))
        
        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipeline

    print("===========================\n")
    print(f"Best Model: {best_name} (Accuracy: {best_score:.4f})")
    
    # 結果を保存用に整形 (レポート出力用などに使えるようにprintしておく)
    print("Comparison Summary:")
    for name, score in results:
        print(f"{name}: {score:.4f}")
    
    # ===========================
    # ハイブリッド予測パイプラインの構築
    # ===========================
    # 最も良かったモデル（またはVoting）を最終的な成果物として返す
    # ここではユーザーの混乱を防ぐため、とりあえずVotingを返すか、Bestを返すか。
    # レポート目的もあるので、Bestを返してmetrics.txtに反映させるのが良さそう。
    
    print(f"Building hybrid prediction pipeline using {best_name}...")
    hybrid_pipeline = RuleBasedClassifier(
        ml_classifier=best_pipeline,
        auxiliary_verbs=config.AUXILIARY_VERBS
    )
    
    # RuleBasedClassifierのfit
    hybrid_pipeline.fit(X_train, y_train)
    
    return hybrid_pipeline, X_test, y_test
