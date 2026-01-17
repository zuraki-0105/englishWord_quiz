import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RuleBasedClassifier(BaseEstimator, ClassifierMixin):
    """
    ルールベースと機械学習を組み合わせたハイブリッド分類器
    
    処理の流れ:
    1. 入力単語が助動詞リストに含まれているかチェック
    2. 含まれていれば「助動詞」として分類
    3. 含まれていなければ機械学習モデルで予測
    
    これにより、助動詞のような限定的な語彙を持つ品詞を確実に分類できます。
    
    Attributes:
        ml_classifier: 機械学習モデル（scikit-learn互換の分類器）
        auxiliary_verbs (set): 助動詞の固定リスト
        classes_ (np.ndarray): 分類可能なクラスのリスト（scikit-learn互換性のため）
    """
    
    def __init__(self, ml_classifier, auxiliary_verbs):
        """
        初期化
        
        Args:
            ml_classifier: 機械学習モデル（既に学習済みのパイプライン）
            auxiliary_verbs (set): 助動詞の固定リスト
        """
        self.ml_classifier = ml_classifier
        self.auxiliary_verbs = auxiliary_verbs
        self.classes_ = None
    
    def fit(self, X, y):
        """
        学習フェーズ
        
        注意: このクラスは既に学習済みのMLモデルをラップするため、
        実際の学習処理は行いません。classes_属性のみ設定します。
        
        Args:
            X: 特徴量（未使用）
            y: ラベル（classes_の設定に使用）
        
        Returns:
            self
        """
        # scikit-learn互換性のため、classes_属性を設定
        if hasattr(self.ml_classifier, 'classes_'):
            self.classes_ = self.ml_classifier.classes_
        else:
            # MLモデルがまだ学習されていない場合、yから推定
            self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X):
        """
        予測フェーズ
        
        各単語に対して:
        1. 助動詞リストに含まれているかチェック
        2. 含まれていれば「助動詞」を返す
        3. 含まれていなければMLモデルで予測
        
        Args:
            X: 入力データ（単語のリスト、pd.Series、またはarray-like）
        
        Returns:
            np.ndarray: 予測された品詞のリスト
        """
        predictions = []
        
        for word in X:
            # 単語を小文字化して正規化
            word_lower = str(word).lower().strip()
            
            # ルールベース判定: 助動詞リストに含まれているか
            if word_lower in self.auxiliary_verbs:
                predictions.append('助動詞')
            else:
                # MLモデルで予測（単一の単語を予測するため、リストでラップ）
                ml_prediction = self.ml_classifier.predict([word])
                predictions.append(ml_prediction[0])
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        確率予測（オプション）
        
        注意: LinearSVCはpredict_proba()をサポートしていないため、
        このメソッドは実装していません。必要な場合は、
        CalibratedClassifierCVでMLモデルをラップしてください。
        
        Args:
            X: 入力データ
        
        Raises:
            NotImplementedError: このメソッドは未実装
        """
        raise NotImplementedError(
            "predict_proba is not supported. "
            "Use CalibratedClassifierCV if probability estimates are needed."
        )
    
    def get_params(self, deep=True):
        """
        パラメータの取得（scikit-learn互換性のため）
        
        Args:
            deep (bool): ネストされたオブジェクトのパラメータも取得するか
        
        Returns:
            dict: パラメータの辞書
        """
        return {
            'ml_classifier': self.ml_classifier,
            'auxiliary_verbs': self.auxiliary_verbs
        }
    
    def set_params(self, **params):
        """
        パラメータの設定（scikit-learn互換性のため）
        
        Args:
            **params: 設定するパラメータ
        
        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
