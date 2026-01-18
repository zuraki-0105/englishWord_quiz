import os
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# プロジェクトルートとconfigの読み込み
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def get_wordnet_features(word):
    """
    WordNetを使用して単語の品詞可能性を取得する
    """
    # WordNetを使用しない場合は、全て0の特徴量を返す
    if not config.USE_WORDNET:
        return {'wn_noun': 0, 'wn_verb': 0, 'wn_adj': 0, 'wn_adv': 0}

    try:
        # ここでインポートすることで、USE_WORDNET=Falseの時にNLTKへの依存をなくす
        from nltk.corpus import wordnet
        synsets = wordnet.synsets(word)
    except (ImportError, LookupError):
        # ライブラリがない、またはデータがない場合も0を返す
        return {'wn_noun': 0, 'wn_verb': 0, 'wn_adj': 0, 'wn_adv': 0}
        
    features = {
        'wn_noun': 0,
        'wn_verb': 0,
        'wn_adj': 0,
        'wn_adv': 0
    }
    
    if not synsets:
        return features
        
    for syn in synsets:
        pos = syn.pos()
        if pos == 'n':
            features['wn_noun'] = 1
        elif pos == 'v':
            features['wn_verb'] = 1
        elif pos == 'a' or pos == 's':
            features['wn_adj'] = 1
        elif pos == 'r':
            features['wn_adv'] = 1
            
    return features

def extract_linguistic_features(word):
    """
    単語から言語的特徴を抽出する関数
    
    品詞判定に有用な特徴:
    - 接尾辞パターン: -ing, -ed, -ly, -tion, -ness, -ful, -ous, -ive, -er, -est など
    - 接頭辞パターン: un-, re-, pre-, dis-, mis- など
    - WordNet特徴: 辞書上の品詞定義
    - 単語の長さ
    - 文字比率: 母音/子音の比率
    - 特殊文字: ハイフンの有無
    
    Args:
        word (str): 英単語のスペル
    
    Returns:
        dict: 特徴量の辞書
    """
    word = str(word).lower().strip()
    
    # ===========================
    # 接尾辞パターン (Suffix Patterns)
    # ===========================
    # 動詞に多い接尾辞
    suffix_ing = 1 if word.endswith('ing') else 0
    suffix_ed = 1 if word.endswith('ed') else 0
    suffix_en = 1 if word.endswith('en') else 0
    suffix_ize = 1 if word.endswith('ize') or word.endswith('ise') else 0
    suffix_ate = 1 if word.endswith('ate') else 0
    suffix_ify = 1 if word.endswith('ify') else 0
    
    # 名詞に多い接尾辞
    suffix_tion = 1 if word.endswith('tion') or word.endswith('sion') else 0
    suffix_ness = 1 if word.endswith('ness') else 0
    suffix_ment = 1 if word.endswith('ment') else 0
    suffix_ity = 1 if word.endswith('ity') else 0
    suffix_er_noun = 1 if word.endswith('er') and len(word) > 3 else 0
    suffix_or = 1 if word.endswith('or') else 0
    suffix_ism = 1 if word.endswith('ism') else 0
    suffix_ist = 1 if word.endswith('ist') else 0
    suffix_ance = 1 if word.endswith('ance') or word.endswith('ence') else 0
    suffix_age = 1 if word.endswith('age') else 0
    
    # 形容詞に多い接尾辞
    suffix_ful = 1 if word.endswith('ful') else 0
    suffix_ous = 1 if word.endswith('ous') else 0
    suffix_ive = 1 if word.endswith('ive') else 0
    suffix_able = 1 if word.endswith('able') or word.endswith('ible') else 0
    suffix_al = 1 if word.endswith('al') else 0
    suffix_ic = 1 if word.endswith('ic') else 0
    suffix_less = 1 if word.endswith('less') else 0
    suffix_y = 1 if word.endswith('y') else 0
    suffix_ent = 1 if word.endswith('ent') else 0
    suffix_ant = 1 if word.endswith('ant') else 0
    suffix_ish = 1 if word.endswith('ish') else 0
    suffix_ary = 1 if word.endswith('ary') else 0
    
    # 副詞に多い接尾辞
    suffix_ly = 1 if word.endswith('ly') else 0
    suffix_ward = 1 if word.endswith('ward') or word.endswith('wards') else 0
    
    # 比較級・最上級
    suffix_er_comp = 1 if word.endswith('er') and len(word) <= 6 else 0
    suffix_est = 1 if word.endswith('est') else 0
    
    # ===========================
    # 接頭辞パターン (Prefix Patterns)
    # ===========================
    prefix_un = 1 if word.startswith('un') else 0
    prefix_re = 1 if word.startswith('re') else 0
    prefix_pre = 1 if word.startswith('pre') else 0
    prefix_dis = 1 if word.startswith('dis') else 0
    prefix_mis = 1 if word.startswith('mis') else 0
    prefix_in = 1 if word.startswith('in') or word.startswith('im') else 0
    prefix_over = 1 if word.startswith('over') else 0
    prefix_under = 1 if word.startswith('under') else 0
    
    # ===========================
    # WordNet 特徴
    # ===========================
    wn_features = get_wordnet_features(word)
    
    # ===========================
    # 単語の長さと構造
    # ===========================
    word_length = len(word)
    
    # 母音の数と比率
    vowels = 'aeiou'
    vowel_count = sum(1 for c in word if c in vowels)
    vowel_ratio = vowel_count / len(word) if len(word) > 0 else 0
    
    # 子音の数と比率
    consonant_count = sum(1 for c in word if c.isalpha() and c not in vowels)
    consonant_ratio = consonant_count / len(word) if len(word) > 0 else 0
    
    # 特殊文字
    has_hyphen = 1 if '-' in word else 0
    has_apostrophe = 1 if "'" in word else 0
    
    # 連続する同じ文字のパターン
    has_double_letter = 1 if any(word[i] == word[i+1] for i in range(len(word)-1)) else 0
    
    # ===========================
    # 特徴量辞書の作成
    # ===========================
    features = {
        # 接尾辞 - 動詞
        'suffix_ing': suffix_ing,
        'suffix_ed': suffix_ed,
        'suffix_en': suffix_en,
        'suffix_ize': suffix_ize,
        'suffix_ate': suffix_ate,
        'suffix_ify': suffix_ify,
        
        # 接尾辞 - 名詞
        'suffix_tion': suffix_tion,
        'suffix_ness': suffix_ness,
        'suffix_ment': suffix_ment,
        'suffix_ity': suffix_ity,
        'suffix_er_noun': suffix_er_noun,
        'suffix_or': suffix_or,
        'suffix_ism': suffix_ism,
        'suffix_ist': suffix_ist,
        'suffix_ance': suffix_ance,
        'suffix_age': suffix_age,
        
        # 接尾辞 - 形容詞
        'suffix_ful': suffix_ful,
        'suffix_ous': suffix_ous,
        'suffix_ive': suffix_ive,
        'suffix_able': suffix_able,
        'suffix_al': suffix_al,
        'suffix_ic': suffix_ic,
        'suffix_less': suffix_less,
        'suffix_y': suffix_y,
        'suffix_ent': suffix_ent,
        'suffix_ant': suffix_ant,
        'suffix_ish': suffix_ish,
        'suffix_ary': suffix_ary,
        
        # 接尾辞 - 副詞
        'suffix_ly': suffix_ly,
        'suffix_ward': suffix_ward,
        
        # 接尾辞 - 比較級・最上級
        'suffix_er_comp': suffix_er_comp,
        'suffix_est': suffix_est,
        
        # 接頭辞
        'prefix_un': prefix_un,
        'prefix_re': prefix_re,
        'prefix_pre': prefix_pre,
        'prefix_dis': prefix_dis,
        'prefix_mis': prefix_mis,
        'prefix_in': prefix_in,
        'prefix_over': prefix_over,
        'prefix_under': prefix_under,
        
        # WordNet
        'wn_noun': wn_features['wn_noun'],
        'wn_verb': wn_features['wn_verb'],
        'wn_adj': wn_features['wn_adj'],
        'wn_adv': wn_features['wn_adv'],
        
        # 単語の構造
        'word_length': word_length,
        'vowel_count': vowel_count,
        'vowel_ratio': vowel_ratio,
        'consonant_count': consonant_count,
        'consonant_ratio': consonant_ratio,
        'has_hyphen': has_hyphen,
        'has_apostrophe': has_apostrophe,
        'has_double_letter': has_double_letter,
    }
    
    return features


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    scikit-learn互換の言語的特徴抽出Transformer
    
    Pipelineで使用できるように、fit()とtransform()メソッドを実装
    """
    
    def __init__(self):
        """初期化"""
        pass
    
    def fit(self, X, y=None):
        """
        学習フェーズ(このTransformerは学習不要)
        
        Args:
            X: 入力データ(pd.Series or array-like)
            y: ラベル(未使用)
        
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """
        特徴量抽出の実行
        
        Args:
            X: 入力データ(pd.Series or array-like)
        
        Returns:
            np.ndarray: 抽出された特徴量の配列
        """
        # 各単語から特徴を抽出
        features_list = [extract_linguistic_features(word) for word in X]
        
        # DataFrameに変換
        features_df = pd.DataFrame(features_list)
        
        # NumPy配列として返す
        return features_df.values
    
    def get_feature_names_out(self, input_features=None):
        """
        特徴量の名前を取得(scikit-learn 1.0+で必要)
        
        Returns:
            list: 特徴量名のリスト
        """
        # サンプル単語で特徴を抽出して、カラム名を取得
        sample_features = extract_linguistic_features('sample')
        return list(sample_features.keys())
