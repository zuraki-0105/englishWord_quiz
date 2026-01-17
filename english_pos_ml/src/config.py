import os

# ===========================
# プロジェクトルートディレクトリの設定
# ===========================
# このファイル(config.py)の2階層上のディレクトリをプロジェクトルートとして設定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===========================
# ディレクトリパスの定義
# ===========================
# データファイルを格納するディレクトリ
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# 学習済みモデルを保存するディレクトリ
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# 評価結果(メトリクス、混同行列など)を保存するディレクトリ
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# ===========================
# ファイルパスの定義
# ===========================
# 入力データ: 英単語とその品詞情報を含むCSVファイル
# englishWord_data_added.csv / englishWord_data_augmented.csv / englishWord_data.csv
DATA_FILE = os.path.join(DATA_DIR, 'englishWord_data_added.csv')

# 学習済み分類モデルの保存先
MODEL_FILE = os.path.join(MODELS_DIR, 'pos_model.joblib')

# 学習済みベクトライザ(TF-IDF)の保存先
VECTORIZER_FILE = os.path.join(MODELS_DIR, 'vectorizer.joblib')

# 評価メトリクス(Accuracy, F1スコアなど)の保存先
METRICS_FILE = os.path.join(OUTPUTS_DIR, 'metrics.txt')

# 混同行列(Confusion Matrix)の保存先
CONFUSION_MATRIX_FILE = os.path.join(OUTPUTS_DIR, 'confusion_matrix.csv')

# ===========================
# CSVカラム名のマッピング
# ===========================
# CSVファイルのカラム名が異なる場合に対応するためのマッピング辞書
# キー: 内部で使用する統一されたカラム名
# 値: CSVファイルで使用される可能性のあるカラム名のリスト
COLUMN_MAPPING = {
    'spell': ['spell', 'スペル', 'word', '単語'],  # 英単語のスペル
    'pos': ['pos', '品詞', 'part_of_speech', 'label', 'ラベル']  # 品詞ラベル
}

# ===========================
# モデル学習パラメータ
# ===========================
# 乱数シード: 再現性を確保するための固定値
RANDOM_STATE = 42

# テストデータの割合: 全データの20%をテスト用に分割
TEST_SIZE = 0.2

# TF-IDF文字n-gramの範囲: 2文字から5文字までの連続した文字列を特徴量として使用
# 例: "running" → "ru", "un", "nn", "ni", "ing", "run", "unn", "nni", "ning", ...
NGRAM_RANGE = (2, 5)

# ===========================
# ルールベース分類用の固定リスト
# ===========================
# 助動詞の固定リスト
# これらの単語は機械学習モデルではなく、ルールベースで「助動詞」として分類される
AUXILIARY_VERBS = {
    # 基本的な助動詞
    'will',      # 〜だろう、〜するつもりだ
    'would',     # 〜だろう、よく〜したものだ
    'can',       # 〜できる、〜でありうる
    'could',     # 〜できた、〜でありえた
    'may',       # 〜かもしれない、〜してもよい
    'might',     # 〜かもしれない
    'must',      # 〜しなければならない、〜に違いない
    'shall',     # 〜しましょうか、〜することになる
    'should',    # 〜すべきだ、〜のはずだ
    'ought',     # 〜すべきだ（ought to）
    'dare',      # あえて〜する
    'need',      # 〜する必要がある
    
    # 複合助動詞（スペースを含む表現）
    'be able to',      # 〜することができる
    'be going to',     # 〜するつもりだ、〜しそうだ
    'be supposed to',  # 〜することになっている、〜すべきだ
    'be allowed to',   # 〜することを許されている
    'be likely to',    # 〜しそうだ、〜する可能性が高い
    'be about to',     # （今まさに）〜しようとしている
    'be expected to',  # 〜すると期待されている/することになっている
    'be required to',  # 〜することが義務付けられている
    'have to',         # 〜しなければならない
    'has to',          # 〜しなければならない
    'had better',      # 〜したほうがよい
    'used to',         # 以前は〜だった
    'need to',         # 〜する必要がある
}
