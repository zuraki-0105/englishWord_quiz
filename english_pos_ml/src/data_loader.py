import pandas as pd
import os
import sys

# srcディレクトリをパスに追加(直接実行時のインポートエラーを防ぐため)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def load_data(filepath=None):
    """
    CSVファイルからデータを読み込み、前処理を行う関数
    
    主な処理内容:
    1. CSVファイルの読み込み(エンコーディングエラーに対応)
    2. カラム名の統一(複数の候補名に対応)
    3. 必須カラムの存在確認
    4. 欠損値の削除
    5. テキストの前処理(小文字化)
    
    Args:
        filepath (str, optional): CSVファイルのパス。Noneの場合はconfig.DATA_FILEを使用
    
    Returns:
        pd.DataFrame: 前処理済みのデータフレーム
    
    Raises:
        FileNotFoundError: 指定されたファイルが存在しない場合
        ValueError: 必須カラムが見つからない場合
    """
    # ファイルパスが指定されていない場合、デフォルトのパスを使用
    if filepath is None:
        filepath = config.DATA_FILE
    
    # ファイルの存在確認
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    
    # CSVファイルの読み込み
    # まずUTF-8で試行し、失敗した場合はcp932(Shift-JIS)で再試行
    # 日本語環境で作成されたCSVファイルに対応するため
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='cp932')

    # カラム名の統一処理
    # CSVファイルのカラム名が異なる場合でも、内部で統一された名前に変換
    rename_dict = {}
    for internal_name, candidates in config.COLUMN_MAPPING.items():
        for col in df.columns:
            if col in candidates:
                rename_dict[col] = internal_name
                break
    
    df = df.rename(columns=rename_dict)
    
    # 必須カラムの存在確認
    # 'spell'(単語のスペル)と'pos'(品詞)は必須
    required_cols = ['spell', 'pos']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available: {df.columns.tolist()}")
    
    # データクリーニング
    # 1. 必須カラムに欠損値がある行を削除
    df = df.dropna(subset=['spell', 'pos'])
    
    # 2. スペルを文字列型に変換し、小文字化
    #    モデルの学習を安定させるため、大文字小文字を統一
    df['spell'] = df['spell'].astype(str).str.lower()
    
    # 読み込み結果の表示
    print(f"Loaded {len(df)} samples.")
    print(f"Columns: {df.columns.tolist()}")
    return df

if __name__ == '__main__':
    # テスト実行: このファイルを直接実行した場合のみ動作
    try:
        df = load_data()
        print(df.head())
    except Exception as e:
        print(e)
