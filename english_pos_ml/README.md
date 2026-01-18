# 🔤 English Part-of-Speech Classification

英単語のスペルから品詞を推測する機械学習モデル構築プロジェクト

## 📋 概要

このプロジェクトは、英単語のスペル(文字列)のみを入力として、その単語の品詞(動詞、名詞、形容詞など)を分類する機械学習モデルを構築します。TF-IDF文字n-gramとLinearSVCを使用した高精度な分類を実現しています。

### 主な特徴
- ✨ **文字レベルのTF-IDF**: 単語のスペルパターンから特徴を抽出
- 🎯 **LinearSVC分類器**: テキスト分類に最適化された高性能モデル
- 📊 **詳細な評価レポート**: Accuracy、F1スコア、混同行列を自動生成
- 🔄 **クラス不均衡対応**: balanced class weightで少数クラスにも対応
- 🌐 **多言語対応**: 日本語・英語のCSVカラム名に対応

## 🗂️ ディレクトリ構成

```
english_pos_ml/
├── data/                      # 入力データ
│   └── englishWord_data.csv   # 英単語と品詞のデータセット
├── src/                       # ソースコード
│   ├── main.py                # メイン実行スクリプト
│   ├── config.py              # 設定ファイル(パス、パラメータ)
│   ├── data_loader.py         # データ読み込み・前処理
│   ├── train.py               # モデル学習
│   └── evaluate.py            # モデル評価
├── models/                    # 学習済みモデル(自動生成)
│   ├── pos_model.joblib       # 学習済み分類器
│   └── vectorizer.joblib      # 学習済みTF-IDFベクトライザ
├── outputs/                   # 評価結果(自動生成)
│   ├── metrics.txt            # 評価指標レポート
│   └── confusion_matrix.csv   # 混同行列
├── requirements.txt           # 依存ライブラリ
└── README.md                  # このファイル
```

## 🚀 クイックスタート

### 1. 環境構築

```bash
pip install -r requirements.txt
```

**必要なライブラリ**:
- `pandas`: データ処理
- `scikit-learn`: 機械学習モデル
- `joblib`: モデルの保存・読み込み

### 2. データ準備

`data/englishWord_data.csv` に以下の形式でデータを配置してください:

| カラム名 | 説明 | 必須 |
|---------|------|------|
| `spell` (または `スペル`, `word`, `単語`) | 英単語のスペル | ✅ |
| `pos` (または `品詞`, `part_of_speech`, `label`) | 品詞ラベル | ✅ |
| `id` | データID | ❌ |
| `meaning` | 意味 | ❌ |

**サンプルデータ**:
```csv
id,spell,pos,meaning
1,run,動詞,走る
2,beautiful,形容詞,美しい
3,quickly,副詞,素早く
```

### 3. モデル学習と評価

```bash
python src/main.py
python english_pos_ml/src/main.py
```

**実行される処理**:
1. データの読み込みと前処理
2. 訓練データ・テストデータの分割(80:20)
3. TF-IDF特徴量の抽出
4. LinearSVCモデルの学習
5. テストデータでの評価
6. モデルと評価結果の保存

### 4. 結果の確認

#### 📈 評価指標 (`outputs/metrics.txt`)
```
Accuracy: 0.5588
Macro F1: 0.3652

Classification Report:
              precision    recall  f1-score   support
      動詞       0.65      0.61      0.63       135
      名詞       0.59      0.58      0.58       133
     形容詞       0.36      0.44      0.39        59
       ...
```

#### 🔢 混同行列 (`outputs/confusion_matrix.csv`)
実際の品詞と予測された品詞の組み合わせを確認できます。

## ⚙️ モデル設定

主要なパラメータは `src/config.py` で設定できます:

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| `RANDOM_STATE` | 42 | 乱数シード(再現性確保) |
| `TEST_SIZE` | 0.2 | テストデータの割合 |
| `NGRAM_RANGE` | (2, 5) | 文字n-gramの範囲 |

## 🔧 カスタマイズ

### モデルの変更
`src/train.py` の分類器を変更することで、他のアルゴリズムも試せます:

```python
# LogisticRegressionに変更する場合
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(
    class_weight='balanced',
    random_state=config.RANDOM_STATE,
    max_iter=1000
)
```

### ハイパーパラメータ調整
GridSearchCVを使用して最適なパラメータを探索できます。

## 📊 性能向上のヒント

現在の精度をさらに向上させるには:

1. **特徴量の追加**: 接尾辞パターン(-ing, -ed, -lyなど)を追加
2. **ハイパーパラメータ調整**: GridSearchCVで最適なn-gram範囲やCパラメータを探索
3. **データ量の増加**: 特に少数クラス(副詞、前置詞など)のデータを追加
4. **アンサンブル学習**: 複数のモデルを組み合わせる
5. **高度なモデル**: RandomForest、XGBoost、ニューラルネットワークを試す

## 📝 ライセンス

このプロジェクトは学習・研究目的で作成されています。


