# ソースコード仕様書 (`src/` ディレクトリ)

このプロジェクトを構成する各 Python スクリプトの役割と主要な機能について説明します。

## 1. 実行・制御関連

### [main.py](file:///c:/Users/tetsu/Programming/app/englishWord_quiz/english_pos_ml/src/main.py)
**役割**: プロジェクト全体の実行を管理するメインエントリーポイント。
- **機能**:
    - `data_loader` を用いたデータの読み込み。
    - `train` を呼び出してモデルの学習。
    - `evaluate` を呼び出してモデルの評価。
    - 学習済みモデル（パイプライン全体）の保存。
    - 設定に基づいたエラー分析の実行。

### [config.py](file:///c:/Users/tetsu/Programming/app/englishWord_quiz/english_pos_ml/src/config.py)
**役割**: プロジェクト全体で使用する定数やパスの集中管理。
- **機能**:
    - データファイル、モデル保存先、出力先のパス定義。
    - ハイパーパラメータ（乱数シード、テストサイズ、n-gram範囲など）の設定。
    - CSVカラム名のマッピング辞書。
    - 助動詞の固定リスト（ルールベース判定用）の保持。
    - 実行制御フラグ（WordNet使用の有無など）の管理。

---

## 2. データ処理・特徴量関連

### [data_loader.py](file:///c:/Users/tetsu/Programming/app/englishWord_quiz/english_pos_ml/src/data_loader.py)
**役割**: CSVデータの読み込みと前処理。
- **機能**:
    - UTF-8およびCP932（Shift-JIS）の自動エンコーディング対応。
    - カラム名の「ゆらぎ」を吸収し、内部共通名（spell, pos）へ統一。
    - 欠損値（NaN）の削除とスペルの小文字化。

### [feature_engineering.py](file:///c:/Users/tetsu/Programming/app/englishWord_quiz/english_pos_ml/src/feature_engineering.py)
**役割**: 単語のスペルから言語学的特徴を抽出。
- **機能**:
    - **接尾辞・接頭辞判定**: `-ing`, `-tion`, `-ly` など、特定の品詞に特徴的な綴りパターンをフラグ化。
    - **単語構造の数値化**: 単語の長さ、母音・子音の比率、特殊文字（ハイフン、アポストロフィ）の有無を計算。
    - **scikit-learn 互換クラス**: `LinguisticFeatureExtractor` クラスとして実装されており、学習パイプラインに直接組み込み可能。

---

## 3. モデル・学習関連

### [train.py](file:///c:/Users/tetsu/Programming/app/englishWord_quiz/english_pos_ml/src/train.py)
**役割**: 機械学習モデルの構築と学習。
- **機能**:
    - `FeatureUnion` を使用し、TF-IDF 特徴と `feature_engineering.py` の独自特徴を統合。
    - **アンサンブル学習**: Voting (多数決), Stacking (重ね合わせ), Boosting (LightGBM) の性能を比較。
    - 精度が最も高いモデルを自動選択し、最終的な予測エンジンを構築。
    - クラス不均衡に対応するため、各学習器に重み付け（balanced）を適用。

### [rule_based_classifier.py](file:///c:/Users/tetsu/Programming/app/englishWord_quiz/english_pos_ml/src/rule_based_classifier.py)
**役割**: ルールベースと機械学習を融合させるラッパークラス。
- **機能**:
    - 予測時に最初に入力単語が「助動詞リスト」に存在するかチェック。
    - 存在すれば即座に「助動詞」と判定し、存在しなければ機械学習モデルへ処理を委譲。
    - scikit-learn の API（fit, predict, predict_proba）に準拠。

---

## 4. 評価・分析関連

### [evaluate.py](file:///c:/Users/tetsu/Programming/app/englishWord_quiz/english_pos_ml/src/evaluate.py)
**役割**: モデルの精度評価と可視化。
- **機能**:
    - Accuracy, Macro F1 スコア、詳細な Classification Report の生成。
    - 混同行列（Confusion Matrix）の CSV 出力。
    - PR曲線 (Precision-Recall) と ROC曲線の描画および画像保存。

### [analyze_errors.py](file:///c:/Users/tetsu/Programming/app/englishWord_quiz/english_pos_ml/src/analyze_errors.py)
**役割**: 予測ミスの詳細分析。
- **機能**:
    - テストデータのうち、予測を外した単語を抽出して `error_analysis.csv` に保存。
    - どの品詞をどの品詞と間違えやすいかなど、頻出ミスパターンの集計。

### [tune_params.py](file:///c:/Users/tetsu/Programming/app/englishWord_quiz/english_pos_ml/src/tune_params.py)
**役割**: ハイパーパラメータの最適化。
- **機能**:
    - `GridSearchCV` を使用し、最適な n-gram 範囲や正則化パラメータ（C）を自動探索。
