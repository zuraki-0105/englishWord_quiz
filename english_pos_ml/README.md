# English Part-of-Speech Classification Model

英単語のスペルから品詞を推測する機械学習モデル構築プロジェクトです。

## ディレクトリ構成
```
english_pos_ml/
  data/          # 入力データ (englishWord_data.csv)
  src/           # ソースコード
    main.py      # 実行エントリポイント
    ...
  models/        # 学習済みモデル (*.joblib)
  outputs/       # 評価結果 (metrics.txt, confusion_matrix.csv)
  requirements.txt
```

## 実行方法

1. 必要なライブラリをインストール:
   ```bash
   pip install -r requirements.txt
   ```

2. データを配置:
   `data/englishWord_data.csv` を配置してください。
   カラム構成: `id`, `spell`, `meaning`, `pos` (または日本語列名も可)

3. モデル学習と評価を実行:
   ```bash
   python src/main.py
   ```

4. 結果の確認:
   - `models/` に学習済みモデルが保存されます。
   - `outputs/` に評価レポートが出力されます。
