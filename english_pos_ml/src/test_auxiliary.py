import sys
import os
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def test_auxiliary_verbs():
    """
    助動詞の固定リスト対応をテストする
    """
    print("=== Testing Auxiliary Verb Classification ===\n")
    
    # 学習済みパイプラインの読み込み
    pipeline_file = os.path.join(config.MODELS_DIR, 'pipeline.joblib')
    
    if not os.path.exists(pipeline_file):
        print(f"Error: Pipeline file not found at {pipeline_file}")
        print("Please run 'python src/main.py' first to train the model.")
        return
    
    pipeline = joblib.load(pipeline_file)
    print(f"Loaded pipeline from {pipeline_file}\n")
    
    # テスト用の助動詞リスト
    test_auxiliary_verbs = [
        'will', 'would', 'can', 'could', 'may', 'might',
        'must', 'shall', 'should', 'ought', 'dare', 'need'
    ]
    
    # テスト用の非助動詞（比較用）
    test_other_words = [
        'run', 'happy', 'quickly', 'book', 'beautiful',
        'in', 'and', 'the', 'very', 'good'
    ]
    
    # 助動詞のテスト
    print("--- Testing Auxiliary Verbs ---")
    print("Expected: All should be classified as '助動詞'\n")
    
    predictions = pipeline.predict(test_auxiliary_verbs)
    
    correct = 0
    for word, pred in zip(test_auxiliary_verbs, predictions):
        is_correct = pred == '助動詞'
        status = "✓" if is_correct else "✗"
        print(f"{status} {word:10s} → {pred}")
        if is_correct:
            correct += 1
    
    accuracy = correct / len(test_auxiliary_verbs) * 100
    print(f"\nAuxiliary Verb Accuracy: {correct}/{len(test_auxiliary_verbs)} ({accuracy:.1f}%)")
    
    # 非助動詞のテスト
    print("\n--- Testing Non-Auxiliary Words ---")
    print("Expected: None should be classified as '助動詞'\n")
    
    predictions = pipeline.predict(test_other_words)
    
    correct = 0
    for word, pred in zip(test_other_words, predictions):
        is_correct = pred != '助動詞'
        status = "✓" if is_correct else "✗"
        print(f"{status} {word:10s} → {pred}")
        if is_correct:
            correct += 1
    
    accuracy = correct / len(test_other_words) * 100
    print(f"\nNon-Auxiliary Accuracy: {correct}/{len(test_other_words)} ({accuracy:.1f}%)")
    
    # 複合助動詞のテスト
    print("\n--- Testing Compound Auxiliary Verbs ---")
    test_compound = ['have to', 'be able to', 'used to', 'had better']
    
    predictions = pipeline.predict(test_compound)
    
    correct = 0
    for word, pred in zip(test_compound, predictions):
        is_correct = pred == '助動詞'
        status = "✓" if is_correct else "✗"
        print(f"{status} {word:15s} → {pred}")
        if is_correct:
            correct += 1
    
    accuracy = correct / len(test_compound) * 100
    print(f"\nCompound Auxiliary Accuracy: {correct}/{len(test_compound)} ({accuracy:.1f}%)")
    
    print("\n=== Test Completed ===")

if __name__ == '__main__':
    test_auxiliary_verbs()
