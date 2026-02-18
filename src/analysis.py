import numpy as np

def extract_top_features(pipeline, class_names, n=10):
    """
    Extracts and prints the top Positive and Negative n-grams for each class.
    Assumes pipeline has steps ['vec', 'clf'].
    """
    vectorizer = pipeline.named_steps['vec']
    ovr_model = pipeline.named_steps['clf']

    feature_names = vectorizer.get_feature_names_out()
    classifiers = ovr_model.estimators_

    print("\n🔍 EXTRACTING DISCRIMINATIVE FEATURES (TOP 10 N-GRAMS)")
    print("=" * 60)

    for i, clf in enumerate(classifiers):
        if i >= len(class_names): break
            
        class_name = class_names[i]
        coefs = clf.coef_[0]
        sorted_indices = np.argsort(coefs)

        # Top Negative (lowest coefficients)
        top_negative = [feature_names[idx] for idx in sorted_indices[:n]]
        
        # Top Positive (highest coefficients)
        top_positive = [feature_names[idx] for idx in sorted_indices[-n:]][::-1]

        print(f"\n🏆 CLASS: {class_name}")
        print(f"   🟢 POSITIVE (+): {', '.join(top_positive)}")
        print(f"   🔴 NEGATIVE (-): {', '.join(top_negative)}")

def print_failure_examples(X_text, y_true, y_pred, class_names, n=5):
    """
    Prints random examples where the model made at least one error.
    """
    # Find indices where at least one label differs
    error_indices = np.where(np.any(y_true != y_pred, axis=1))[0]
    
    print(f"\n❌ FOUND {len(error_indices)} TEXTS WITH ERRORS.")
    print(f"Showing {n} random examples for Qualitative Analysis:\n")

    if len(error_indices) < n:
        n = len(error_indices)
        
    np.random.seed(42)
    sample_indices = np.random.choice(error_indices, n, replace=False)

    for i, idx in enumerate(sample_indices):
        original_text = X_text[idx]
        
        # Get labels
        true_labels = [class_names[j] for j, val in enumerate(y_true[idx]) if val == 1]
        pred_labels = [class_names[j] for j, val in enumerate(y_pred[idx]) if val == 1]

        print("=" * 80)
        print(f"📝 EXAMPLE #{i+1} (Index {idx})")
        print("-" * 80)
        print(f"TEXT: {original_text}\n")
        print(f"🧠 TRUE:      {true_labels}")
        print(f"🤖 PREDICTED: {pred_labels}")
        print("=" * 80 + "\n")