import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from sklearn.metrics import multilabel_confusion_matrix

def plot_class_distribution(df, label_cols, title="Class Distribution"):
    """
    Plots the count and percentage of each class in the dataset.
    """
    # 1. Calculate Counts and Percentages
    class_counts = df[label_cols].sum().sort_values(ascending=False)
    class_percs = (class_counts / len(df)) * 100
    
    # 2. Print Summary
    print(f"📊 {title} ANALYSIS (N={len(df)})")
    print("-" * 40)
    summary_df = pd.DataFrame({'Count': class_counts, 'Percentage': class_percs})
    print(summary_df)
    
    # 3. Check for Empty Examples
    no_label_count = len(df[df[label_cols].sum(axis=1) == 0])
    print(f"\n⚠️ Arguments with NO labels: {no_label_count} ({no_label_count/len(df)*100:.2f}%)")

    # 4. Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=class_counts.values, y=class_counts.index, hue=class_counts.index, palette="viridis", legend=False)
    
    plt.title(f"{title} (Total N={len(df)})", fontsize=15)
    plt.xlabel("Number of Positive Examples", fontsize=12)
    plt.ylabel("Human Value Label", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add percentage text
    for i, v in enumerate(class_counts.values):
        plt.text(v + 10, i, f"{class_percs.values[i]:.1f}%", va='center', fontsize=10)
        
    plt.tight_layout()
    plt.show()

def display_random_examples(df, n=5, label_cols=None):
    """
    Prints random examples with their active labels.
    """
    if label_cols is None:
        # Try to guess label columns if not provided (exclude common metadata)
        metadata = ['Argument ID', 'Conclusion', 'Stance', 'Premise', 'Usage', 'Source']
        label_cols = [c for c in df.columns if c not in metadata]

    samples = df.sample(n, random_state=42)

    for idx, row in samples.iterrows():
        print(f"🆔 ID: {row.get('Argument ID', 'N/A')}")
        print(f"📢 CONCLUSION: {row.get('Conclusion', 'N/A')}")
        print(f"⚖️ STANCE: {row.get('Stance', 'N/A')}")
        print(f"📝 PREMISE: {row.get('Premise', 'N/A')}")
        print("-" * 30)
        print("🧠 ACTUAL HUMAN VALUES:")
        
        has_values = False
        for val in label_cols:
            if row[val] == 1:
                print(f"   ✅ {val}")
                has_values = True
                
        if not has_values:
            print("   (No values annotated)")
            
        print("=" * 80 + "\n")

def plot_stratification_check(y_train, y_test, label_cols, filename="stratification_check.png"):
    """
    Plots the distribution of labels in Train vs Test to verify stratification.
    """
    # 1. Calculate Proportions
    train_dist = np.mean(y_train, axis=0)
    test_dist = np.mean(y_test, axis=0)

    # 2. Prepare Data for Plotting
    strat_df = pd.DataFrame({
        'Label': label_cols,
        'Train Set': train_dist,
        'Test Set': test_dist,
        'Difference': train_dist - test_dist
    })
    
    # Melt for side-by-side comparison
    df_melted = strat_df.melt(id_vars=['Label', 'Difference'], 
                              value_vars=['Train Set', 'Test Set'], 
                              var_name='Dataset', 
                              value_name='Prevalence')

    # 3. Create Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})

    # --- TOP PLOT: Absolute Distribution ---
    sns.barplot(data=df_melted, x='Label', y='Prevalence', hue='Dataset', 
                palette=['#2c3e50', '#e74c3c'], ax=ax1)
    
    ax1.set_title('Stratification Verification: Train vs. Validation Distribution', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Prevalence (0.0 - 1.0)', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_xticklabels([])
    
    sns.barplot(data=df_melted, x='Label', y='Prevalence', hue='Dataset', 
                palette=['#2c3e50', '#e74c3c'], ax=ax1)
    
    ax1.set_title('Stratification Verification: Train vs. Validation Distribution', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Prevalence (0.0 - 1.0)', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_xticklabels([])
    ax1.legend(title='Split', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # --- BOTTOM PLOT: Residuals (The Error) ---
    sns.barplot(data=strat_df, x='Label', y='Difference', color='purple', ax=ax2)
    
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title('Deviation (Train - Val)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Difference', fontsize=12)
    ax2.set_xlabel('Human Value Label', fontsize=12)
    
    # Rotate labels for readability
    ax2.set_xticklabels(strat_df['Label'], rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # Add Error Metric
    mean_error = np.mean(np.abs(strat_df['Difference']))
    plt.figtext(0.5, 0.01, f"Mean Absolute Deviation: {mean_error:.5f} (Lower is Better)", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


# src/visualization.py (append this)
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

def plot_multilabel_confusion(y_true, y_pred, class_names, filename="confusion_matrix.png"):
    """
    Plots a grid of normalized confusion matrices for each class.
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    
    num_classes = len(class_names)
    cols = 4
    rows = math.ceil(num_classes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows))
    axes = axes.flatten()

    for i, (matrix, name) in enumerate(zip(mcm, class_names)):
        # Normalize by row (True Label) to show Recall vs Specificity
        row_sums = matrix.sum(axis=1)[:, np.newaxis]
        # Add epsilon to prevent division by zero
        normalized_matrix = matrix.astype('float') / (row_sums + 1e-10)

        sns.heatmap(normalized_matrix, annot=True, fmt='.1%', cmap='Blues', ax=axes[i],
                    xticklabels=['Pred: No', 'Pred: Yes'], 
                    yticklabels=['True: No', 'True: Yes'],
                    cbar=False, vmin=0, vmax=1)
        
        axes[i].set_title(f'{name}', fontweight='bold', fontsize=14)
        axes[i].set_ylabel('True Label', fontsize=10)
        axes[i].set_xlabel('Predicted Label', fontsize=10)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()
    ax1.legend(title='Split', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # --- BOTTOM PLOT: Residuals (The Error) ---
    sns.barplot(data=strat_df, x='Label', y='Difference', color='purple', ax=ax2)
    
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title('Deviation (Train - Val)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Difference', fontsize=12)
    ax2.set_xlabel('Human Value Label', fontsize=12)
    
    # Rotate labels for readability
    ax2.set_xticklabels(strat_df['Label'], rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # Add Error Metric
    mean_error = np.mean(np.abs(strat_df['Difference']))
    plt.figtext(0.5, 0.01, f"Mean Absolute Deviation: {mean_error:.5f} (Lower is Better)", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_multilabel_confusion(y_true, y_pred, class_names, filename="confusion_matrix.png"):
    """
    Plots a grid of normalized confusion matrices for each class.
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    
    num_classes = len(class_names)
    cols = 4
    rows = math.ceil(num_classes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows))
    axes = axes.flatten()

    for i, (matrix, name) in enumerate(zip(mcm, class_names)):
        # Normalize by row (True Label) to show Recall vs Specificity
        row_sums = matrix.sum(axis=1)[:, np.newaxis]
        # Add epsilon to prevent division by zero
        normalized_matrix = matrix.astype('float') / (row_sums + 1e-10)

        sns.heatmap(normalized_matrix, annot=True, fmt='.1%', cmap='Blues', ax=axes[i],
                    xticklabels=['Pred: No', 'Pred: Yes'], 
                    yticklabels=['True: No', 'True: Yes'],
                    cbar=False, vmin=0, vmax=1)
        
        axes[i].set_title(f'{name}', fontweight='bold', fontsize=14)
        axes[i].set_ylabel('True Label', fontsize=10)
        axes[i].set_xlabel('Predicted Label', fontsize=10)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()