import pandas as pd
import os

def load_dataset(data_dir="data"):
    """
    Loads and merges arguments and labels. 
    Returns separate DataFrames for Train (combined with Val) and Test.
    """
    
    # Helper to load and merge a single split
    def _load_split(split_name):
        args_path = os.path.join(data_dir, f"arguments-{split_name}.tsv")
        labels_path = os.path.join(data_dir, f"labels-{split_name}.tsv")
        
        if not os.path.exists(args_path):
            return None
            
        args = pd.read_csv(args_path, sep='\t')
        labels = pd.read_csv(labels_path, sep='\t')
        return pd.merge(args, labels, on="Argument ID")

    # Load everything
    df_train = _load_split("training")
    df_val = _load_split("validation")
    df_test = _load_split("test")

    # Combine Train and Validation
    if df_train is not None and df_val is not None:
        df_train_full = pd.concat([df_train, df_val], ignore_index=True)
    else:
        df_train_full = df_train

    return df_train_full, df_test