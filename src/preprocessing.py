def create_text_features(df):
    """
    Combines Conclusion, Stance, and Premise into a single 'text' column.
    """
    df = df.copy()
    
    # We ensure columns exist and fill NaNs to prevent errors
    for col in ['Conclusion', 'Stance', 'Premise']:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from DataFrame")
        df[col] = df[col].fillna("")
        
    # Concatenate
    df['text'] = df['Conclusion'] + " " + df['Stance'] + " " + df['Premise']
    return df