import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def apply_group_kfold(df: pl.DataFrame, n_splits: int) -> pl.DataFrame:
    """
    (Task: Member 2)
    Note: Requires eager evaluation (DataFrame, not LazyFrame) for scikit-learn compatibility.
    Applies StratifiedGroupKFold on the 'term_id' column to prevent data leakage 
    while maintaining balanced label (1/0) distributions across all folds.
    Returns dataframe with a new 'fold' integer column.
    """
    # 1. Eğer dataframe boşsa doğrudan boş bir fold kolonu ekleyip dönelim (Edge-case kontrolü)
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.Int32).alias("fold"))

    # 2. scikit-learn ile uyum için gerekli yapıları hazırlayalım
    # StratifiedGroupKFold için gerçek özellik matrisine (X) ihtiyacımız yok, indexler üzerinden gideceğiz.
    # RAM'i şişirmemek için sadece satır sayısı uzunluğunda sıfırlardan oluşan minik bir X_dummy üretiyoruz.
    X_dummy = np.zeros(len(df))
    
    # Stratified mantığı için hedef değişken (y) ve izolasyon için gruplar (groups)
    y = df["label"].to_numpy()
    groups = df["term_id"].to_numpy()
    
    # 3. Boş bir fold array'i oluşturalım (Varsayılan olarak -1 veriyoruz)
    fold_assignments = np.full(len(df), -1, dtype=np.int32)
    
    # 4. StratifiedGroupKFold nesnesini başlatalım
    # random_state ve shuffle vererek fold dağılımının her çalışmada tutarlı olmasını sağlıyoruz
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 5. İterasyonla her satıra hangi fold'a ait olduğunu yazalım
    # split() fonksiyonu hem grupları korur hem de y'deki label dağılımını fold'lara eşit yayar
    for fold_idx, (_, val_indices) in enumerate(sgkf.split(X_dummy, y, groups=groups)):
        fold_assignments[val_indices] = fold_idx
        
    # 6. Oluşan fold array'ini Polars DataFrame'e yeni bir kolon olarak ekleyelim
    df_with_folds = df.with_columns(
        pl.Series(fold_assignments, dtype=pl.Int32).alias("fold")
    )
    
    return df_with_folds
