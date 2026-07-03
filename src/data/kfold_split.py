import polars as pl
import numpy as np
from sklearn.model_selection import GroupKFold

def apply_group_kfold(df: pl.DataFrame, n_splits: int) -> pl.DataFrame:
    """
    (Task: Member 2)
    Note: Requires eager evaluation (DataFrame, not LazyFrame) for scikit-learn compatibility.
    Applies GroupKFold on the 'term_id' column to prevent data leakage.
    Returns dataframe with a new 'fold' integer column.
    """
    # 1. Eğer dataframe boşsa doğrudan boş bir fold kolonu ekleyip dönelim (Edge-case kontrolü)
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.Int32).alias("fold"))

    # 2. scikit-learn ile uyum için gerekli yapıları hazırlayalım
    # GroupKFold için gerçek özellik matrisine (X) ihtiyacımız yok, indexler üzerinden gideceğiz.
    # Bu yüzden sadece satır sayısı kadar bir NumPy array'i oluşturmak hafıza için en hafif çözümdür.
    X_dummy = np.zeros(len(df))
    
    # Hedef değişken (y) ve gruplayacağımız kolon (groups)
    y = df["label"].to_numpy()
    groups = df["term_id"].to_numpy()
    
    # 3. Boş bir fold array'i oluşturalım (Varsayılan olarak -1 veriyoruz)
    fold_assignments = np.full(len(df), -1, dtype=np.int32)
    
    # 4. GroupKFold nesnesini başlatalım
    gkf = GroupKFold(n_splits=n_splits)
    
    # 5. İterasyonla her satıra hangi fold'a ait olduğunu yazalım
    # split() fonksiyonu grupları (term_id) baz alarak train ve val indexlerini ayırır
    for fold_idx, (_, val_indices) in enumerate(gkf.split(X_dummy, y, groups=groups)):
        fold_assignments[val_indices] = fold_idx
        
    # 6. Oluşan fold array'ini Polars DataFrame'e yeni bir kolon olarak ekleyelim
    # fold değerleri 0, 1, 2, 3, 4 şeklinde dağılacak
    df_with_folds = df.with_columns(
        pl.Series(fold_assignments).alias("fold")
    )
    
    return df_with_folds