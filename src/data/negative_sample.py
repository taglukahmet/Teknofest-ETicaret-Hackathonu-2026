import polars as pl

def generate_negative_samples(flat_df: pl.LazyFrame, ratio: int) -> pl.LazyFrame:
    """
    (Task: Member 2)
    Algorithmically generates YOK (0) pairs for the dataset seamlessly in a 100% Lazy pipeline.
    Avoids memory explosion by avoiding .collect(), NumPy conversion, and Cartesian joins.
    """
    # 1. Mevcut pozitif çiftleri (VAR) çakışma kontrolü için ayırıyoruz (Hâlâ Lazy)
    pos_pairs = flat_df.select(["term_id", "item_id"]).unique()
    
    # Havuzları tamamen Lazy tutuyoruz
    unique_terms = flat_df.select(["term_id"]).unique()
    unique_items = flat_df.select(["item_id", "category_id"]).unique()
    
    # --- A. RANDOM NEGATIVE SAMPLING (Saf Polars Lazy Yöntemi) ---
    # NumPy indexlemeleri yerine Polars'ın Lazy uyumlu .sample() fonksiyonunu kullanıyoruz.
    # Çakışma (anti-join) ihtimaline karşı %50 daha fazla örnek çekip, hiyerarşiyi koruyoruz.
    oversample_ratio = int(ratio * 1.5)
    
    sampled_terms = unique_terms.sample(fraction=oversample_ratio, with_replacement=True, seed=42)
    sampled_items = unique_items.select(["item_id"]).sample(fraction=oversample_ratio, with_replacement=True, seed=42)
    
    # İki ayrı rastgele seriyi yatayda güvenle birleştirmek için geçici bir index (row_index) ekliyoruz
    random_negs = (
        sampled_terms.with_row_index("idx")
        .join(sampled_items.with_row_index("idx"), on="idx", how="inner")
        .drop("idx")
        .join(pos_pairs, on=["term_id", "item_id"], how="anti") # Doğru olanları ayıkla
        .with_columns([
            pl.lit(0).alias("label"),
            pl.concat_str([pl.lit("NEG_TRN_RND_"), pl.int_range(0, pl.len())]).alias("id")
        ])
    )
    
    # --- B. HARD NEGATIVE SAMPLING (List Aggregation & List Expression Yöntemi) ---
    # Satır sayısının patlamasını engellemek için kategorideki ürünleri bir "Liste" halinde topluyoruz.
    collapsed_items = unique_items.group_by("category_id").agg(pl.col("item_id").alias("item_pool"))
    
    # Terimlerin ait olduğu kategorileri buluyoruz
    term_categories = flat_df.join(unique_items, on="item_id", how="inner").select(["term_id", "category_id"]).unique()
    
    # Kartezyen çarpım yerine, terime ait kategori listesini tek bir satır olarak joinliyoruz (Satır sayısı düz kalır)
    hard_negs = (
        term_categories.join(collapsed_items, on="category_id", how="inner")
        # List Expression kullanarak her satırdaki havuzdan tam 1 adet rastgele ürün seçiyoruz
        .with_columns(
            pl.col("item_pool").list.sample(n=1, seed=42).list.get(0).alias("item_id")
        )
        .drop("item_pool")
        .join(pos_pairs, on=["term_id", "item_id"], how="anti") # Yanlışlıkla doğru üretileni ele
        .with_columns([
            pl.lit(0).alias("label"),
            pl.concat_str([pl.lit("NEG_TRN_HRD_"), pl.int_range(0, pl.len())]).alias("id")
        ])
    )
    
    # --- C. BİRLEŞTİRME VE FORMATLAMA (Lazy Concat) ---
    # Orijinal dataframe'e TRN_... şeklinde id ekleyelim
    original_formatted = flat_df.with_columns(
        pl.concat_str([pl.lit("TRN_"), pl.int_range(0, pl.len())]).alias("id")
    ).select(["id", "term_id", "item_id", "label"])
    
    # Negatifleri oranlamak (ratio) için son filtreleri uyguluyoruz
    # LazyFrame satır sayısını tam bilmediğimiz için kesirsel sample (fraction) kullanarak ratio'yu dengeliyoruz
    final_randoms = random_negs.sample(fraction=(ratio / 2) / oversample_ratio, seed=42).select(["id", "term_id", "item_id", "label"])
    final_hards = hard_negs.sample(fraction=(ratio / 2), with_replacement=True, seed=42).select(["id", "term_id", "item_id", "label"])
    
    # Hepsini dikeyde birleştirip LazyFrame olarak geri dönüyoruz
    return pl.concat([original_formatted, final_randoms, final_hards])

