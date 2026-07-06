import polars as pl
# Projenizdeki config dosyasını import ediyoruz
import config 

def generate_negative_samples(flat_df: pl.LazyFrame, ratio: int) -> pl.LazyFrame:
    """
    (Task: Member 2)
    Algorithmically generates YOK (0) pairs for the dataset seamlessly in a 100% Lazy pipeline.
    Optimizes memory with pl.Int8 for labels and uses config.RANDOM_SEED for reproducibility.
    """
    # 1. Mevcut pozitif çiftleri (VAR) çakışma kontrolü için ayırıyoruz
    pos_pairs = flat_df.select(["term_id", "item_id"]).unique()
    
    unique_terms = flat_df.select(["term_id"]).unique()
    unique_items = flat_df.select(["item_id", "category_id"]).unique()
    
    # --- A. RANDOM NEGATIVE SAMPLING ---
    oversample_ratio = int(ratio * 1.5)
    
    # seed=42 yerine config.RANDOM_SEED entegrasyonu
    sampled_terms = unique_terms.sample(fraction=oversample_ratio, with_replacement=True, seed=config.RANDOM_SEED)
    sampled_items = unique_items.select(["item_id"]).sample(fraction=oversample_ratio, with_replacement=True, seed=config.RANDOM_SEED)
    
    random_negs = (
        sampled_terms.with_row_index("idx")
        .join(sampled_items.with_row_index("idx"), on="idx", how="inner")
        .drop("idx")
        .join(pos_pairs, on=["term_id", "item_id"], how="anti")
        .with_columns([
            # Veri tipi uyuşmazlığını önlemek için açıkça pl.Int8 tanımlandı
            pl.lit(0, dtype=pl.Int8).alias("label"),
            pl.concat_str([pl.lit("NEG_TRN_RND_"), pl.int_range(0, pl.len())]).alias("id")
        ])
    )
    
    # --- B. HARD NEGATIVE SAMPLING ---
    collapsed_items = unique_items.group_by("category_id").agg(pl.col("item_id").alias("item_pool"))
    term_categories = flat_df.join(unique_items, on="item_id", how="inner").select(["term_id", "category_id"]).unique()
    
    hard_negs = (
        term_categories.join(collapsed_items, on="category_id", how="inner")
        # List expression içinde config.RANDOM_SEED kullanıldı
        .with_columns(
            pl.col("item_pool").list.sample(n=1, seed=config.RANDOM_SEED).list.get(0).alias("item_id")
        )
        .drop("item_pool")
        .join(pos_pairs, on=["term_id", "item_id"], how="anti")
        .with_columns([
            # Veri tipi uyuşmazlığını önlemek için açıkça pl.Int8 tanımlandı
            pl.lit(0, dtype=pl.Int8).alias("label"),
            pl.concat_str([pl.lit("NEG_TRN_HRD_"), pl.int_range(0, pl.len())]).alias("id")
        ])
    )
    
    # --- C. BİRLEŞTİRME VE FORMATLAMA ---
    original_formatted = flat_df.with_columns(
        pl.concat_str([pl.lit("TRN_"), pl.int_range(0, pl.len())]).alias("id")
    ).select(["id", "term_id", "item_id", "label"])
    
    # Son örneklemelerde de config.RANDOM_SEED kullanıldı
    final_randoms = random_negs.sample(fraction=(ratio / 2) / oversample_ratio, seed=config.RANDOM_SEED).select(["id", "term_id", "item_id", "label"])
    final_hards = hard_negs.sample(fraction=(ratio / 2), with_replacement=True, seed=config.RANDOM_SEED).select(["id", "term_id", "item_id", "label"])
    
    return pl.concat([original_formatted, final_randoms, final_hards])