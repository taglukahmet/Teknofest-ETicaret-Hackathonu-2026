import polars as pl
import numpy as np

def generate_negative_samples(flat_df: pl.LazyFrame, ratio: int) -> pl.LazyFrame:
    """
    (Task: Member 2)
    Algorithmically generates YOK (0) pairs for the dataset.
    Appends generated rows to the original frame and returns it.
    """
    # 1. Mevcut pozitif çiftleri (VAR) çakışma kontrolü için ayıralım
    # unique() kullanarak çoklu satırları teke indirip hafızayı rahatlatıyoruz
    pos_pairs = flat_df.select(["term_id", "item_id"]).unique()
    
    # Benzersiz terimler ve ürünleri (kategorileriyle birlikte) havuz olarak çekelim
    unique_terms = flat_df.select(["term_id"]).unique()
    unique_items = flat_df.select(["item_id", "category_id"]).unique()
    
    # Kaç adet negatif üretileceğini hesaplayalım
    # LazyFrame olduğu için collect_schema().length() veya benzeri yerine 
    # eager çalıştıracağımız kısımlar için collect() kullanacağız.
    pos_count = flat_df.select(pl.len()).collect().item()
    total_neg_needed = pos_count * ratio
    
    # Yarı yarıya Random ve Hard negative üretelim
    n_random_needed = total_neg_needed // 2
    n_hard_needed = total_neg_needed - n_random_needed
    
    # --- A. RANDOM NEGATIVE SAMPLING ---
    # Eager tarafta NumPy ile hızlıca index seçip Lazy havuzla birleştireceğiz
    terms_eager = unique_terms.collect()
    items_eager = unique_items.collect()
    
    np.random.seed(42)
    # Çakışma (anti-join) ihtimaline karşı %30 daha fazla rastgele index üretiyoruz
    oversample_factor = 1.3
    rnd_size = int(n_random_needed * oversample_factor)
    
    random_term_idx = np.random.randint(0, len(terms_eager), size=rnd_size)
    random_item_idx = np.random.randint(0, len(items_eager), size=rnd_size)
    
    # NumPy array'lerini Polars LazyFrame'e dönüştürme
    random_negs_df = pl.DataFrame({
        "term_id": terms_eager["term_id"].to_numpy()[random_term_idx],
        "item_id": items_eager["item_id"].to_numpy()[random_item_idx]
    }).lazy()
    
    # Orijinal doğruları (VAR) temizle ve ihtiyacımız kadarını al
    random_negs = (
        random_negs_df
        .unique()
        .join(pos_pairs, on=["term_id", "item_id"], how="anti")
        .head(n_random_needed)
        .with_columns([
            pl.lit(0).alias("label"),
            pl.concat_str([pl.lit("NEG_TRN_RND_"), pl.int_range(0, pl.len())]).alias("id")
        ])
    )
    
    # --- B. HARD NEGATIVE SAMPLING ---
    # Terimlerin gerçek kategorilerini eşleştirelim
    term_categories = flat_df.join(unique_items, on="item_id", how="inner").select(["term_id", "category_id"]).unique()
    
    # Aynı kategorideki tüm yanlış eşleşmeleri (kategori içi cross-join gibi) bulalım
    # Büyük datalarda patlamamak için sadece gerekli kolonları tutuyoruz
    hard_negs_pool = term_categories.join(unique_items, on="category_id", how="inner").select(["term_id", "item_id"])
    
    # Orijinal doğruları (VAR) temizle, karıştır ve her terim için ilkini seç (Grup başına 1 yanlış ürün)
    hard_negs = (
        hard_negs_pool
        .join(pos_pairs, on=["term_id", "item_id"], how="anti")
        .sample(fraction=1.0, shuffle=True, seed=42) # Vektörize karıştırma
        .group_by("term_id")
        .first()
        .head(n_hard_needed)
        .with_columns([
            pl.lit(0).alias("label"),
            pl.concat_str([pl.lit("NEG_TRN_HRD_"), pl.int_range(0, pl.len())]).alias("id")
        ])
    )
    
    # --- C. BİRLEŞTİRME VE FORMATLAMA ---
    # Orijinal dataframe'e TRN_... şeklinde id ekleyelim (Eğer önceden yoksa)
    original_formatted = flat_df.with_columns(
        pl.concat_str([pl.lit("TRN_"), pl.int_range(0, pl.len())]).alias("id")
    ).select(["id", "term_id", "item_id", "label"])
    
    # Negatiflerin kolon sıralamasını orijinal data ile eşitleyelim
    random_negs = random_negs.select(["id", "term_id", "item_id", "label"])
    hard_negs = hard_negs.select(["id", "term_id", "item_id", "label"])
    
    # Hepsini dikeyde birleştir (Concat)
    final_df = pl.concat([original_formatted, random_negs, hard_negs])
    
    return final_df