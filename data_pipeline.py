import polars as pl
import os

base_path = '/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle/'

# GÖREV 1: Format Dönüştürme (Data Prep 1)
dtypes_items = {'item_id': pl.String, 'title': pl.String, 'category': pl.String, 'brand': pl.String, 'gender': pl.String, 'age_group': pl.String, 'attributes': pl.String}
dtypes_terms = {'term_id': pl.String, 'query': pl.String}
dtypes_train = {'id': pl.String, 'term_id': pl.String, 'item_id': pl.String, 'label': pl.Int8}
dtypes_sub = {'id': pl.String, 'term_id': pl.String, 'item_id': pl.String}

pl.read_csv(base_path + 'items.csv', schema_overrides=dtypes_items).write_parquet('items.parquet')
pl.read_csv(base_path + 'terms.csv', schema_overrides=dtypes_terms).write_parquet('terms.parquet')
pl.read_csv(base_path + 'training_pairs.csv', schema_overrides=dtypes_train).write_parquet('training_pairs.parquet')
pl.read_csv(base_path + 'submission_pairs.csv', schema_overrides=dtypes_sub).write_parquet('submission_pairs.parquet')

# GÖREV 2: Veri Birleştirme (Flattening - Data Prep 2)
items_lazy = pl.scan_parquet('items.parquet')
terms_lazy = pl.scan_parquet('terms.parquet')
train_pairs_lazy = pl.scan_parquet('training_pairs.parquet')
sub_pairs_lazy = pl.scan_parquet('submission_pairs.parquet')

train_merged = (
    train_pairs_lazy
    .join(terms_lazy, on='term_id', how='left')
    .join(items_lazy, on='item_id', how='left')
    .fill_null("unknown")
).collect()

sub_merged = (
    sub_pairs_lazy
    .join(terms_lazy, on='term_id', how='left')
    .join(items_lazy, on='item_id', how='left')
    .fill_null("unknown")
).collect()

# GÖREV 4: Metin Serileştirme (Text Serialization - Data Prep 4)
def temizle(sutun_adi):
    return pl.col(sutun_adi).str.to_lowercase().str.strip_chars()

def metinleri_birlestir(df):
    return df.with_columns(
        yapay_zeka_metni = pl.concat_str([
            pl.lit("arama: "), temizle("query"),
            pl.lit(" [sep] ürün: "), temizle("title"),
            pl.lit(" | kategori: "), temizle("category"),
            pl.lit(" | marka: "), temizle("brand"),
            pl.lit(" | hedef: "), temizle("gender"), pl.lit(", "), temizle("age_group"),
            pl.lit(" | özellikler: "), temizle("attributes")
        ])
    )

train_final = metinleri_birlestir(train_merged)
train_final.write_parquet('train_hazir.parquet')

test_final = metinleri_birlestir(sub_merged)
test_final.write_parquet('test_hazir.parquet')
