import polars as pl

###
#Görev 4) Text Serialization    
###

#lazy okuyoruz
df = pl.scan_parquet("training_flattened.parquet")

# yeni "text" isimli istenen columnu ekliyoruz
df = df.with_columns(
    pl.concat_str(
        [
            pl.lit("Arama: "),
            pl.col("query"),
            pl.lit(" [SEP] Ürün: "),
            pl.col("title"),
            pl.lit(" | Kategori: "),
            pl.col("category"),
            pl.lit(" | Marka: "),
            pl.col("brand"),
            pl.lit(" | Hedef: "),
            pl.col("gender"),
            pl.lit(", "),
            pl.col("age_group"),
            pl.lit(" | Özellikler: "),
            pl.col("attributes"),
        ]
    )
    .str.to_lowercase()
    .str.strip_chars()
    .alias("text")
)

result = df.collect()
result.write_parquet("training_serialized.parquet")