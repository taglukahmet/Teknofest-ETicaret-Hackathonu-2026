import polars as pl


def serialize_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    (Task: Member 1)
    Concatenates individual text columns into a single structured sequence string.
    Returns dataframe with a new 'input_text' column.
    """

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
        .alias("input_text")
    )

    return df