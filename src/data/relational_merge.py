import polars as pl


def build_flattened_dataset(pairs_path: str, items_path: str, terms_path: str) -> pl.LazyFrame:
    """
    (Task: Member 1)
    Executes lazy left-joins to combine pairs with their respective catalog and search term text.
    Returns the uncollected LazyFrame computation graph.
    """

    #lazy okuyoruz:
    training_pairs = pl.scan_parquet(pairs_path)
    terms = pl.scan_parquet(terms_path)
    items = pl.scan_parquet(items_path)

    # ilk join: training_pairs + terms, term_id üzerinden
    flattened = training_pairs.join(
        terms,
        on="term_id",
        how="left",
    )

    # ikinci join: ara sonuc + items, item_id üzerinden
    flattened = flattened.join(
        items,
        on="item_id",
        how="left",
    )

    # Null degerleri hallediyoruz:
    flattened = flattened.with_columns(
        pl.col("query").fill_null("unknown"),
        pl.col("title").fill_null("unknown"),
        pl.col("category").fill_null("unknown"),
        pl.col("brand").fill_null("unknown"),
        pl.col("gender").fill_null("unknown"),
        pl.col("age_group").fill_null("unknown"),
        pl.col("attributes").fill_null("unknown"),
    )

    return flattened