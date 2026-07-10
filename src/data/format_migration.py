import os
import polars as pl


def convert_csv_to_parquet(data_dir: str) -> None:
    """
    (Task: Member 1)
    Reads raw CSVs from data_dir and writes them to disk as columnar Parquet files.
    """

    # tahmin ettirmeden typelari biz veriyoruz. farkliligin az oldugu yerlerde
    # Utf8(string) yerine categorical kullaniyoruz ki tasarruf edelim
    items_schema = {
        "item_id": pl.Utf8,
        "title": pl.Utf8,
        "category": pl.Utf8,
        "brand": pl.Categorical,
        "gender": pl.Categorical,
        "age_group": pl.Categorical,
        "attributes": pl.Utf8,
    }

    terms_schema = {
        "term_id": pl.Utf8,
        "query": pl.Utf8,
    }

    training_schema = {
        "id": pl.Utf8,
        "term_id": pl.Utf8,
        "item_id": pl.Utf8,
        "label": pl.Int8,
    }

    submission_schema = {
        "id": pl.Utf8,
        "term_id": pl.Utf8,
        "item_id": pl.Utf8,
    }

    sample_sub_schema = {
        "id": pl.Utf8,
        "prediction": pl.Int64,
    }

    files = {
        "items": ("items.csv", items_schema),
        "terms": ("terms.csv", terms_schema),
        "training_pairs": ("training_pairs.csv", training_schema),
        "submission_pairs": ("submission_pairs.csv", submission_schema),
        "sample_submission": ("sample_submission.csv", sample_sub_schema),
    }

    # hepsini Parquet'e ceviriyoruz:
    for name, (filename, schema) in files.items():
        csv_path = os.path.join(data_dir, filename)
        parquet_path = os.path.join(data_dir, f"{name}.parquet")

        df = pl.read_csv(csv_path, schema=schema)
        df.write_parquet(parquet_path)