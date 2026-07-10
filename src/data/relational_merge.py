import polars as pl

###
#Görev 2) Flattening
###

# Hepsini RAM'e yüklemek yerine lazy okuzoruz:
training_pairs = pl.scan_parquet("training_pairs.parquet")
terms = pl.scan_parquet("terms.parquet")
items = pl.scan_parquet("items.parquet")

#ilk join: training_pairs + terms, term_id üzerinden
flattened = training_pairs.join(
    terms, 
    on="term_id",
    how="left" 
)

#ikinci join: ara sonuc + items, item_id üzerinden
flattened = flattened.join(
    items,
    on="item_id",
    how="left"
)

#Null degerleri hallediyoruu:
flattened = flattened.with_columns(
    pl.col("query").fill_null("unknown"),
    pl.col("title").fill_null("unknown"),
    pl.col("category").fill_null("unknown"),
    pl.col("brand").fill_null("unknown"),
    pl.col("gender").fill_null("unknown"),
    pl.col("age_group").fill_null("unknown"),
    pl.col("attributes").fill_null("unknown"),
)

#sonunda sonuc üzeri deneme:
result = flattened.collect()
result.write_parquet("training_flattened.parquet")
