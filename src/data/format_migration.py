import polars as pl

###
#Görev 1) Veri Formatlama
###

#tahmin ettirmeden typelari biz veriyoruz. farkliligin az oldugu yerlerde Utf8(string) yerine categorical kullaniyoruz ki  tasarruf edelim
items_schema = {
    "item_id" : pl.Utf8,
    "title" : pl.Utf8,
    "category" : pl.Utf8,
    "brand" : pl.Categorical,
    "gender" : pl.Categorical,
    "age_group" : pl.Categorical,
    "attributes" : pl.Utf8
}

terms_schema = {
    "term_id" : pl.Utf8,
    "query" : pl.Utf8
}

training_schema = {
    "id" : pl.Utf8,
    "term_id" : pl.Utf8,
    "item_id" : pl.Utf8,
    "label" : pl.Int64 #neden int 64? sadece 1se neden int8 ya da daha iyisi bool degil?
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

#hepsini Parquete ceviriyoruz:
for name, (path, schema) in files.items():
    df = pl.read_csv(path, schema=schema)
    df.write_parquet(f"{name}.parquet")
    #print(name, df.shape, df.estimated_size("mb"), "MB") #beklenen boyut