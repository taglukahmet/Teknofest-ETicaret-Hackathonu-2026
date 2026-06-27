# Teknofest-ETicaret-Hackathonu-2026

## Dosya yapısı bu şekilde olacaktır:
```text
Teknofest-ETicaret-Hackathonu-2026/
│
├── requirements.txt
├── README.md
│
├── data_raw/                  # (Folder excluded from Git via .gitignore)
│   ├── items.csv
│   ├── terms.csv
│   └── training_pairs.csv
│
└── src/
    ├── __init__.py
    ├── config.py              # Configuration & Hyperparameters (Owned by: Yusuf)
    │
    ├── data/                  # --- THE DATA PIPELINE ---
    │   ├── __init__.py
    │   ├── format_migration.py # Parquet conversion (Owned by: Yağız)
    │   ├── relational_merge.py # SQL-style ID merging (Owned by: Yağız)
    │   ├── negative_sample.py  # Hard/Random YOK generation (Owned by: Mine)
    │   ├── text_serialize.py   # String template construction (Owned by: Yağız)
    │   └── kfold_split.py      # GroupKFold isolation (Owned by: Mine)
    │
    ├── models/                # --- THE TRANSFORMER ARCHITECTURE ---
    │   ├── __init__.py
    │   ├── architecture.py     # Cross-Encoder Model Class (Owned by: Ahmet)
    │   └── inference.py        # Blind test prediction execution (Owned by: Ahmet)
    │
    ├── training/              # --- THE PYTORCH ENGINE ---
    │   ├── __init__.py
    │   ├── dataset.py          # Arrow memory-mapped DataLoader (Owned by: İrem)
    │   └── train_loop.py       # Backpropagation & W&B hooks (Owned by: Member İrem & Yusuf(for W&B))
    │
    └── post_processing/       # --- THE OPTIMIZATION LAYER ---
        ├── __init__.py
        ├── metrics.py          # Threshold loop optimization (Owned by: Yusuf)
        └── ensemble.py         # OOF Soft Voting (Owned by: Yusuf)


```
