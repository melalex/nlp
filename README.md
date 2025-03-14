# UNLP 2025 Challenge. Manipulation detection and classification.

* [Task Description](./references/task_description.md)
* [Planing Notes](./references/planning.md)

## Environment setup

Run:
```
make venv
```

## Project structure

```
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── models             <- Trained and serialized models, model predictions, or model summaries
├── notebooks          <- Jupyter notebooks. 
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
├── requirements.txt   <- The requirements file for reproducing the analysis environment
└── src                <- Source code for use in this project.
    ├── __init__.py  <- Makes src a Python module
    ├── data         <- Scripts to download or generate data
    │   └── make_dataset.py
    ├── features     <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    ├── models       <- Scripts to train models and then use trained models to make predictions
    │   ├── predict_model.py
    │   └── train_model.py
    └── visualization <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

## Naming convention

### Jupyter notebooks

Naming convention is a number (for ordering), the creator's initials, and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.

## Branching Strategy

Branch per feature. Name: dev/{initials}/{feature_name}