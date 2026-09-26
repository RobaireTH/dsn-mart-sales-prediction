# DSN Mart sales prediction

My solution for the DSN Bootcamp Qualification Hackathon 2026 (ML track) on Kaggle. The task is to predict `total_sales` for a product in a store, scored with RMSE.

The notebook `dsn_mart_sales_prediction.ipynb` has the EDA, the modelling decisions, the final model and the experiments that didn't make it. It's saved with outputs.

## Approach

Sales are roughly the product's price times the units sold, and units depend mostly on the store. The final model blends:

- 65% a simple model: average units per store × a per-product price estimate (midpoint of its observed prices), with a small shrunk adjustment per product
- 35% CatBoost and XGBoost on the cleaned features

CV RMSE is 1067.90 (5-fold, repeated 3 times).

| Submission | Change | Public LB |
|---|---|---|
| v5 | simple model + tree blend | 1071.15 |
| v6 | product mean price | 1068.76 |
| v7 | midpoint price | 1068.65 |
| v8 | mean units per store, fixed blend weights | 1068.58 |

`submissions/` has the four files. The notebook reproduces v8 exactly on macOS (Apple Silicon). On Kaggle, XGBoost builds slightly different trees, and that run scored 1068.69.

## Running it

The competition data isn't included. Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition's Data tab into `data/`, then:

```bash
pip install -r requirements.txt
jupyter notebook dsn_mart_sales_prediction.ipynb
```

On macOS, XGBoost and LightGBM need OpenMP (`brew install libomp`).
