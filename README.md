# Supply Chain & Logistics Analytics — ML + Power BI

This project predicts shipment delay risk and exports analysis-ready files for Power BI.

## 1. Dataset
Download the **DataCo Smart Supply Chain Dataset** and place the CSV in:
`dataset/DataCoSupplyChainDataset.csv`

Dataset source: https://data.mendeley.com/datasets/8gx2fvg2k6/5

The pipeline also accepts a differently named CSV if you pass its path to the script.

## 2. Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Run the project
From the project root:

```bash
python src/train_model.py
```

This creates:
- `models/delivery_delay_pipeline.joblib` — preprocessing + trained model
- `outputs/predictions.csv` — each row with actual and predicted risk
- `outputs/model_metrics.csv` — accuracy, precision, recall, F1, ROC-AUC
- `outputs/confusion_matrix.csv`
- `outputs/feature_importance.csv`
- `outputs/powerbi_orders.csv` — clean data for Power BI
- `outputs/summary_metrics.csv` — KPI values

If your CSV is elsewhere:
```bash
python src/train_model.py --data "C:/path/to/your/file.csv"
```

## 4. Open in Power BI
1. Open Power BI Desktop.
2. Select **Get data → Text/CSV**.
3. Import `outputs/powerbi_orders.csv`.
4. Optionally import `outputs/predictions.csv`, `outputs/model_metrics.csv`,
   `outputs/feature_importance.csv`, and `outputs/summary_metrics.csv`.
5. Build the report using the dashboard plan in `powerbi/DASHBOARD_GUIDE.md`.

`powerbi_orders.csv` includes model predictions joined by the original row order.
For reliable matching, the pipeline adds `Record_ID`.

## 5. Suggested dashboard pages
- Executive Overview
- Logistics Performance
- Delivery Risk Predictions
- Model Performance & Drivers

## Notes
- The target is `Late_delivery_risk` (0 = not late-risk, 1 = late-risk) when present.
- The model excludes known post-delivery/outcome fields such as actual delivery date,
  delivery status, and days for shipping (real).
- This is a portfolio/academic demonstration, not a production dispatch system.
- Check the dataset's license and attribution requirements before redistribution.
