# Power BI Dashboard Guide

## Import
Use **Get Data → Text/CSV** and import:
- `outputs/powerbi_orders.csv`
- `outputs/model_metrics.csv`
- `outputs/summary_metrics.csv`
- `outputs/feature_importance.csv`
- `outputs/confusion_matrix.csv`

## Page 1 — Executive Overview
Suggested visuals:
- Card: Total Orders (count of Record_ID)
- Card: Late-risk orders (sum of Actual_Late_Risk)
- Card: Late-risk rate
- Card: High-risk predictions
- Donut: Risk_Band
- Bar chart: Predicted risk by shipping mode (if that column exists)
- Column/line chart: Orders by month (if an order date field exists)

## Page 2 — Logistics Performance
- Bar chart: Actual late-risk by Shipping Mode
- Bar chart: Actual late-risk by Market / Region
- Matrix: Region × Shipping Mode
- Slicers: Market, Region, Category Name, Shipping Mode

## Page 3 — ML Delivery Risk
- Card: High-risk shipments
- Donut: Low / Medium / High risk
- Histogram/column chart: Late_Risk_Probability
- Table: Record_ID, Predicted_Late_Risk, Late_Risk_Probability, Risk_Band
- Filter: Risk_Band = High

## Page 4 — Model Performance
Use `model_metrics.csv` for metric cards or a bar chart.
Use `confusion_matrix.csv` for a matrix visual.
Use `feature_importance.csv` for a horizontal bar chart of top 10 drivers.

## Useful DAX measures
Create these measures on `powerbi_orders`:

```DAX
Total Orders = COUNTROWS(powerbi_orders)

Late Risk Orders =
CALCULATE(COUNTROWS(powerbi_orders), powerbi_orders[Actual_Late_Risk] = 1)

Not Late Risk Orders =
CALCULATE(COUNTROWS(powerbi_orders), powerbi_orders[Actual_Late_Risk] = 0)

Late Risk % =
DIVIDE([Late Risk Orders], [Total Orders], 0)

High Risk Predictions =
CALCULATE(COUNTROWS(powerbi_orders), powerbi_orders[Risk_Band] = "High")

Average Predicted Risk =
AVERAGE(powerbi_orders[Predicted_Late_Risk_Probability])
```

Column names may differ depending on the dataset version. Use the Fields pane to select the matching columns.
