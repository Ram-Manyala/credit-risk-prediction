# Credit Risk Dashboard - Power BI Template Setup

This folder contains specifications and data exports for creating Power BI dashboards.

## Files

1. **dashboard_specifications.md** - Detailed specifications for building the Power BI dashboard
2. **credit_risk_dashboard.pbit** - Power BI template file (to be created)

## Quick Setup Guide

### Step 1: Run the Python Pipeline
```bash
cd credit-risk-prediction
python -m src.main
```

This generates:
- `data/features/credit_features.csv` - Main data for dashboard
- `outputs/figures/*.png` - Visualization images
- `outputs/reports/evaluation_metrics.json` - Model metrics

### Step 2: Open Power BI Desktop
1. Open Power BI Desktop
2. Get Data → Text/CSV → Select `data/features/credit_features.csv`
3. Transform data if needed in Power Query Editor
4. Load data

### Step 3: Import Images
1. Insert → Image
2. Navigate to `outputs/figures/`
3. Import: roc_curve.png, confusion_matrix.png, feature_importance.png, etc.

### Step 4: Create Visualizations
Follow the specifications in `dashboard_specifications.md` to create:
- Executive Summary page
- Risk Analysis page
- Model Performance page
- Customer Segmentation page
- Portfolio Deep Dive page

### Step 5: Add DAX Measures
Copy DAX formulas from specifications to create calculated measures.

### Step 6: Apply Formatting
- Use the color scheme defined in specifications
- Add company logo if available
- Ensure consistent fonts and sizing

### Step 7: Publish
1. File → Publish → Publish to Power BI
2. Select workspace
3. Configure scheduled refresh

## Data Dictionary

| Column | Description | Type |
|--------|-------------|------|
| customer_id | Unique identifier | Text |
| credit_score | Credit score (300-850) | Number |
| loan_amount | Loan amount in USD | Currency |
| interest_rate | Interest rate % | Decimal |
| dti_ratio | Debt-to-income ratio | Decimal |
| composite_risk_score | Risk score (0-100) | Number |
| risk_category | Risk level category | Text |
| default | Default status (0/1) | Number |

## Support

For questions about the data or dashboard, refer to the main project README.
