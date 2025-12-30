# Power BI Dashboard Specifications
## Credit Risk Prediction Dashboard

### Overview
This document provides specifications for building the Credit Risk Prediction dashboard in Power BI. The dashboard visualizes key risk metrics, model performance, and portfolio analysis.

---

## Data Sources

### Primary Data Files
1. **credit_features.csv** - Located in `data/features/`
   - Contains all engineered features and predictions
   - Key columns: customer_id, credit_score, dti_ratio, loan_amount, default, composite_risk_score, risk_category

2. **evaluation_metrics.json** - Located in `outputs/reports/`
   - Contains model performance metrics
   - Key metrics: AUC-ROC, Gini, Precision, Recall, F1-Score

---

## Dashboard Pages

### Page 1: Executive Summary

**KPI Cards (Top Row)**
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Total Loans    │  │  Default Rate   │  │  Model AUC      │  │  Portfolio      │
│    10,000       │  │     15.2%       │  │     0.86        │  │  $45.2M         │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Visuals:**
1. **Risk Distribution Donut Chart**
   - Measure: Count of customers
   - Legend: risk_category (Low, Medium, High, Very High)
   - Colors: Green → Yellow → Orange → Red

2. **Default Rate by Credit Tier (Clustered Bar Chart)**
   - Axis: credit_tier
   - Values: Default Rate (% of defaults)
   - Sort: Credit tier order (Poor → Excellent)

3. **Monthly Application Trend (Line Chart)**
   - Axis: application_date (Month)
   - Values: Count of applications, Default count
   - Legend: Application vs Defaults

4. **Loan Portfolio by Purpose (Treemap)**
   - Group: loan_purpose
   - Values: Sum of loan_amount
   - Color saturation: Default rate

---

### Page 2: Risk Analysis

**Visuals:**

1. **Risk Score Distribution (Histogram)**
   - Values: composite_risk_score
   - Bins: 10-20 bins
   - Reference line: Mean risk score

2. **Default Rate by Risk Category (Stacked Bar)**
   - Axis: risk_category
   - Values: Default count, Non-default count
   - Data labels: Percentage

3. **Credit Score vs Default Rate (Scatter Plot)**
   - X-axis: credit_score
   - Y-axis: default (aggregated as mean)
   - Trend line: Polynomial

4. **DTI Ratio Impact (Area Chart)**
   - Axis: dti_ratio (binned)
   - Values: Default rate
   - Show trend

5. **Risk Heatmap (Matrix)**
   - Rows: credit_tier
   - Columns: income_bracket
   - Values: Average default rate
   - Conditional formatting: Green (low) to Red (high)

---

### Page 3: Model Performance

**Visuals:**

1. **Model Metrics Card**
```
┌────────────────────────────────────────────────┐
│           MODEL PERFORMANCE METRICS            │
├────────────────┬───────────────────────────────┤
│ AUC-ROC        │ 0.86                          │
│ Gini           │ 0.72                          │
│ Precision      │ 0.78                          │
│ Recall         │ 0.72                          │
│ F1-Score       │ 0.75                          │
│ KS Statistic   │ 0.52                          │
└────────────────┴───────────────────────────────┘
```

2. **ROC Curve Image**
   - Import from: outputs/figures/roc_curve.png
   - Size: 400x400 pixels

3. **Confusion Matrix Image**
   - Import from: outputs/figures/confusion_matrix.png
   - Size: 350x300 pixels

4. **Feature Importance (Bar Chart)**
   - Data: Top 10 features
   - Axis: Feature name
   - Values: Importance score
   - Sort: Descending
   - Color: Gradient based on importance

5. **Score Distribution Image**
   - Import from: outputs/figures/score_distribution.png

---

### Page 4: Customer Segmentation

**Visuals:**

1. **Customer Segments Table**
   - Columns: Segment, Count, Avg Loan, Avg Credit Score, Default Rate
   - Segments based on risk_category

2. **Age Group Analysis (Clustered Column)**
   - Axis: age_group
   - Values: Default rate, Average loan amount (secondary axis)

3. **Employment Impact (Combo Chart)**
   - Axis: employment_length (binned)
   - Column: Count of customers
   - Line: Default rate

4. **Geographic Distribution (Map)** *(if state data available)*
   - Or use **Home Ownership Analysis**:
   - Axis: home_ownership
   - Values: Default rate, Count

5. **Loan Purpose Risk (Waterfall Chart)**
   - Category: loan_purpose
   - Values: Default rate deviation from average

---

### Page 5: Portfolio Deep Dive

**Slicers:**
- Date Range (application_date)
- Risk Category (risk_category)
- Credit Tier (credit_tier)
- Loan Purpose (loan_purpose)
- Loan Amount Range (loan_amount)

**Visuals:**

1. **Detailed Customer Table**
   - Columns: Customer ID, Credit Score, Loan Amount, Risk Score, Risk Category, Default Status
   - Conditional formatting on Risk Score
   - Drill-through enabled

2. **Loan Amount Distribution (Box Plot)**
   - Category: risk_category
   - Values: loan_amount

3. **Interest Rate Analysis (Scatter)**
   - X: interest_rate
   - Y: default_rate
   - Size: loan_amount
   - Color: risk_category

4. **Credit Utilization Impact (Gauge Charts)**
   - Three gauges for Low/Medium/High utilization
   - Show default rates

---

## DAX Measures

```dax
// Total Loans
Total Loans = COUNTROWS('credit_features')

// Default Rate
Default Rate = 
DIVIDE(
    CALCULATE(COUNTROWS('credit_features'), 'credit_features'[default] = 1),
    COUNTROWS('credit_features'),
    0
)

// Total Portfolio Value
Total Portfolio = SUM('credit_features'[loan_amount])

// Average Credit Score
Avg Credit Score = AVERAGE('credit_features'[credit_score])

// High Risk Count
High Risk Count = 
CALCULATE(
    COUNTROWS('credit_features'),
    'credit_features'[risk_category] IN {"High", "Very High"}
)

// Risk Rate
Risk Rate = 
DIVIDE([High Risk Count], [Total Loans], 0)

// YoY Default Change
YoY Default Change = 
VAR CurrentYear = CALCULATE([Default Rate], YEAR('credit_features'[application_date]) = YEAR(TODAY()))
VAR PriorYear = CALCULATE([Default Rate], YEAR('credit_features'[application_date]) = YEAR(TODAY()) - 1)
RETURN CurrentYear - PriorYear

// Loan to Income Ratio
Avg LTI Ratio = AVERAGE('credit_features'[loan_to_income])

// Expected Loss
Expected Loss = 
SUMX(
    'credit_features',
    'credit_features'[loan_amount] * ('credit_features'[composite_risk_score] / 100) * 0.4
)
```

---

## Color Scheme

```
Primary Colors:
- Deep Blue: #1E3A5F (Headers, Primary elements)
- Light Blue: #4A90D9 (Secondary elements)
- White: #FFFFFF (Background)

Risk Colors:
- Low Risk: #2ECC71 (Green)
- Medium Risk: #F1C40F (Yellow)
- High Risk: #E67E22 (Orange)
- Very High Risk: #E74C3C (Red)

Neutral:
- Gray: #7F8C8D
- Light Gray: #ECF0F1
```

---

## Interactivity

### Cross-Filtering
- All visuals on same page should cross-filter
- Exception: KPI cards (no cross-filter)

### Drill-Through
- Enable drill-through from any customer segment to detailed customer view
- Right-click context menu enabled

### Tooltips
- Custom tooltips showing:
  - Customer count
  - Average loan amount
  - Default rate
  - Risk distribution

### Bookmarks
1. "Executive View" - High-level KPIs only
2. "Risk Focus" - Risk-related visuals
3. "Performance View" - Model metrics
4. "Full Dashboard" - All elements visible

---

## Data Refresh

- **Frequency**: Daily
- **Schedule**: 6:00 AM (before business hours)
- **Source**: Import mode (for performance)
- **Incremental Refresh**: Enabled for application_date

---

## Publishing

1. Publish to Power BI Service workspace: "Credit Risk Analytics"
2. Set up Row-Level Security (RLS) if needed
3. Create Power BI App for distribution
4. Enable mobile view for executive access

---

## Notes for Implementation

1. Before importing data, run the complete Python pipeline to generate all data files
2. Ensure all PNG files from /outputs/figures/ are imported as images
3. Create calculated columns in Power BI for any derived fields not in source data
4. Test all cross-filtering and drill-through functionality
5. Optimize for performance by removing unused columns

---

*Dashboard Specification v1.0*
*Credit Risk Prediction Project*
