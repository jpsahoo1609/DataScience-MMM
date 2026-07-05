# Mixed Effects Modeling for Marketing Mix Analysis (MMM)

## Overview

This project demonstrates the application of a Mixed-Effects Model to measure the impact of marketing activities on sales while accounting for store-level variability.

Traditional linear regression assumes that all stores behave similarly. However, in real-world retail and marketing environments, sales observations are naturally grouped by stores, regions, or markets, which introduces hierarchical structure into the data.

To address this, a Mixed-Effects Model was used to separate:

- Marketing-driven sales impact (Fixed Effects)
- Store-specific performance differences (Random Effects)

---

## Business Problem

A retail organization wants to understand:

> How much do advertising spend and promotional campaigns contribute to sales growth?

At the same time, some stores consistently outperform others because of factors such as:

- Location
- Customer demographics
- Store size
- Operational efficiency
- Local market conditions

Ignoring these differences can lead to biased estimates of marketing effectiveness.

---

## Project Objective

Build a model that:

1. Quantifies the impact of Ad Spend on Sales
2. Measures promotional sales lift
3. Accounts for store-level heterogeneity
4. Compares traditional regression with Mixed Effects Modeling
5. Generates actionable business insights for budget allocation

---

## Dataset

A synthetic retail dataset was created to simulate a realistic Marketing Mix Modeling (MMM) use case.

### Features

| Variable | Description |
|-----------|------------|
| Store | Store identifier |
| AdSpend | Marketing investment |
| Promotion | Promotion flag (0/1) |
| Sales | Sales revenue |

### Sales Generation Logic

Sales were simulated using:

```text
Sales =
Base Sales
+ Ad Spend Effect
+ Promotion Effect
+ Store-Specific Effect
+ Random Noise
```

This structure allows the model to learn both marketing effects and store-level variability.

---

## Methodology

### Step 1: Exploratory Data Analysis

- Distribution analysis
- Store-level sales comparison
- Validation of hierarchical structure

### Step 2: Baseline Linear Regression

Model:

```text
Sales ~ AdSpend + Promotion
```

Purpose:

- Establish baseline relationships
- Estimate marketing impact without accounting for store variation

### Step 3: Mixed Effects Model

Model:

```text
Sales ~ AdSpend + Promotion + (1 | Store)
```

Where:

#### Fixed Effects

- AdSpend
- Promotion

These represent the average marketing impact across all stores.

#### Random Effects

- Store

Captures store-specific deviations from the overall average.

---

## Why Mixed Effects?

Sales observations within the same store are not fully independent.

A traditional OLS model assumes:

```text
All stores share the same baseline sales level.
```

This assumption is unrealistic in most retail environments.

Mixed Effects Modeling allows us to:

- Control for store-specific variability
- Produce more reliable marketing impact estimates
- Improve model interpretability
- Better reflect real-world business structures

---

## Key Insights

### Marketing Impact

Advertising spend showed a positive relationship with sales, indicating that increased investment contributes to revenue growth.

### Promotion Effectiveness

Promotional campaigns generated measurable sales uplift across stores.

### Store-Level Heterogeneity

Even after controlling for marketing activities, some stores consistently outperformed others, demonstrating the importance of accounting for random effects.

### Business Value

The model helps decision-makers:

- Allocate marketing budgets more effectively
- Identify high-performing stores
- Detect underperforming locations
- Improve marketing ROI measurement

---

## Tech Stack

- Python
- Pandas
- NumPy
- Statsmodels
- Seaborn
- Matplotlib

---

## Skills Demonstrated

- Marketing Mix Modeling (MMM)
- Regression Analysis
- Mixed Effects Models
- Fixed vs Random Effects
- Statistical Modeling
- Business Insight Generation
- Model Interpretation
- Retail Analytics

---

## Repository Structure

```text
├── notebook.ipynb
├── README.md
└── requirements.txt
```

---

## Interview Takeaway

This project demonstrates how Mixed-Effects Models can be used in Marketing Mix Modeling scenarios to estimate the impact of marketing activities while accounting for hierarchical business structures such as stores, regions, or markets. The approach produces more realistic estimates than traditional regression and better supports marketing budget optimization decisions.
