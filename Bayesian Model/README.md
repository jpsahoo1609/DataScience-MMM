# MMM2 - Bayesian Sales Driver Analysis Using Walmart Data

## Project Overview

This project demonstrates the use of **Bayesian Regression** to identify and quantify the key drivers of Walmart weekly sales.

Unlike traditional Linear Regression, Bayesian Modeling not only estimates the impact of each feature but also quantifies the uncertainty associated with that impact through credible intervals.

The goal was to answer:

- What factors influence weekly sales?
- How strongly do they influence sales?
- How confident are we in those findings?
- Can historical sales behavior improve forecasting?

---

## Business Problem

Retail organizations need to understand the factors affecting sales performance so that they can make better decisions around:

- Demand Forecasting
- Pricing Strategy
- Promotion Planning
- Inventory Management
- Store Operations

Using Walmart Sales data, a Bayesian Regression model was built to identify the most influential business drivers of weekly sales.

---

## Dataset

The analysis was performed using the Walmart Sales Forecasting dataset.

Data Sources:

- train.csv
- features.csv
- stores.csv

Target Variable:

```text
Weekly_Sales
```

Features Used:

```text
Temperature
Fuel_Price

MarkDown1
MarkDown2
MarkDown3
MarkDown4
MarkDown5

CPI
Unemployment

Size

Weekly_sales_lag_1
Rolling_4
```

---

## Feature Engineering

### Weekly Sales Lag

```text
Weekly_sales_lag_1
```

Represents the previous week's sales.

Purpose:

- Capture temporal dependency
- Understand how historical sales influence future sales

---

### Rolling 4 Week Average

```text
Rolling_4
```

Represents the average sales of the previous four weeks.

Purpose:

- Capture sales trends
- Smooth short-term fluctuations
- Improve forecasting stability

---

### Standardization

Used:

```python
StandardScaler()
```

Purpose:

- Normalize feature scales
- Improve Bayesian sampler convergence
- Prevent large magnitude features from dominating the model

---

## Bayesian Regression Model

The model assumes:

```text
Weekly Sales

=

Intercept

+

Temperature Effect

+

Fuel Price Effect

+

Promotion Effects

+

Economic Effects

+

Historical Sales Effects

+

Noise
```

Implemented using:

```python
PyMC
```

---

## Why Bayesian Regression?

Traditional Linear Regression gives:

```text
Fuel Price Effect = -0.40
```

Bayesian Regression gives:

```text
Fuel Price Effect = -0.40

95% Credible Interval

[-0.69 , -0.10]
```

This allows business users to understand:

- Estimated impact
- Confidence around the estimate
- Risk associated with business decisions

---

## Key Findings

### Positive Sales Drivers

- Temperature showed a strong positive relationship with Weekly Sales.
- CPI showed a positive relationship with Weekly Sales.
- Historical sales patterns contributed significantly to forecasting performance.

### Negative Sales Drivers

- Fuel Price showed a negative relationship with Weekly Sales.
- Multiple markdown variables showed negative relationships, indicating possible promotional activity during low-demand periods.

### Historical Demand Patterns

The strongest business insight was the importance of historical demand.

Features such as:

```text
Weekly_sales_lag_1
Rolling_4
```

helped capture recurring sales behavior and demand trends.

---

## Business Impact

This solution helps stakeholders:

### Demand Forecasting

Predict future weekly sales using:

- Historical sales
- Promotions
- Economic indicators
- Store characteristics

### Driver Identification

Understand:

- Which variables increase sales
- Which variables decrease sales
- Which variables have uncertain impact

### Risk-Aware Decision Making

Bayesian Modeling provides confidence intervals, enabling more informed business decisions compared to traditional point estimates.

---

## Tech Stack

```text
Python
PySpark
Pandas
NumPy
Scikit-Learn
PyMC
ArviZ
```

---

## Project Workflow

```text
Data Loading
      ↓
Data Merging
      ↓
Feature Engineering
      ↓
Lag Features
      ↓
Rolling Features
      ↓
Feature Scaling
      ↓
Bayesian Regression
      ↓
Posterior Analysis
      ↓
Business Insights
```

---

## Key Learning

This project demonstrates how Bayesian techniques can move beyond simple prediction and help businesses answer:

- What drives sales?
- How strongly does it drive sales?
- How confident are we in that conclusion?

These concepts form the foundation for advanced Marketing Mix Modeling (MMM), Bayesian MMM, and Hierarchical Bayesian Modeling.

---

## Future Enhancements

- Bayesian Marketing Mix Modeling (MMM)
- Adstock Transformations
- Saturation Curves
- Hierarchical Bayesian Models
- Multi-Store Effects
- Marketing Attribution
- Budget Optimization
