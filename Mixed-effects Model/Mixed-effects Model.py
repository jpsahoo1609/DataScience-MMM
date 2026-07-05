# Databricks notebook source
# MAGIC %md
# MAGIC #Business Problem
# MAGIC
# MAGIC A retail company wants to understand the true impact of marketing investments on sales across multiple stores.
# MAGIC
# MAGIC While stores receive different levels of advertising spend and promotional activities, they also have inherent differences such as location, store size, and operational efficiency. These store-specific factors can influence sales performance and may lead to biased estimates when using a traditional linear regression model.
# MAGIC
# MAGIC **Objective**\
# MAGIC 1- Build a baseline OLS regression model to estimate the relationship between marketing variables and sales.\
# MAGIC 2- Identify the limitations of treating all stores as identical.\
# MAGIC 3- Use a **Mixed-Effects Model** with:
# MAGIC
# MAGIC
# MAGIC     Fixed Effects: AdSpend, Promotion
# MAGIC     Random Effect: Store
# MAGIC
# MAGIC
# MAGIC 4- Measure the overall contribution of marketing activities to sales.\
# MAGIC 5- Estimate store-specific effects after controlling for marketing activities.\
# MAGIC 6- Generate business insights for marketing budget allocation and store performance optimization.
# MAGIC
# MAGIC

# COMMAND ----------

import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000

stores = np.random.choice(
    ['A','B','C','D','E'],
    n
)

# Saying that , okay this store creates more and less sales 
store_effect = {
    'A':10000,
    'B':3000,
    'C':0,
    'D':-500,
    'E':-1100
}

adspend = np.random.normal(
    1000,
    200,
    n
)

promotion = np.random.binomial(
    1,
    0.3,
    n
)

sales = (
    5000
    + 2*adspend
    + 500*promotion
    + [store_effect[s] for s in stores]
    + np.random.normal(0,300,n)
)

df = pd.DataFrame({
    'Store':stores,
    'AdSpend':adspend,
    'Promotion':promotion,
    'Sales':sales
})

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC "I wanted to measure the impact of Ad Spend and Promotions on Sales while accounting for differences between stores.
# MAGIC

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Before modeling, I explored the distribution of sales and checked whether there were meaningful differences across stores. This helped justify the use of store-level random effects.If all stores look identical:.No need for Mixed Effects.

# COMMAND ----------

df.groupby("Store")["Sales"].mean()


# COMMAND ----------

df.groupby("Store")["AdSpend"].sum()

# COMMAND ----------

df.groupby("Store")["Promotion"].sum()

# COMMAND ----------

import seaborn as sns 
import matplotlib.pyplot as plt 

sns.boxplot(data=df,x="Store",y="Sales")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Clear variation in average sales across stores, indicating to build a model and find out which factors affect and how much and is there a random effect(in this case'store effect')?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Building a regression

# COMMAND ----------

# MAGIC %md
# MAGIC Train-test splitting evaluates predictive performance, whereas mixed-effects modeling is used to properly model hierarchical data. Since observations were grouped by store, I used a mixed-effects model to estimate both the overall impact of marketing variables and the store-level variability simultaneously.

# COMMAND ----------

# MAGIC %md
# MAGIC # Statsmodel way

# COMMAND ----------

# DBTITLE 1,Install statsmodels
# MAGIC %pip install statsmodels

# COMMAND ----------

import statsmodels.formula.api as smf
ols = smf.ols("Sales ~ AdSpend + Promotion",data=df).fit()
print(ols.summary())


# COMMAND ----------

# MAGIC %md
# MAGIC # Scikit-learn way

# COMMAND ----------

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
X = df[["AdSpend","Promotion"]]
y = df["Sales"]

model = lr.fit(X,y)
print(model.intercept_)
print(model.coef_)

# COMMAND ----------

# MAGIC %md
# MAGIC Model says if promotion increases then sales increases. But for Store A - promotion done is 121 , same as store D.But hte sales has a huge difference. There is some other factor/randomness we need to find out.

# COMMAND ----------

# MAGIC %md
# MAGIC # Building mixed effects model 

# COMMAND ----------

# MAGIC %md
# MAGIC Now we tell the model- 
# MAGIC
# MAGIC AdSpend and Promotion are fixed effects.
# MAGIC Store is a random effect.

# COMMAND ----------

mixed =  smf.mixedlm("Sales ~ AdSpend + Promotion",data=df, groups=df["Store"])
result = mixed.fit()
print(result.summary())

# COMMAND ----------

mixed =  smf.mixedlm("Sales ~ AdSpend + Promotion",data=df, groups=df["Store"])
result = mixed.fit(reml=False)
print(result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC If we will not use "result = mixed.fit(reml=False)" , it will fit using REML(Restricted Maximum Likelihood)- means the model did not converge properly.

# COMMAND ----------

# MAGIC %md
# MAGIC grouped "Store" to say my model that - Rows belonging to the same store
# MAGIC should be treated as related.

# COMMAND ----------

# MAGIC %md
# MAGIC # Fixed Effects
# MAGIC AdSpend- For every $1 increase in AdSpend,
# MAGIC sales increase by approximately $2.03.\
# MAGIC Promotion - Running a Promotion, 
# MAGIC increases sales by approx 478units

# COMMAND ----------

# MAGIC %md
# MAGIC # Random effects

# COMMAND ----------

#randomness
result.random_effects

# COMMAND ----------

# MAGIC %md
# MAGIC Store A performs better than average. +7710\
# MAGIC Store E performs worse than average. -3378

# COMMAND ----------

# MAGIC %md
# MAGIC Random effects capture store-specific deviations from the overall sales baseline after accounting for ad spend and promotions.

# COMMAND ----------

# MAGIC %md
# MAGIC # OLS vs Mixed Effects 

# COMMAND ----------

print("OLS AIC:", ols.aic)
print("Mixed AIC:", result.aic)

# Lower the AIC,better the model.

# COMMAND ----------

# MAGIC %md
# MAGIC Taking consideration for store-level variability improved model fit.\
# MAGIC Mixed Model fits better.

# COMMAND ----------

