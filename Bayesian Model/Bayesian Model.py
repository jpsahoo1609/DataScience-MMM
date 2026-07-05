# Databricks notebook source
# MAGIC %md
# MAGIC # Business Problem
# MAGIC
# MAGIC **Walmart** wants to understand which factors influence weekly sales and quantify the uncertainty around those factors so that business teams can make more confident decisions regarding inventory planning and promotions.
# MAGIC
# MAGIC **Objective**
# MAGIC
# MAGIC 1. What factors drive weekly sales?
# MAGIC
# MAGIC 2. How strongly do they affect sales?
# MAGIC
# MAGIC 3. How confident are we about those effects?
# MAGIC
# MAGIC 4. Can previous week's sales help predict future sales?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading Data 

# COMMAND ----------

df_features = spark.read.format('csv').option('inferSchema','true').option('header','true').load('/Volumes/workspace/default/databricks_firstvolume/features_walmart.csv')
df_sales = spark.read.format('csv').option('inferSchema','true').option('header','true').load('/Volumes/workspace/default/databricks_firstvolume/train.csv')
df_store = spark.read.format('csv').option('inferSchema','true').option('header','true').load('/Volumes/workspace/default/databricks_firstvolume/stores_walmart.csv')

df_features.display()
df_sales.display()
df_store.display()

# COMMAND ----------

# MAGIC %pip install pymc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merging for a single Dataframe 

# COMMAND ----------

df = df_sales.join(df_features, on=['Store','Date','isHoliday'], how='inner')
df = df.join(df_store, on=['Store'], how='inner')
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Inspection

# COMMAND ----------

df.count()

# COMMAND ----------

df.show(5)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col 
for c in df.columns:
    null_count = df.filter(col(c).isNull()).count()
    print(c,null_count )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Date Features 

# COMMAND ----------

from pyspark.sql.functions import year,month,weekofyear
df = df.withColumn("Year",year("Date"))
df = df.withColumn("Month",month("Date"))
df = df.withColumn("Week",weekofyear("Date"))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lag Feature 

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import lag

w = Window.partitionBy("Store","Dept").orderBy("Date")
df = df.withColumn("Weekly_sales_lag_1", lag('Weekly_Sales',1).over(w))

# COMMAND ----------

# MAGIC %md
# MAGIC Weekly sales are recorded at the department level. Partitioning only by Store would mix sales across departments and produce incorrect lag values. Therefore lag features must be generated within each Store-Department combination to preserve the temporal sequence of the same business entity.

# COMMAND ----------

df =  df.orderBy("Store","Dept","Date")
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rolling Mean

# COMMAND ----------

from pyspark.sql.functions import avg
w = Window.partitionBy("Store","Dept").orderBy("Date").rowsBetween(-3,0)
df = df.withColumn("Rolling_4",avg("Weekly_Sales").over(w))

# COMMAND ----------

df.orderBy("Store","Dept","Date")
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC I created a 4-week rolling average feature to capture recent sales trends and smooth short-term fluctuations. This helps the model learn underlying demand patterns rather than reacting to weekly noise.
# MAGIC Lets take row 5 , this tells us that that week sales is much higher than the recent 4-week average.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop Nulls, lag create NULLS

# COMMAND ----------

df = df.dropna()
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time -Based Split

# COMMAND ----------

df.groupby("year").count().display()

# COMMAND ----------

from pyspark.sql.functions import col 
df_train = df.filter(col("Date")<"2012-01-01")
df_test = df.filter(col("Date")>="2012-01-01")
print(df_train.count())
print(df_test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection

# COMMAND ----------

features = ["Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment", "Size", "Weekly_sales_lag_1","Rolling_4",]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to Pandas , as PyMC works with pandas/numpy

# COMMAND ----------

pdf = df_train.select(features  + ["Weekly_Sales"]).toPandas()
pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standardize Features

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
import numpy as np

# Replace 'NA' strings with NaN
pdf[features] = pdf[features].replace('NA', np.nan)

scalar = StandardScaler()
X = scalar.fit_transform(pdf[features])
y = scalar.fit_transform(pdf[["Weekly_Sales"]]).flatten()


# COMMAND ----------

# MAGIC %md
# MAGIC I used StandardScaler to normalize features so that variables with large magnitudes do not dominate smaller-scale variables. In Bayesian modeling, scaling also helps the sampler converge more efficiently. fit() learns the mean and standard deviation from the training data, transform() applies the scaling, and fit_transform() performs both operations in a single step.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bayesian Regression

# COMMAND ----------

# MAGIC %md
# MAGIC intercept- sales ,before considering features \
# MAGIC beta- coefficients , beta[0]=Temperature effect , beta[1]=Fuel Price Effect \
# MAGIC why Normal Distrbution- temperature effect could be anything around 0 , after seeing data MODEL learns the actual value \
# MAGIC mu- equivalent to 'Predicted Sales' - intercept+pm.math.dot(X,beta)\
# MAGIC --Sales=  Intercept + B1(temperature) + B2(Fuel Price + B3(Lag Sales))--\
# MAGIC sigma- noise, i.e weather , local events , random facotrs, all became noise which is sigma \
# MAGIC oberved=y- actual sales \
# MAGIC pm.sample- integrates multiple possible values 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fixing errors

# COMMAND ----------

pdf[features].isnull().sum()

# COMMAND ----------

pdf[features] = pdf[features].fillna(0)
pdf

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
import numpy as np

# Replace 'NA' strings with NaN
pdf[features] = pdf[features].replace('NA', np.nan)

scalar = StandardScaler()
X = scalar.fit_transform(pdf[features])
y = scalar.fit_transform(pdf[["Weekly_Sales"]]).flatten()

# COMMAND ----------

import pymc as pm
with pm.Model() as model:
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    beta  =pm.Normal("beta", mu=0, sigma=10, shape=X.shape[1])
    sigma = pm.HalfNormal("sigma",sigma=10)
    mu = intercept + pm.math.dot(X,beta)
    sales = pm.Normal("Weekly_Sales",mu=mu,sigma=sigma,oberved=y)
    trace = pm.sample(10,tune=10)

# COMMAND ----------

# MAGIC %md
# MAGIC it deos not only check temperature vs sales , instead it asks- "After accounting for Fuel_Price, CPI, Unemployment, Lag Sales and Rolling Mean, what additional effect does Temperature have on Sales?"
# MAGIC In Bayesian Regression, the coefficient of a feature is estimated while simultaneously considering all other features in the model. Therefore, the Temperature coefficient represents the independent contribution of Temperature to sales after controlling for variables such as Fuel Price, CPI, Unemployment, and Lag Sales. This is different from correlation, which examines only the pairwise relationship between two variables.

# COMMAND ----------

# MAGIC %md
# MAGIC Bayesian regression , tries multiple combination of different features to see the best fit to the sales and then give hte result 

# COMMAND ----------

import arviz as az

az.summary(trace)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC              mean   hdi_3%  hdi_97%
# MAGIC
# MAGIC beta[0]      120      90      150
# MAGIC
# MAGIC beta[1]      -40     -60      -20
# MAGIC
# MAGIC beta[2]      0.75    0.70     0.80
# MAGIC

# COMMAND ----------

import arviz as az
print(
    az.summary(trace)
      .filter(like="beta", axis=0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # What I found
# MAGIC **1. Temperature**
# MAGIC mean- 0.95
# MAGIC CI = [0.93,0.97]
# MAGIC Temperature has a strong positive impact on sales. As temperature increases, Walmart sales tend to increase.
# MAGIC
# MAGIC **2. Fuel Price**
# MAGIC mean- -0.4
# MAGIC Higher fuel prices appear to reduce Walmart sales.
# MAGIC
# MAGIC **3. Unemployment**
# MAGIC mean- 0.7
# MAGIC Positive effect in this sample.
# MAGIC
# MAGIC **4.Weekly_sales_lag_1**
# MAGIC mean- 0.2
# MAGIC Previous week's sales may influence current sales, but the model is not yet confident.
# MAGIC
# MAGIC **5.  Rolling_4**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC I built a Bayesian regression model on Walmart weekly sales data to identify key sales drivers while quantifying uncertainty. The analysis showed that temperature, CPI, and unemployment had positive relationships with sales, while fuel prices and several markdown variables showed negative relationships. Bayesian modeling allowed me to evaluate not only the magnitude of these effects but also the confidence around them using credible intervals. This provides more actionable insights for business stakeholders compared to traditional regression models.
# MAGIC

# COMMAND ----------

