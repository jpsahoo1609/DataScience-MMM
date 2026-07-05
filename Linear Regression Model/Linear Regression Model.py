# Databricks notebook source
# MAGIC %md
# MAGIC # Problem Statement
# MAGIC A company wants to understand the impact of different advertising channels on sales and identify which marketing channel provides the highest return on investment (ROI).
# MAGIC
# MAGIC **The objective is to:**
# MAGIC
# MAGIC Analyze the relationship between advertising spend and sales.
# MAGIC Build a Linear Regression model using:
# MAGIC
# MAGIC TV Ad Budget
# MAGIC Radio Ad Budget
# MAGIC Newspaper Ad Budget
# MAGIC
# MAGIC
# MAGIC Determine which advertising channel contributes the most to sales.
# MAGIC Use the model coefficients as an estimate of the incremental ROI of each channel.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Data
# MAGIC

# COMMAND ----------

df = spark.read.format('csv').option('inferSchema', 'true').option('header','true').load('/Volumes/workspace/default/databricks_firstvolume/Advertising Budget and Sales.csv')
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Data

# COMMAND ----------

df.count()

# COMMAND ----------

df.columns

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col 
df = df.withColumnRenamed('_c0', 'ID')
df.display()

# COMMAND ----------

df.describe().display()

# COMMAND ----------

from pyspark.sql import functions as F
avg_sales = (df.filter(F.col("Sales ($)")>=2)).agg(F.avg("Sales ($)")).collect()[0][0]
# Replace 
df = df.withColumn("Sales",F.when(F.col("Sales ($)") < 2, F.lit(avg_sales)).otherwise(F.col("Sales ($)")))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Nulls 

# COMMAND ----------

from pyspark.sql.functions import col
for c in df.columns:
    null_count = df.filter(col(c).isNull()).count()
    print(c, null_count)

# COMMAND ----------

df = df.dropDuplicates()

# COMMAND ----------

from pyspark.sql.functions import sum,avg,max,min
df.agg(sum("Sales ($)"),max("Sales ($)"),min("Sales ($)")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Correlation

# COMMAND ----------

cor_TV_Sales = df.stat.corr("TV Ad Budget ($)","Sales ($)")
cor_Radio_Sales = df.stat.corr("Radio Ad Budget ($)","Sales ($)")
cor_Newspaper_Sales = df.stat.corr("Newspaper Ad Budget ($)","Sales ($)")

print("Correlation between TV Ad Budget and Sales: ",cor_TV_Sales)
print("Correlation between Radio Ad Budget and Sales: ",cor_Radio_Sales)
print("Correlation between Newspaper Ad Budget and Sales: ",cor_Newspaper_Sales)

# COMMAND ----------

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Split & model building 
# MAGIC
# MAGIC

# COMMAND ----------

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.display()
test_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Building 

# COMMAND ----------

df.display()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

vecAssembler = VectorAssembler(inputCols=["TV Ad Budget ($)","Radio Ad Budget ($)","Newspaper Ad Budget ($)"],outputCol="features")
df = vecAssembler.transform(df)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LinearRegression( featuresCol="features",labelCol="Sales ($)")
lr_model = lr.fit(train_df)
# Prediction
predictions = lr_model.transform(test_df)

predictions.select(
    "Sales ($)",
    "prediction"
).show()

# COMMAND ----------

predictions.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cofficients 
# MAGIC

# COMMAND ----------

print( "Intercept:", lr_model.intercept)
print("Coefficients:", lr_model.coefficients)

# COMMAND ----------

for feature,coef in zip(["TV", "Radio", "Newspaper"],lr_model.coefficients):
    print(f"{feature}: {coef}")

# COMMAND ----------

# MAGIC %md
# MAGIC Radio add contributes more to the sales. Every additional 1$ spent on radio advertising is associated with an increase of approx 0.18$ in sales.
# MAGIC Among all investment channels radio ad delivers the highest return 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Quality

# COMMAND ----------

print("R2:", lr_model.summary.r2)
print("RMSE:", lr_model.summary.rootMeanSquaredError)

# COMMAND ----------

# MAGIC %md
# MAGIC **The Linear Regression** analysis indicates that Radio advertising is the most effective channel for driving sales and delivers the highest estimated ROI. TV advertising contributes positively but has a much lower impact, while Newspaper advertising shows almost no measurable influence on sales.
# MAGIC Based on the model results, the company should prioritize additional investment in Radio advertising, maintain selective investment in TV advertising, and reconsider the effectiveness of Newspaper advertising within the marketing mix.

# COMMAND ----------

