import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
from pyspark.sql.functions import col, sum, when
from pyspark.sql.functions import count, when, isnull
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

%matplotlib inline

from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName('heart-disease-analysis').getOrCreate()

# Load the data into a PySpark DataFrame
heart_disease_df = spark.read.format('csv').option('header', 'true').load('framingham.csv')

# Drop the 'education' column
heart_df_drop = heart_disease_df.drop('education')

# Rename the 'male' column to 'Sex_male'
heart_df_rename = heart_df_drop.withColumnRenamed('male', 'Sex_male')

# Count the number of null values in each column
null_counts = heart_df_rename.select([sum(when(col(c) == 'NA', 1).otherwise(0)).alias(c) for c in heart_df_rename.columns])

# Display the null counts for each column
null_counts.show()

# Replace the string 'NA' with the actual null value
heart_df_replace = heart_df_rename.na.replace('NA', None)

# Drop rows containing null values
heart_df = heart_df_replace.dropna(how='any', thresh=None, subset=None)

heart_df.show()

# Count the number of rows for each category in the 'TenYearCHD' column
heart_df.groupby('TenYearCHD').count().show()

# Visualize the count of each category in the 'TenYearCHD' column
sn.countplot(x='TenYearCHD',data=heart_df.toPandas())

# Show summary statistics of the data
heart_df.describe().show()

# Convert PySpark DataFrame to pandas DataFrame
heart_df_pd = heart_df.toPandas()

# Visualize the pairwise relationships between features
sn.pairplot(data=heart_df_pd, x_vars=['age', 'cigsPerDay'], y_vars=['totChol', 'sysBP'])

import pyspark.sql.functions as F
import matplotlib.pyplot as plt

def draw_histograms(dataframe, features, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(20,20))
    axes = axes.ravel()

    for i, feature in enumerate(features):
        # Convert the column to a numerical type
        dataframe = dataframe.withColumn(feature, dataframe[feature].cast("double"))
        
        # Compute the histogram using PySpark functions
        bins, counts = dataframe.select(feature).rdd.flatMap(lambda x: x).histogram(20)
        
        # Plot the histogram using Matplotlib
        axes[i].bar(bins[:-1], counts, width=(bins[1]-bins[0]), color='midnightblue')
        axes[i].set_title(feature + " Distribution", color='DarkRed')

    fig.tight_layout()
    plt.show()

draw_histograms(heart_df, heart_df.columns, 6, 3)
# Create a vector assembler to assemble the features into a single vector column
assembler = VectorAssembler(inputCols=['Sex_male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'], outputCol='features')

# Use the assembler to transform the DataFrame and select the features and target columns
heart_df_assembled = assembler.transform(heart_df).select('features', 'TenYearCHD')

# Split the data into training and test sets
train_df, test_df = heart_df_assembled.randomSplit([0.8, 0.2], seed=42)

# Create a logistic regression model and fit it to the training data
lr = LogisticRegression(featuresCol='features', labelCol='TenYearCHD')
lr_model = lr.fit(train_df)

# Make predictions on the test data and evaluate the model's performance
predictions = lr_model.transform(test_df)
predictions.select('TenYearCHD', 'prediction', 'probability').show()

# Print the model's summary
print(lr_model.summary)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Make predictions on the test data and evaluate the model's performance
predictions = lr_model.transform(test_df)

# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='TenYearCHD', rawPredictionCol='rawPrediction')
accuracy = evaluator.evaluate(predictions)

# Print the accuracy
print("Accuracy:", accuracy)