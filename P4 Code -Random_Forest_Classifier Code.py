# Import necessary libraries
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

# create a spark session
spark = SparkSession.builder.appName("Random Forest Classifier").getOrCreate()

# Load the data into a DataFrame
train_data_load = spark.read.format("csv").option("header", "false").load("adult.data")
test_data_load= spark.read.format("csv").option("header", "false").load("adult.test")

# Add column names to the DataFrame
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]
train_data_df = train_data_load.toDF(*columns)
test_data_df = test_data_load.toDF(*columns)

# Remove leading and trailing whitespaces from the categorical columns
categorical_columns = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "sex", "native_country", "income"]
for column in categorical_columns:
    train_data = train_data_df.withColumn(column, trim(col(column)))
    test_data = test_data_df.withColumn(column, trim(col(column)))

    # Add a label column to the data using StringIndexer
indexer = StringIndexer(inputCol="income", outputCol="label")
train_data = indexer.fit(train_data).transform(train_data)
test_data = indexer.fit(test_data).transform(test_data)

# Vectorize the features
categorical_features = ["workclass", "education", "marital_status", "occupation",
                        "relationship", "race", "sex", "native_country"]
numeric_features = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assembler_inputs = [c + "_encoded" for c in categorical_features] + numeric_features
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Convert categorical columns to indices and one-hot encode them
indexers = [StringIndexer(inputCol=c, outputCol=c + "_index") for c in categorical_features]
encoders = [OneHotEncoder(inputCol=c + "_index", outputCol=c + "_encoded") for c in categorical_features]

# Build the Random Forest model
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# Chain indexers, encoders, assembler and the model in a Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
accuracy = evaluator.evaluate(predictions)

print("Model accuracy: ", accuracy)