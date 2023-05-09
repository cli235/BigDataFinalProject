import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

hc = (SparkSession.builder
                  .appName('Toxic Comment Classification')
                  .enableHiveSupport()
                  .config("spark.executor.memory", "2G") # max memory for executor. Should be smaller than available memory on each worker
                  .config("spark.driver.memory","10G") # max memory for driver. 
                  .config("spark.executor.cores","2") # Number of cores allocated to each executor. Should be smaller than CPU cores in cluster
                  .config("spark.python.worker.memory","4G") # max memory for each worker
                  .config("spark.driver.maxResultSize","0")
                  .config("spark.sql.crossJoin.enabled", "true")
                  .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                  .config("spark.default.parallelism","4")
                  .getOrCreate())

def to_spark_df(fin):
    try:
        df = pd.read_csv(fin, encoding='latin1', error_bad_lines=False)
    except pd.errors.ParserError as e:
        print(f"Skipping problematic rows in file {fin}...")
        df = pd.read_csv(fin, encoding='latin1', error_bad_lines=False, quoting=csv.QUOTE_NONE)

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.astype(str)
    sdf = hc.createDataFrame(df)
    sdf = sdf.select([F.col(col).alias(col.strip().replace(" ", "_")) for col in sdf.columns])
    return sdf

# Load the train-test sets
train = to_spark_df("train.csv")
test = to_spark_df("test.csv")

#Check the dataframe records
train.show()
test.show()

out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]
train.show(5)
# View some toxic comments
train.filter(F.col('toxic') == 1).show(5)

# Basic sentence tokenizer
tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
wordsData = tokenizer.transform(train)
# Count the words in a document
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
tf = hashingTF.transform(wordsData)
tf.select('rawFeatures').take(2)

# Build the idf model and transform the original token frequencies into their tf-idf counterparts
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tf) 
tfidf = idfModel.transform(tf)
tfidf.select("features").first()

REG = 0.1
lr = LogisticRegression(featuresCol="features", labelCol='toxic', regParam=REG)
tfidf.show(5)

tfidf = tfidf.withColumn("toxic", col("toxic").cast("double"))
lrModel = lr.fit(tfidf.limit(3000))
res_train = lrModel.transform(tfidf)
res_train.select("id", "toxic", "probability", "prediction").show(20)
res_train.show(5)

extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
(res_train.withColumn("proba", extract_prob("probability"))
 .select("proba", "prediction").show())

test_tokens = tokenizer.transform(test)
test_tf = hashingTF.transform(test_tokens)
test_tfidf = idfModel.transform(test_tf)

test_res = test.select('id')
test_res.head()

tfidf = tfidf.withColumn("severe_toxic", col("severe_toxic").cast("double"))
tfidf = tfidf.withColumn("obscene", col("obscene").cast("double"))
tfidf = tfidf.withColumn("threat", col("threat").cast("double"))
tfidf = tfidf.withColumn("insult", col("insult").cast("double"))
tfidf = tfidf.withColumn("identity_hate", col("identity_hate").cast("double"))

test_probs = []
for col in out_cols:
    print(col)
    lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=REG)
    print("...fitting")
    lrModel = lr.fit(tfidf)
    print("...predicting")
    res = lrModel.transform(test_tfidf)
    print("...appending result")
    test_res = test_res.join(res.select('id', 'probability'), on="id")
    print("...extracting probability")
    test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")
    test_res.show(5)

    test_res.coalesce(1).write.csv('/sample_data/spark_lr.csv', mode='overwrite', header=True)
    !cat /sample_data/spark_lr.csv/part*.csv > spark_lr.csv