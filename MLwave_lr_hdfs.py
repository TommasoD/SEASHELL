from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from utils import estimate_df_size

#/opt/spark/bin/spark-submit --master spark://spark-master:7077 \
#/opt/spark-apps/main.py

def init_spark():
  sql = SparkSession.builder\
    .appName("hdfs_ECG-classifiers-MultinomialLR")\
    .getOrCreate()
  sc = sql.sparkContext
  return sql,sc

Sizes=[]
sql,sc = init_spark()

print(sql)
print(sc)

# Load and parse the data file, converting it to a DataFrame.
data = sql.read.load("hdfs://namenode:9000/user/root/input/Applications/Data/featureData.txt",format = "csv", inferSchema="true", sep=",", header="true")

df_size = estimate_df_size(data,sc)
Sizes.append(df_size)

labelIndexer = StringIndexer(inputCol="Labels", outputCol="indexedLabel").fit(data)


#forse questa è opzionale per la natura stessa del file featureData che presenta le features già "raggruppati" a vettore
cols = data.columns
cols.remove("Labels")
assembler = VectorAssembler(inputCols=cols,outputCol="features")

data = assembler.transform(data)

df_size = estimate_df_size(data,sc)
Sizes.append(df_size)


featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=100).fit(data)

(trainingData, testData) = data.randomSplit([0.7, 0.3])
dt = LogisticRegression(maxIter=5, regParam=0.03, 
                        elasticNetParam=0.5, featuresCol="indexedFeatures", labelCol="indexedLabel")
    # Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

    # Make predictions.
predictions = model.transform(testData)

df_size = estimate_df_size(predictions,sc)
Sizes.append(df_size)

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")


print('F1-Score ', evaluator.evaluate(
                          predictions, 
                          {evaluator.metricName: 'f1'})
)
print('Precision ', evaluator.evaluate(
                          predictions,
                          {evaluator.metricName: 'weightedPrecision'})
)
print('Recall ', evaluator.evaluate(
                          predictions, 
                          {evaluator.metricName: 'weightedRecall'})
)
print('Accuracy ', evaluator.evaluate(
                          predictions, 
                          {evaluator.metricName: 'accuracy'})
)