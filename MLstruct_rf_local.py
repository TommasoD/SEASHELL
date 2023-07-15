from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext
from pyspark.sql.functions import mean, col, split, regexp_extract, when, lit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import roc_curve, auc
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from utils import estimate_df_size


#N.B. convertire i dataframe in panda Dataframe se vuoi printare "statistiche sui dati" mediante matplotlib e Seaborn
#https://www.kaggle.com/code/kallefischer/pyspark-stroke-prediction#Connect-to-Spark





def init_spark():
  sql = SparkSession.builder\
    .appName("local_Strokepred-app_RandomForest")\
    .getOrCreate()
  sc = sql.sparkContext
  return sql,sc

Sizes = []

sql,sc = init_spark()

file = "/opt/spark-data/healthcare-dataset-stroke-data.csv"

data = sql.read.load(file,format = "csv", inferSchema="true", sep=",", header="true")

size_df = estimate_df_size(data,sc)
Sizes.append(size_df)


data = data.withColumn("age", data["age"].cast("integer"))
data = data.withColumn("bmi", data["bmi"].cast("integer"))
data = data.withColumn("avg_glucose_level", data["avg_glucose_level"].cast("integer"))
data = data.withColumn("age", data["age"].cast("integer"))
data = data.withColumn("avg_glucose_level", data["avg_glucose_level"].cast("integer"))
data = data.withColumn("bmi", data["bmi"].cast("integer"))

data = data.na.fill({"bmi" : 28})
data = data.drop("id")

data = data.withColumn('Blood&Heart', data.hypertension * data.heart_disease)
data = data.withColumn('Obesity', data["bmi"] * data["avg_glucose_level"]/1000)


data=data.withColumn('age',
                    when(data['age'] < 30, "Adult").
                    when((data['age'] > 30) & (data['age'] < 40), "MiddleAge").
                    otherwise("OldAge"))



indexer = StringIndexer(inputCol="age", outputCol="age_label")
indexer.fit(data).transform(data)
temp_sdf = indexer.fit(data).transform(data)
data = temp_sdf.withColumn("age", temp_sdf["age_label"].cast("integer"))

indexer = StringIndexer(inputCol="gender", outputCol="gender_label")
indexer.fit(data).transform(data)
temp_sdf = indexer.fit(data).transform(data)
data = temp_sdf.withColumn("gender", temp_sdf["gender_label"].cast("integer"))

indexer = StringIndexer(inputCol="ever_married", outputCol="ever_married_label")
indexer.fit(data).transform(data)#.show(5)
temp_sdf = indexer.fit(data).transform(data)
data = temp_sdf.withColumn("ever_married", temp_sdf["ever_married_label"].cast("integer"))

indexer = StringIndexer(inputCol="work_type", outputCol="work_type_label")
indexer.fit(data).transform(data)#.show(5)
temp_sdf = indexer.fit(data).transform(data)
data = temp_sdf.withColumn("work_type", temp_sdf["work_type_label"].cast("integer"))

indexer = StringIndexer(inputCol="smoking_status", outputCol="smoking_status_label")
indexer.fit(data).transform(data)#.show(5)
temp_sdf = indexer.fit(data).transform(data)
data = temp_sdf.withColumn("smoking_status", temp_sdf["smoking_status_label"].cast("integer"))

indexer = StringIndexer(inputCol="Residence_type", outputCol="Residence_type_label")
indexer.fit(data).transform(data)#.show(5)
temp_sdf = indexer.fit(data).transform(data)
data = temp_sdf.withColumn("Residence_type", temp_sdf["Residence_type_label"].cast("integer"))

data = data.withColumn("age_label", data["age_label"].cast("integer"))
data = data.withColumn("gender_label", data["gender_label"].cast("integer"))
data = data.withColumn("ever_married_label", data["ever_married_label"].cast("integer"))
data = data.withColumn("work_type_label", data["work_type_label"].cast("integer"))
data = data.withColumn("smoking_status_label", data["smoking_status_label"].cast("integer"))
data = data.withColumn("Residence_type_label", data["Residence_type_label"].cast("integer"))

data = data.drop("age","gender","ever_married","work_type","smoking_status","Residence_type")
data=data[["gender_label","age_label","hypertension","heart_disease","ever_married_label","work_type_label","Residence_type_label","avg_glucose_level","bmi","smoking_status_label","stroke"]]

feature = VectorAssembler(inputCols = data.drop('stroke').columns, outputCol='features')
feature_vector = feature.transform(data)

size_df = estimate_df_size(feature_vector,sc)
Sizes.append(size_df)

feat_lab_df = feature_vector.select(['features', 'stroke'])
train, test = feat_lab_df.randomSplit([0.8, 0.2])

rf = RandomForestClassifier(labelCol='stroke')

paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10, 20])\
                              .addGrid(rf.maxBins, [20, 32, 50])\
                              .addGrid(rf.numTrees, [20, 40, 60])\
                              .addGrid(rf.impurity, ['gini', 'entropy'])\
                              .addGrid(rf.minInstancesPerNode, [1, 5, 10])\
                              .build()
    
tvs = TrainValidationSplit(estimator=rf,
                           estimatorParamMaps=paramGrid,
                           evaluator=MulticlassClassificationEvaluator(labelCol='stroke'),
                           trainRatio=0.8)

rf_model = tvs.fit(train)
rf_model_pred = rf_model.transform(test)

results = rf_model_pred.select(['probability', 'stroke'])

results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)

metrics = metric(scoreAndLabels)
rf_acc = round(MulticlassClassificationEvaluator(labelCol='stroke', metricName='accuracy').evaluate(rf_model_pred), 4)
rf_prec = round(MulticlassClassificationEvaluator(labelCol='stroke', metricName='weightedPrecision').evaluate(rf_model_pred), 4)
rf_roc = round(metrics.areaUnderROC, 4)

rf_dict = {'Accuracy': rf_acc, 'Precision': rf_prec, 'ROC Score': rf_roc}
print(rf_dict)
print("DFs sizes",Sizes)



