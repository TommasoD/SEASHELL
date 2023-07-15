from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import gc
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.types import StringType,BooleanType,DateType,DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def init_spark():
  sql = SparkSession.builder\
    .appName("local_ICU-stays-pred_LR")\
    .getOrCreate()
  sc = sql.sparkContext
  return sql,sc



# i dati utilizzati sono di physionet.org challenge 2012 sono però in un formato diverso dunque c'è una prima fase di reshaping per renderlo nel formato a noi più
#congeniale, nel seguente programma viene saltata questa fase di preprocessing utilizzando un dato già presente e preprocessato da kaggle
#se venisse richiesto quà trovi come fare il pre-processing e tutto https://www.kaggle.com/code/msafi04/predict-icu-mortality-shap/notebook oppure
# fai una prima fase in panda e poi converti un un dataframe Spark

file = "/opt/spark-data/icu_mortality_train.csv"

sql,sc = init_spark()

#lavoriamo con panda per il momento 
df = sql.read.load(file,format = "csv", inferSchema="true", sep=",", header="true").toPandas()
df = df.replace([-1.0, np.inf, -np.inf], np.nan)
df['age_group'] = pd.cut(df['Age'], bins = 9, labels = ['<20', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s'])

df['Height'] = df['Height'].fillna(df['Height'].median())
df['Weight'] = df['Weight'].fillna(df['Weight'].median())
df['bmi'] = df.apply(lambda x: round((x['Weight'] / (x['Height'] ** 2)) * 10000, 2), axis = 1)
df['bmi_group'] = pd.cut(df['bmi'], bins = [df['bmi'].min(), 18.5, 24.9, 29.9, df['bmi'].max()], labels = ['Underweight', 'Healthy', 'Overweight', 'Obesity'])

del df['bmi']
gc.collect()

cat_features = ['Gender', 'ICUType', 'age_group', 'bmi_group']
num_features = [c for c in df.columns if c not in cat_features]
num_features = [c for c in num_features if c not in ['RecordID', 'In-hospital_death']]
cat_features, num_features

#lavoriamo con dataframe panda per poi passare ad un dataframe spark 
df_0 = df[df['In-hospital_death'] == 0].copy()
df_1 = df[df['In-hospital_death'] == 1].copy()
#Impute Numerical Features with mean value
df_0[num_features] = df_0[num_features].fillna(df_0[num_features].mean())
df_1[num_features] = df_1[num_features].fillna(df_1[num_features].mean())

#Impute Categorical Features with most frequent value
for col in cat_features:
    df_0[col] = df_0[col].fillna(df_0[col].value_counts().index[0])
    df_1[col] = df_1[col].fillna(df_1[col].value_counts().index[0])

#concat both df, shuffle and reset index
df = pd.concat([df_0, df_1], axis = 0).sample(frac = 1).reset_index(drop = True)

#print(cat_features, num_features)

data = sql.createDataFrame(df)

#anzichè fare il label casta ad integer i valori in string e poi abilità la pipeline di sotto 

indexer = StringIndexer(inputCol="age_group", outputCol="age_label")
indexer.fit(data).transform(data)
temp_sdf = indexer.fit(data).transform(data)
data = temp_sdf.withColumn("age_group", temp_sdf["age_label"].cast("integer"))


indexer = StringIndexer(inputCol="bmi_group", outputCol="bmi_label")
indexer.fit(data).transform(data)
temp_sdf = indexer.fit(data).transform(data)
data = temp_sdf.withColumn("bmi_group", temp_sdf["bmi_label"].cast("integer"))


feature = VectorAssembler(inputCols = data.drop('In-hospital_death').columns, outputCol='features')
feature_vector = feature.transform(data)

feat_lab_df = feature_vector.select(['features', 'In-hospital_death'])
train, test = feat_lab_df.randomSplit([0.8, 0.2])

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric



lr = LogisticRegression(labelCol='In-hospital_death')

paramGrid = ParamGridBuilder().addGrid(lr. regParam, (0.01, 0.1))\
                              .addGrid(lr.maxIter, (5, 10))\
                              .addGrid(lr.tol, (1e-4, 1e-5))\
                              .addGrid(lr.elasticNetParam, (0.25, 0.75))\
                              .build()

tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=MulticlassClassificationEvaluator(labelCol='In-hospital_death'),
                           trainRatio=0.8)

lr_model = tvs.fit(train)
lr_model_pred = lr_model.transform(test)
results = lr_model_pred.select(['probability', 'In-hospital_death'])

results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)

metrics = metric(scoreAndLabels)
lr_acc = round(MulticlassClassificationEvaluator(labelCol='In-hospital_death', metricName='accuracy').evaluate(lr_model_pred), 4)
lr_prec = round(MulticlassClassificationEvaluator(labelCol='In-hospital_death', metricName='weightedPrecision').evaluate(lr_model_pred), 4)
lr_roc = round(metrics.areaUnderROC, 4)

lr_dict = {'Accuracy': lr_acc, 'Precision': lr_prec, 'ROC Score': lr_roc}
print(lr_dict)
















#cast to double in-hospital-deat
#labelIndexer = StringIndexer(inputCol="In-hospital_death", outputCol="indexedLabel").fit(data)
#
#cols = data.columns
#cols.remove("In-hospital_death")
#assembler = VectorAssembler(inputCols=cols,outputCol="features")
#
#data = assembler.transform(data)
#featureIndexer =\
#        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=100).fit(data)
#
#(trainingData, testData) = data.randomSplit([0.7, 0.3])
#
#dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
#
#    # Chain indexers and tree in a Pipeline
#pipeline = Pipeline(stages=[labelIndexer, assembler, featureIndexer, dt])
#
#    # Train model.  This also runs the indexers.
#model = pipeline.fit(trainingData)
#
#    # Make predictions.
#predictions = model.transform(testData)
#
#    # Select example rows to display.
##predictions.select("prediction", "indexedLabel", "features").show(100)
#predictions.printSchema()
#predictions.filter(predictions.prediction != predictions.indexedLabel).select(col("prediction"),col("indexedLabel"),col("probability")).show(100)
#
#evaluator = MulticlassClassificationEvaluator(
#    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
#accuracy = evaluator.evaluate(predictions)
#print("Accuracy = %g " % accuracy)
