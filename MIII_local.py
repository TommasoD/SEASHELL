from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from utils import get_statistics
#TODO: ADD DATABASE SAVE SUCH AS MAIN.PY

def init_spark():
  sql = SparkSession.builder\
    .appName("local_SQL_Analysis_1")\
    .getOrCreate()
  sc = sql.sparkContext
  return sql,sc


sql,sc = init_spark()

files = ["/opt/spark-data/mimic/ADMISSIONS.csv","/opt/spark-data/mimic/PATIENTS.csv","/opt/spark-data/mimic/DIAGNOSES_ICD.csv", "/opt/spark-data/mimic/D_ICD_PROCEDURES.csv"]

#PREPARE THE DATA BY READING IT AND ESTIMATE THE SIZE OF DATAFRAMES

df1 = sql.read.load(files[0],format = "csv", inferSchema="true", sep=",", header="true")

df2 = sql.read.load(files[1],format = "csv", inferSchema="true", sep=",", header="true")

df_join = df1.join(df2,df1.SUBJECT_ID ==  df2.SUBJECT_ID,"inner").select(df1.SUBJECT_ID,col("GENDER"),col("ADMISSION_TYPE"),col("DIAGNOSIS"))


#PRINT DEL NUMERO DI AMMISSIONI ALL'OSPEDALE DI UOMINI-DONNE-NEUTRAL IN EMERGENCY
rownum_m= df_join.filter((df_join.GENDER == 'M') & (df_join.ADMISSION_TYPE == 'EMERGENCY')).count()
rownum_f= df_join.filter((df_join.GENDER == 'F') & (df_join.ADMISSION_TYPE == 'EMERGENCY')).count()
rownum_n = df_join.filter((df_join.GENDER != 'F') & (df_join.GENDER != 'M') & (df_join.ADMISSION_TYPE == 'EMERGENCY')).count()
print(f"Number of male admission: {rownum_m}\nNumber of female admission{rownum_f}\nNumbero fo admission neutral:{rownum_n}")

#ANALISI STATISTICA SULLE ADMISSION IN BASE AL GENDER E ALLA DIAGNOSI DI SEPSI - ADMISSIONS JOIN PATIENTS
rownum_m= df_join.filter((df_join.GENDER == 'M') & (df_join.DIAGNOSIS == 'SEPSIS')).count()
rownum_f= df_join.filter((df_join.GENDER == 'F') & (df_join.DIAGNOSIS == 'SEPSIS')).count()
rownum_n = df_join.filter((df_join.GENDER != 'F') & (df_join.GENDER != 'M') & (df_join.DIAGNOSIS == 'SEPSIS')).count()
print(f"Number of male sepsis: {rownum_m}\nNumber of female sepsis:{rownum_f}\nNumbero fo admission neutral sepsis:{rownum_n}")



#PREPARE THE DATA BY READING IT AND ESTIMATE THE SIZE OF DATAFRAMES

df3 = sql.read.load(files[2],format = "csv", inferSchema="true", sep=",", header="true")

df4 = sql.read.load(files[3],format = "csv", inferSchema="true", sep=",", header="true")

df_join_2 = df3.join(df4,df3.ICD9_CODE ==  df4.ICD9_CODE,"inner").select(df3.ICD9_CODE,col("SHORT_TITLE"),col("LONG_TITLE"),col("SUBJECT_ID"))



#PRINT THE TOTAL NUM OF DIAGNOSES CONCERNING THYROID
rownum = df_join_2.filter(df_join_2.SHORT_TITLE.contains('thyroid') | df_join_2.LONG_TITLE.contains('thyroid')).count()
print(f"The total number of diagnoses concerning thyroid is:{rownum}")

dfs = [df1,df2,df3,df4,df_join_2,df_join]

#list1,list2 = get_statistics(dfs,files,sql,sc)


print(get_statistics(dfs,files,sql,sc))


