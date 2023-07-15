from pyspark.sql import SparkSession
from utils import estimate_df_size

def init_spark():
  sql = SparkSession.builder\
    .appName("hdfs_MIMIC-analysis_800MB")\
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
  sc = sql.sparkContext
  return sql,sc

def main():

  sql,sc = init_spark()
  file1 = "hdfs://namenode:9000/user/root/input/Applications/Data/PRESCRIPTIONS.csv"

  df1 = sql.read.load(file1,format = "csv", inferSchema="true", sep=",", header="true")

  df1.show()
  print("CHEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEECK ME",df1.count())

  #df_size = estimate_df_size(df1,sc)
  #print("HERES THE DFSIZEEEEEEEEEEEEEEEEEEEEEEEEEEE: ",df_size)

  df1 = df1.groupBy("DRUG_TYPE").count()
  df1.show()



  

  
if __name__ == '__main__':
  main()
