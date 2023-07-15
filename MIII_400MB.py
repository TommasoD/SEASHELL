from pyspark.sql import SparkSession


def init_spark():
  sql = SparkSession.builder\
    .appName("hdfs_MIMIC-analysis_400MB")\
    .getOrCreate()
  sc = sql.sparkContext
  return sql,sc

def main():

  sql,sc = init_spark()
  file1 = "hdfs://namenode:9000/user/root/input/Applications/Data/OUTPUTEVENTS.csv"

  df1 = sql.read.load(file1,format = "csv", inferSchema="true", sep=",", header="true")

 
  rownum_m= df1.filter((df1.ITEMID == '40055') & (df1.VALUE >= 200)).count()

  print(f"Here's the rownum {rownum_m}")


  

  
if __name__ == '__main__':
  main()

