from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType
import requests

spark = SparkSession.builder.appName("SmartMeter").getOrCreate()

schema = StructType() \
    .add("timestamp", StringType()) \
    .add("voltage", DoubleType()) \
    .add("current", DoubleType()) \
    .add("power", DoubleType()) \
    .add("label", StringType())

df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "smart-meter") \
    .load()

data = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")).select("data.*")

def send(row):
    try:
        requests.post("http://localhost:8000/data", json=row.asDict())
    except:
        pass

query = data.writeStream.foreach(send).start()
query.awaitTermination()
