# Databricks notebook source
import dlt
import pyspark.sql.functions as F
from pyspark.sql.types import TimestampType
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType, BooleanType, DoubleType, TimestampType

# COMMAND ----------

schema = StructType([
    StructField("err",StringType(),True),
    StructField("fluentd_size_bytes",LongType(),True),
    StructField("kubernetes_container_image",StringType(),True),
    StructField("kubernetes_labels_app",StringType(),True),
    StructField("kubernetes_labels_chart",StringType(),True),
    StructField("kubernetes_labels_component",StringType(),True),
    StructField("kubernetes_labels_pod-template-hash",StringType(),True),
    StructField("kubernetes_labels_release",StringType(),True),
    StructField("kubernetes_labels_system",StringType(),True),
    StructField("kubernetes_namespace_name",StringType(),True),
    StructField("kubernetes_pod_ip",StringType(),True),
    StructField("kubernetes_pod_name",StringType(),True),
    StructField("level",LongType(),True),
    StructField("logger",StringType(),True),
    StructField("msg",StringType(),True),
    StructField("responseTime",StringType(),True),
    StructField("spid",StringType(),True),
    StructField("statusCode",StringType(),True),
    StructField("stream",StringType(),True),
    StructField("subject",StringType(),True),
    StructField("tag",StringType(),True),
    StructField("time",TimestampType(),True),
    StructField("traceToken",StringType(),True),
])

# COMMAND ----------

@dlt.table(
    name="k8_nxs_dev_smartchecklist_logs",
    comment="Bronze table for Kubernetes logs",
    table_properties={"quality": "bronze"},
)
def raw_events_data():
    return (
        spark.readStream 
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .schema(schema)
        .load("s3://sm.logs.dev/kubernetes/nxs-dev.smartchecklist.nxs-dev-smartchecklist-core-api/2024/*/*/*/*")
        .withColumn("recordFilePath", F.col("_metadata.file_path"))
        .withColumn("processedDate", F.current_timestamp())
    )