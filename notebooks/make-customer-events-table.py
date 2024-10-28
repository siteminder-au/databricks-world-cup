# Databricks notebook source
import re
from pyspark.sql.functions import col, regexp_extract, lit
from pyspark.sql.types import StringType
from pyspark.sql.functions import get_json_object

# Dreamliner - provisioned nxs property product
table = spark.table("dev_hackathon.graylog.k8_nxs_dev_dreamliner_logs")
table = table.filter(col("msg").contains("provisioned nxs property product"))
table = table.withColumn(
    "spid", regexp_extract(col("msg"), r"spid=([0-9a-f-]*)", 1)
)
table = table.withColumn(
    "product", regexp_extract(col("msg"), r"product=([a-z-]*)", 1)
)
table = table.withColumn("msg", lit("Provisioned nxs property product"))
table = table.select("time", "msg", "spid", "product", "traceToken")
print("Number of rows in dreamliner table", table.count())
dreamliner = table

# Midas - created property billing plan
table = spark.table("dev_hackathon.graylog.k8_nxs_dev_midas_logs")
table = table.filter(col("msg").contains("created property billing plan"))
table = table.withColumn(
    "billingPlan",
    regexp_extract(col("msg"), r"created property billing plan\s(\w+)", 1),
)
table = table.withColumn("product", lit(None).cast(StringType()))
table = table.withColumn("msg", lit("Created property billing plan"))
table = table.select("time", "msg", "spid", "product", "billingPlan")
print("Number of rows in midas table", table.count())
midas = table

# Smartchecklist - task complete
table = spark.table("dev_hackathon.graylog.k8_nxs_dev_smartchecklist_logs")
table = table.filter(col("msg").contains("Updated property"))
table = table.withColumn(
    "taskCode",
    regexp_extract(col("msg"), r"Updated property.+for task\s(.*)", 1),
)
table = table.withColumn("msg", lit("Completed task"))
table = table.select("time", "msg", "spid", "taskCode")
print("Number of rows in smartchecklist table", table.count())
smartchecklist = table

# EI - new business event (need to use raw table since some fields are missing)
table = spark.table("dev_hackathon.graylog.k8_nxs_dev_ei_logs_raw")
table = table.filter(col("value").contains("Received request to get new business"))
table = table.withColumn("time", get_json_object(col("value"), "$.time"))
table = table.withColumn("msg", get_json_object(col("value"), "$.msg"))
table = table.withColumn("accountId", get_json_object(col("value"), "$.accountId"))
table = table.withColumn("opportunityId", get_json_object(col("value"), "$.opportunityId"))
table = table.withColumn("traceToken", get_json_object(col("value"), "$.traceToken"))
table = table.select("time", "msg","accountId", "opportunityId","traceToken")
table = dreamliner.join(table, "traceToken").select(table.time, table.msg, dreamliner.spid, table.accountId, table.opportunityId).orderBy("time")
table = table.dropDuplicates(["time"])
print("Number of rows in EI table", table.count())
table.show(8, False)
ei = table

# Append dreamliner and midas together
silver = dreamliner.unionByName(midas, allowMissingColumns=True).unionByName(smartchecklist, allowMissingColumns=True).unionByName(ei, allowMissingColumns=True).select("time", "msg", "spid", "product", "billingPlan","accountId", "opportunityId","taskCode").orderBy("time")
print("Number of rows in silver table", silver.count())
silver.show(20, False)

# Create or replace the silver table
silver.write.mode("overwrite").saveAsTable("dev_hackathon.graylog.customer_lifecycle_events_gold_table")