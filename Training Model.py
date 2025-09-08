# Databricks notebook source
display(spark.sql("select*from akash.financial.Sample_data"))

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import json

# Define schema
schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("customer_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("card_number", StringType(), True),
    StructField("transaction_date", StringType(), True),  # or TimestampType()
    StructField("transaction_amount", DoubleType(), True),
    StructField("currency", StringType(), True),
    StructField("merchant_id", StringType(), True),
    StructField("store_location", StringType(), True),
    StructField("payment_method", StringType(), True)
])

# Convert schema to JSON
schema_json = schema.json()

# Print JSON schema
print(schema_json)

# If you want pretty JSON format
schema_dict = json.loads(schema_json)
print(json.dumps(schema_dict, indent=4))

# COMMAND ----------

from pyspark.sql import SparkSession
from datetime import datetime
import json

# Initialize Spark session
spark = SparkSession.builder.appName("TableToJSON").getOrCreate()

# Replace with your actual catalog, database, and table name
# Example: 'catalog_name.database_name.table_name'
table_name = "akash.financial.sample_data"

# Read table from Unity Catalog
df = spark.table(table_name)

# Convert timestamp columns to string
df_with_strings = df.withColumn("transaction_date", df["transaction_date"].cast(StringType()))

# Collect data as list of dictionaries
json_data = [row.asDict() for row in df_with_strings.collect()]

# Print the JSON data with pretty formatting
print(json.dumps(json_data, indent=4))

# Optionally, write to local tmp file on the driver node
output_path = "file:/Workspace/Users/akashk@datapattern.ai/output.json"
with open("/Workspace/Users/akashk@datapattern.ai/output.json", "w") as f:
    json.dump(json_data, f, indent=4)

# To view the file in notebook, you can use:
# %sh cat /tmp/output.json

# Stop Spark session
spark.stop()

# COMMAND ----------

import json
from textwrap import dedent
import re
from pyspark.sql.types import StringType

# ----------------- Step 1: Load table -----------------
# Databricks provides a default Spark session called `spark`
table_name = "akash.financial.Sample_data"

# Read table from Unity Catalog
df = spark.table(table_name)

# Convert timestamp columns to string if needed
df_with_strings = df.withColumn("transaction_date", df["transaction_date"].cast(StringType()))

# Collect sample data as a list of dictionaries
json_data = [row.asDict() for row in df_with_strings.collect()]

# ----------------- Step 2: Prepare input for AI -----------------
input_for_ai = {
    "table_name": table_name,
    "columns": [{"name": field.name, "type": str(field.dataType)} for field in df.schema.fields],
    "sample_data": json_data
}

# Check input
print(json.dumps(input_for_ai, indent=4))

# ----------------- Step 3: Build AI prompt -----------------
expected_schema_hint = {
    "table_description": "string",
    "columns": {
        "<column_name>": {
            "description": "string",
            "classification": "PII | financial | sensitive | general"
        }
    }
}

prompt = dedent(f"""
You are a data catalog assistant.

Given the following table schema and sample rows, produce STRICT JSON matching this shape:
{json.dumps(expected_schema_hint, indent=2)}

Rules:
- Only output valid JSON (no prose, no markdown).
- Classify each column as one of: PII, financial, sensitive, general.
- Be precise and concise.

Schema & Sample Data:
{json.dumps(input_for_ai)}
""").strip()

# ----------------- Step 4: Utility to parse model response -----------------
def robust_json_loads(text: str):
    """
    Tries to parse JSON. If the model adds stray text, extracts the first JSON block.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Capture a JSON object using a simple bracket matcher
        m = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if m:
            candidate = m.group(0)
            return json.loads(candidate)
        raise

# ----------------- Ready to pass 'prompt' to AI -----------------
# print(prompt)


# COMMAND ----------

# --- Databricks Model Serving (Llama) ---
import requests
import os

# DATABRICKS_HOST   = dbutils.secrets.get(scope="my_scope", key="https://dbc-bc498a13-d140.cloud.databricks.com/editor/notebooks/2501555814062667?o=1149556702135849#command/7663071826872929")   # e.g. https://adb-xxxx.azuredatabricks.net
# DATABRICKS_TOKEN  = dbutils.secrets.get(scope="my_scope", key="")
DATABRICKS_HOST   = "https://dbc-bc498a13-d140.cloud.databricks.com"
DATABRICKS_TOKEN  = "dapi57c1244e6efde6f5bdbdf3a002f27ae9"
ENDPOINT_NAME     = "databricks-meta-llama-3-3-70b-instruct"  # your serving endpoint name

headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

payload = {
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1,
    "max_tokens": 1500
}

# Databricks OpenAI-compatible route (if enabled) OR use serving endpoints invoke URL:
# 1) If OpenAI-compatible is enabled on your endpoint:
# url = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"
# 2) If your endpoint expects MLflow serving invoke (common):
url = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"

resp = requests.post(url, headers=headers, json=payload, timeout=60)
resp.raise_for_status()

out = resp.json()

# Responses vary by configuration. Handle both common shapes:
if isinstance(out, dict) and "choices" in out:
    raw = out["choices"][0]["message"]["content"]
elif isinstance(out, dict) and "predictions" in out:
    # Some endpoints return {"predictions": [{"content": "..."}]}
    raw = out["predictions"][0].get("content", "")
else:
    # Fallback: try to stringify
    raw = json.dumps(out)

metadata_json = robust_json_loads(raw)
print(json.dumps(metadata_json, indent=2))


# COMMAND ----------

# Databricks PySpark Script
# Purpose: Push model-generated metadata (tags/classifications) into Unity Catalog

from pyspark.sql import SparkSession
import json

spark = SparkSession.builder.getOrCreate()

# ============================================
# 1. Your model-generated JSON metadata
# ============================================

model_output = {
  "table_description": "Financial transaction data",
  "columns": {
    "transaction_id": {
      "description": "Unique transaction identifier",
      "classification": "general"
    },
    "customer_id": {
      "description": "Unique customer identifier",
      "classification": "PII"
    },
    "customer_name": {
      "description": "Customer name",
      "classification": "PII"
    },
    "email": {
      "description": "Customer email address",
      "classification": "PII"
    },
    "card_number": {
      "description": "Credit/debit card number",
      "classification": "sensitive"
    },
    "transaction_date": {
      "description": "Date and time of transaction",
      "classification": "general"
    },
    "transaction_amount": {
      "description": "Amount of transaction",
      "classification": "financial"
    },
    "currency": {
      "description": "Currency of transaction",
      "classification": "general"
    },
    "merchant_id": {
      "description": "Unique merchant identifier",
      "classification": "general"
    },
    "store_location": {
      "description": "Location of store where transaction occurred",
      "classification": "general"
    },
    "payment_method": {
      "description": "Method used for payment",
      "classification": "general"
    }
  }
}

# ============================================
# 2. Define source & destination tables
# ============================================

source_table = "akash.financial.sample_data"
target_table = "akash.financial.sample_data_with_tags"

# ============================================
# 3. Load the source table
# ============================================

df = spark.table(source_table)

# For now, just copy data to new table; later you can enrich with model predictions if needed
df.write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable(target_table)

print(f"✅ Table '{target_table}' created in Unity Catalog.")

# ============================================
# 4. Apply Table-level Description
# ============================================

table_desc = model_output.get("table_description", "")
if table_desc:
    spark.sql(f"ALTER TABLE {target_table} SET TBLPROPERTIES ('description' = '{table_desc}')")
    print(f"✅ Table description added: {table_desc}")

# ============================================
# 5. Apply Column-level Tags & Comments
# ============================================

for col_name, col_info in model_output["columns"].items():
    description = col_info.get("description", "")
    classification = col_info.get("classification", "")

    # Add column comment
    if description:
        spark.sql(f"ALTER TABLE {target_table} ALTER COLUMN {col_name} COMMENT '{description}'")
    
    # Add tags dynamically
    if classification:
        spark.sql(f"""
            ALTER TABLE {target_table} ALTER COLUMN {col_name}
            SET TAGS ('Classification' = '{classification}')
        """)
    print(f"✅ Column '{col_name}' tagged as '{classification}' with comment '{description}'.")

print("🎉 All tags and descriptions applied successfully!")
