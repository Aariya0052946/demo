# Databricks notebook source
display(spark.sql("select * from akash.train.train_1"))

# COMMAND ----------

import json

# Table name in Unity Catalog
table_name = "akash.train.train_1"  # Change if needed

# Load table
df = spark.table(table_name)

# Get schema in JSON
schema_json = df.schema.json()

# Print compact JSON
print(schema_json)

# Pretty-print JSON
schema_dict = json.loads(schema_json)
print(json.dumps(schema_dict, indent=4))


# COMMAND ----------

import json
from datetime import date, datetime
from decimal import Decimal

# Helper to make JSON serializable
def default_serializer(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)

# Your table name
table_name = "akash.train.train_1"

# Read table
df = spark.table(table_name)

# Collect data as list of dicts
json_data = [row.asDict() for row in df.collect()]

# Print pretty JSON
print(json.dumps(json_data, indent=4, default=default_serializer))

# Save to file
output_path = "/Workspace/Users/akashk@datapattern.ai/AIoutput.json"
with open(output_path, "w") as f:
    json.dump(json_data, f, indent=4, default=default_serializer)

print(f"✅ JSON written to: {output_path}")

# Stop Spark session
spark.stop()


# COMMAND ----------

import json
from textwrap import dedent
import re
from datetime import date, datetime
from decimal import Decimal


# ----------------- Step 1: Load table -----------------
table_name = "akash.train.train_1"

# Read table from Unity Catalog
df = spark.table(table_name)

# Helper to make JSON serializable
def default_serializer(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)

# Collect sample data as list of dicts
json_data = [row.asDict() for row in df.collect()]

# ----------------- Step 2: Prepare input for AI -----------------
input_for_ai = {
    "table_name": table_name,
    "columns": [{"name": field.name, "type": str(field.dataType)} for field in df.schema.fields],
    "sample_data": json_data
}

# Print JSON
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

def robust_json_loads(text: str):
    """
    Tries to parse JSON. If model adds stray text, extracts first JSON block.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

print("\n✅ Spark session started and prompt ready for AI.")


# COMMAND ----------

