# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Extraction

# COMMAND ----------

# Import the requests library to interact with HTTP APIs
# This will be used to fetch weather data from the Open-Meteo API
import requests

# COMMAND ----------

# Define the Open-Meteo API endpoint URL
url = "https://api.open-meteo.com/v1/forecast"

# Configure API request parameters:
parametros = {
    "latitude": -38.37,        # Latitude coordinate for the location (likely Argentina)
    "longitude": -60.28,       # Longitude coordinate for the location
    "hourly": "temperature_2m", # Request hourly temperature data at 2 meters above ground
    "past_days": 30,           # Retrieve data from the past 30 days
}

# COMMAND ----------

# Make HTTP GET request to the API with our defined parameters
response = requests.get(url, params=parametros)

# Check if the request was successful (HTTP status code 200 = OK)
if response.status_code == 200:
    # Parse the JSON response into a Python dictionary
    data = response.json()
    print("Conexion existosa")  # Connection successful
else:
    # Print error message if request failed
    print(f"error: {response.status_code}")

# COMMAND ----------

# Inspect the structure of the API response
# First, show the top-level keys in the response data
print(data.keys())
# Then, show the keys within the 'hourly' data section
print(data["hourly"].keys())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Proccesing

# COMMAND ----------

# Create the Bronze layer DataFrame (raw data ingestion)
# Combine timestamp and temperature lists into tuples (rows)
filas = list(zip(data["hourly"]["time"], data["hourly"]["temperature_2m"]))

# Define column names for our DataFrame
columnas = ["fecha_hora_texto", "temperatura"]  # datetime_text, temperature

# Create a Spark DataFrame from the raw API data
df_bronze = spark.createDataFrame(filas, schema=columnas)

# Display the bronze DataFrame to verify data ingestion
display(df_bronze)

# COMMAND ----------

# Transform Bronze data to Silver layer (cleaned & standardized)
from pyspark.sql import functions as F

# Convert text datetime to Spark timestamp and drop the text column
df_silver = df_bronze.withColumn(
    "fecha_hora",
    F.to_timestamp("fecha_hora_texto", "yyyy-MM-dd'T'HH:mm")
).drop("fecha_hora_texto")

# Display the Silver DataFrame to verify timestamp conversion
display(df_silver)

# COMMAND ----------

# Save Silver layer DataFrame as a Delta table for downstream processing
df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("clima_silver")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick Sql for table creation

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create the Gold layer table using SQL for feature engineering
# MAGIC -- Adds hour, day of week, and lagged temperature as model signals
# MAGIC CREATE OR REPLACE TABLE clima_gold AS
# MAGIC SELECT 
# MAGIC     fecha_hora,
# MAGIC     hour(fecha_hora) as hora,             -- Extract hour of day
# MAGIC     dayofweek(fecha_hora) as dia_semana,  -- Extract day of the week
# MAGIC     temperatura as objetivo_temp,         -- Target value: temperature
# MAGIC     LAG(temperatura, 1) OVER (ORDER BY fecha_hora) as temp_h_anterior -- Previous hour temp
# MAGIC FROM clima_silver
# MAGIC WHERE temperatura IS NOT NULL

# COMMAND ----------

# MAGIC %md
# MAGIC ## Machine Learning

# COMMAND ----------

# Prepare data for machine learning model
# Load Gold table, drop rows with nulls for reliable training
df_ml_input = spark.table("clima_gold")
df_ml_final = df_ml_input.dropna()
print(f"Registros disponibles: {df_ml_final.count()}")  # Report available records

# COMMAND ----------

# Assemble features for ML model: hour, weekday, previous hour temp
from pyspark.ml.feature import VectorAssembler
pistas = ["hora", "dia_semana", "temp_h_anterior"]  # Model input columns
assembler = VectorAssembler(inputCols=pistas, outputCol="features")

df_entrenamiento = assembler.transform(df_ml_final)
# Display the features and target for visualization
display(df_entrenamiento.select("features", "objetivo_temp"))

# COMMAND ----------

# Train and evaluate a linear regression model to predict temperature
from pyspark.ml.regression import LinearRegression

# Split dataset into training and test sets (80/20 split)
train_df, test_df = df_entrenamiento.randomSplit([0.8, 0.2], seed=67)
lr = LinearRegression(featuresCol="features", labelCol="objetivo_temp")

# Fit model on training data
modelo_clima = lr.fit(train_df)

# Generate predictions on test data
predicciones = modelo_clima.transform(test_df)
# Display prediction results alongside features and true values
display(predicciones.select("features", "objetivo_temp", "prediction"))

# COMMAND ----------

# Save ML predictions (timestamp, actual temp, predicted temp) as Delta table
predicciones.select("fecha_hora", "objetivo_temp", "prediction") \
    .write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("clima_predicciones")