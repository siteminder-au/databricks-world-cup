# Databricks notebook source
# MAGIC %md
# MAGIC # Driver notebook
# MAGIC
# MAGIC - [agent]($./agent): contains the code to build the agent.
# MAGIC - [config.yml]($./config.yml): contains the configurations.
# MAGIC - [**driver**]($./driver): logs, evaluate, registers, and deploys the agent.
# MAGIC
# MAGIC This notebook uses Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) to deploy the agent defined in the [agent]($./agent) notebook. The notebook does the following:
# MAGIC 1. Logs the agent to MLflow
# MAGIC 2. Evaluate the agent with Agent Evaluation
# MAGIC 3. Registers the agent to Unity Catalog
# MAGIC 4. Deploys the agent to a Model Serving endpoint
# MAGIC
# MAGIC ## Prerequisities
# MAGIC
# MAGIC - Review the contents of [config.yml]($./config.yml) as it defines the tools available to your agent, the LLM endpoint, and the agent prompt.
# MAGIC - Review and run the [agent]($./agent) notebook in this folder to view the agent's code, iterate on the code, and test outputs.
# MAGIC
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow langchain==0.2.16 langgraph-checkpoint==1.0.12  langchain_core langchain-community==0.2.16 langgraph==0.2.16 pydantic
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Log the agent as code from the [agent]($./agent) notebook. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Log the model to MLflow
import os
import mlflow

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "[\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T00:06:05.095Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"reservation in store\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T00:06:05.112Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"reservation created or found\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample01\",\n    \"time\": \"2024-07-09T00:06:05.122Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"tracking event logged\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample01\",\n    \"subscriberCodes\":[\"subscriberSampleCode01\"],\n    \"time\": \"2024-07-09T00:06:05.122Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"smx reservation event published\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"subscriberCode\":\"subscriberSampleCode01\",\n    \"time\": \"2024-07-09T00:06:06.407Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"subscriberCode\":\"subscriberSampleCode01\",,\n    \"subscriberHotelCode\":\"subscriberHotelCodeSample02\",\n    \"time\": \"2024-07-09T00:06:06.417Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"POST /reservations 200 696.094303\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T01:37:16.134Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"reservation in store\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T01:37:16.147Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"reservation created or found\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample05\",\n    \"time\": \"2024-07-09T01:37:16.154Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"tracking event logged\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample05\",\n    \"subscriberCodes\":[\"subscriberCodeSample03\",\"subscriberCodeSample04\"],\n    \"time\": \"2024-07-09T01:37:16.154Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"smx reservation event published\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample03\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:37:25.213Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample03\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:37:25.223Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 200 2965.766002\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 500,\n    \"time\": \"2024-07-09T01:37:56.709Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 500 10071.976669\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:38:25.575Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:38:25.584Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 200 2965.766002\"\n  }\n]"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            'agent',
        ),
        pip_requirements=[
            f"langchain==0.2.16",
            f"langchain-community==0.2.16",
            f"langgraph-checkpoint==1.0.12",
            f"langgraph==0.2.16",
            f"pydantic",
        ],
        model_config="config.yml",
        artifact_path='agent',
        input_example=input_example,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.

# COMMAND ----------

import pandas as pd

eval_examples = [
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "[\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T00:06:05.095Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"reservation in store\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T00:06:05.112Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"reservation created or found\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample01\",\n    \"time\": \"2024-07-09T00:06:05.122Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"tracking event logged\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample01\",\n    \"subscriberCodes\":[\"subscriberSampleCode01\"],\n    \"time\": \"2024-07-09T00:06:05.122Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"smx reservation event published\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"subscriberCode\":\"subscriberSampleCode01\",\n    \"time\": \"2024-07-09T00:06:06.407Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"subscriberCode\":\"subscriberSampleCode01\",,\n    \"subscriberHotelCode\":\"subscriberHotelCodeSample02\",\n    \"time\": \"2024-07-09T00:06:06.417Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"POST /reservations 200 696.094303\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T01:37:16.134Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"reservation in store\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T01:37:16.147Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"reservation created or found\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample05\",\n    \"time\": \"2024-07-09T01:37:16.154Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"tracking event logged\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample05\",\n    \"subscriberCodes\":[\"subscriberCodeSample03\",\"subscriberCodeSample04\"],\n    \"time\": \"2024-07-09T01:37:16.154Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"smx reservation event published\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample03\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:37:25.213Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample03\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:37:25.223Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 200 2965.766002\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 500,\n    \"time\": \"2024-07-09T01:37:56.709Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 500 10071.976669\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:38:25.575Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:38:25.584Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 200 2965.766002\"\n  }\n]"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "[\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T00:06:05.095Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"reservation in store\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T00:06:05.112Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"reservation created or found\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample01\",\n    \"time\": \"2024-07-09T00:06:05.122Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"tracking event logged\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample01\",\n    \"subscriberCodes\":[\"subscriberSampleCode01\"],\n    \"time\": \"2024-07-09T00:06:05.122Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"smx reservation event published\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"subscriberCode\":\"subscriberSampleCode01\",\n    \"time\": \"2024-07-09T00:06:06.407Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"subscriberCode\":\"subscriberSampleCode01\",,\n    \"subscriberHotelCode\":\"subscriberHotelCodeSample02\",\n    \"time\": \"2024-07-09T00:06:06.417Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"POST /reservations 200 696.094303\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T01:37:16.134Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"reservation in store\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T01:37:16.147Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"reservation created or found\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample05\",\n    \"time\": \"2024-07-09T01:37:16.154Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"tracking event logged\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample05\",\n    \"subscriberCodes\":[\"subscriberCodeSample03\",\"subscriberCodeSample04\"],\n    \"time\": \"2024-07-09T01:37:16.154Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"smx reservation event published\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample03\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:37:25.213Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample03\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:37:25.223Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 200 2965.766002\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 500,\n    \"time\": \"2024-07-09T01:37:56.709Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 500 10071.976669\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:38:25.575Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:38:25.584Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 200 2965.766002\"\n  }\n]"
                },
                {
                    "role": "assistant",
                    "content": "This appears to be a JSON array of log messages from a system that handles hotel reservations. Each log message is an object with several properties, including:\n\n* `reservationId`: a unique identifier for the reservation\n* `system`: the system that generated the log message (e.g. \"smx-core\" or \"subscriberx\")\n* `component`: the component of the system that generated the log message (e.g. \"res-pipeline-api\" or \"delivery-api\")\n* `time`: the timestamp of the log message in ISO 8601 format\n* `traceToken`: a unique identifier for the log message\n* `msg`: a human-readable message describing the event that triggered the log message\n* `publisherCode`, `publisherHotelCode`, `subscriberCode`, `subscriberHotelCode`, `statusCode`: additional metadata about the reservation and the systems involved in processing it\n\nThe log messages appear to be related to the processing of hotel reservations, including events such as:\n\n* \"reservation in store\"\n* \"reservation created or found\"\n* \"tracking event logged\"\n* \"smx reservation event published\"\n* \"delivered reservation\"\n* \"POST /reservations\" (with a status code and response time)\n\nThere are two separate reservations being processed in this log output, with IDs \"reservationIdSample02\" and \"reservationIdSample01\". The log messages for each reservation are interleaved in the output.\n\nIs there something specific you would like to know or do with this data?"
                },
                {
                    "role": "user",
                    "content": "What happen to reservation #reservationIdSample02?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "[\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T00:06:05.095Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"reservation in store\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T00:06:05.112Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"reservation created or found\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample01\",\n    \"time\": \"2024-07-09T00:06:05.122Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"tracking event logged\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample01\",\n    \"subscriberCodes\":[\"subscriberSampleCode01\"],\n    \"time\": \"2024-07-09T00:06:05.122Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"smx reservation event published\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"subscriberCode\":\"subscriberSampleCode01\",\n    \"time\": \"2024-07-09T00:06:06.407Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample02\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"samplePublisherCode01\",\n    \"subscriberCode\":\"subscriberSampleCode01\",,\n    \"subscriberHotelCode\":\"subscriberHotelCodeSample02\",\n    \"time\": \"2024-07-09T00:06:06.417Z\",\n    \"traceToken\": \"85511dbb-fb5c-44b3-83b2-9a4c291d9645\",\n    \"msg\": \"POST /reservations 200 696.094303\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T01:37:16.134Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"reservation in store\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"time\": \"2024-07-09T01:37:16.147Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"reservation created or found\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample05\",\n    \"time\": \"2024-07-09T01:37:16.154Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"tracking event logged\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"smx-core\",\n    \"component\": \"res-pipeline-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"publisherHotelCode\":\"publisherHotelCodeSample05\",\n    \"subscriberCodes\":[\"subscriberCodeSample03\",\"subscriberCodeSample04\"],\n    \"time\": \"2024-07-09T01:37:16.154Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"smx reservation event published\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample03\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:37:25.213Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample03\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:37:25.223Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 200 2965.766002\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 500,\n    \"time\": \"2024-07-09T01:37:56.709Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 500 10071.976669\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:38:25.575Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"delivered reservation\"\n  },\n  {\n    \"reservationId\": \"reservationIdSample01\",\n    \"system\": \"subscriberx\",\n    \"component\": \"delivery-api\",\n    \"publisherCode\":\"publisherCodeSample02\",\n    \"subscriberCode\":\"subscriberCodeSample04\",\n    \"statusCode\": 200,\n    \"time\": \"2024-07-09T01:38:25.584Z\",\n    \"traceToken\": \"53e97d09-4704-4761-9533-724594f31ab1\",\n    \"msg\": \"POST /reservations 200 2965.766002\"\n  }\n]"
                },
                {
                    "role": "assistant",
                    "content": "This appears to be a JSON array of log messages from a system that handles hotel reservations. Each log message is an object with several properties, including:\n\n* `reservationId`: a unique identifier for the reservation\n* `system`: the system that generated the log message (e.g. \"smx-core\" or \"subscriberx\")\n* `component`: the component of the system that generated the log message (e.g. \"res-pipeline-api\" or \"delivery-api\")\n* `time`: the timestamp of the log message in ISO 8601 format\n* `traceToken`: a unique identifier for the log message\n* `msg`: a human-readable message describing the event that triggered the log message\n* `publisherCode`, `publisherHotelCode`, `subscriberCode`, `subscriberHotelCode`, `statusCode`: additional metadata about the reservation and the systems involved in processing it\n\nThe log messages appear to be related to the processing of hotel reservations, including events such as:\n\n* \"reservation in store\"\n* \"reservation created or found\"\n* \"tracking event logged\"\n* \"smx reservation event published\"\n* \"delivered reservation\"\n* \"POST /reservations\" (with a status code and response time)\n\nThere are two separate reservations being processed in this log output, with IDs \"reservationIdSample02\" and \"reservationIdSample01\". The log messages for each reservation are interleaved in the output.\n\nIs there something specific you would like to know or do with this data?"
                },
                {
                    "role": "user",
                    "content": "What happen to reservation #reservationIdSample02?"
                },
                {
                    "role": "assistant",
                    "content": "Based on the log messages, here is what happened to reservation #reservationIdSample02:\n\n1. The reservation was stored in the system (`\"msg\": \"reservation in store\"`).\n2. The reservation was created or found in the system (`\"msg\": \"reservation created or found\"`).\n3. A tracking event was logged for the reservation (`\"msg\": \"tracking event logged\"`).\n4. The reservation event was published to subscribers (`\"msg\": \"smx reservation event published\"`).\n5. The reservation was delivered to the subscriber \"subscriberSampleCode01\" (`\"msg\": \"delivered reservation\"`).\n6. The delivery of the reservation to \"subscriberSampleCode01\" was successful, with a status code of 200 and a response time of 696.094303ms (`\"msg\": \"POST /reservations 200 696.094303\"`).\n\nOverall, it appears that the reservation was successfully processed and delivered to the subscriber \"subscriberSampleCode01\"."
                },
                {
                    "role": "user",
                    "content": "How about reservation #reservationIdSample01?"
                }
            ]
        },
        "expected_response": None
    }
]

eval_dataset = pd.DataFrame(eval_examples)
display(eval_dataset)

# COMMAND ----------

import mlflow
import pandas as pd

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",  # replace `chain` with artifact_path that you used when calling log_model.
        data=eval_dataset,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "main"
schema = "default"
model_name = ""
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

# Deploy the model to the review app and a model serving endpoint
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version)