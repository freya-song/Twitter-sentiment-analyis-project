import json
import boto3
import time
from datetime import datetime
from text_pre_processing.text_pre_processing import PreProcessor

my_pre_processor = PreProcessor(256)
sage_maker_client = boto3.client("runtime.sagemaker")
bucket_name = "chenfei-bucket"
s3 = boto3.resource("s3")

def lambda_handler(event, context):
    now = datetime.now()
    request_date = now.strftime("%d/%m/%Y %H:%M:%S")
    log_path = "/".join(["HW5", "Lambda-logs", now.strftime("%Y%m%d"), now.strftime("%Y%m%d%H:%M:%S")])
    tweet = event['tweet']
    
    pre_processing_start = time.time()
    features = my_pre_processor.pre_process(tweet)
    pre_processing_end = time.time()
    pre_processing_time = pre_processing_end - pre_processing_start

    model_payload = {
        "embedding_input": features
    }

    inference_start = time.time()
    model_response = sage_maker_client.invoke_endpoint(
        EndpointName = "CNN",
        ContentType = "application/json",
        Body = json.dumps(model_payload)
        )
    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    result = json.loads(model_response["Body"].read().decode())
    response = {}

    if result["predictions"][0][0] >= 0.5:
        response["sentiment"] = "positive"
    else:
        response["sentiment"] = "negative"

    log = json.dumps(
        {
            "request_time": request_date,
            "tweet": tweet,
            "pre_processing_time": pre_processing_time,
            "model_inference_time": inference_time,
            "probability": result["predictions"][0][0],
            "sentiment": response["sentiment"]
        },
        indent = 2)

    s3.Bucket(bucket_name).put_object(Key = log_path, Body = log)
    
    return response
