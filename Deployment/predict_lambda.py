import json
import boto3
import time
from datetime import datetime
from text_pre_processing.text_pre_processing import PreProcessor

my_pre_processor = PreProcessor()
sage_maker_client = boto3.client("runtime.sagemaker")
bucket_name = "chenfei-bucket"
s3 = boto3.resource("s3")
# Model weights
weights = [0.1, 0.2, 0.7]

def lambda_handler(event, context):
    now = datetime.now()
    request_date = now.strftime("%d/%m/%Y %H:%M:%S")
    log_path = "/".join(["Sentiment-Analysis", "Lambda-logs", now.strftime("%Y%m%d"), now.strftime("%Y%m%d%H:%M:%S")])
    tweet = event['tweet']
    
    pre_processing_start = time.time()
    features = my_pre_processor.pre_process(tweet)
    pre_processing_end = time.time()
    pre_processing_time = pre_processing_end - pre_processing_start

    model_payload = {
        "embedding_input": features
    }

    inference_start = time.time()
    cnn_5_response = sage_maker_client.invoke_endpoint(
        EndpointName = "CNN-5",
        ContentType = "application/json",
        Body = json.dumps(model_payload)
        )
    cnn_10_response = sage_maker_client.invoke_endpoint(
        EndpointName = "CNN-non-stopwords-10",
        ContentType = "application/json",
        Body = json.dumps(model_payload)
        )
    cnn_lstm_response = sage_maker_client.invoke_endpoint(
        EndpointName = "CnnLstm-non-stopwords-10",
        ContentType = "application/json",
        Body = json.dumps(model_payload)
        )
    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    cnn_5_result = json.loads(cnn_5_response["Body"].read().decode())['predictions'][0][0]
    cnn_10_result = json.loads(cnn_10_response["Body"].read().decode())['predictions'][0][0]
    cnn_lstm_result = json.loads(cnn_lstm_response["Body"].read().decode())['predictions'][0][0]
    response = {}

    prob = weights[0] * cnn_5_result + weights[1] * cnn_10_result + weights[2] * cnn_lstm_result
    if prob >= 0.5:
        response["sentiment"] = "positive"
    else:
        response["sentiment"] = "negative"
    
    log = json.dumps(
        {
            "request_time": request_date,
            "tweet": tweet,
            "pre_processing_time": pre_processing_time,
            "model_inference_time": inference_time,
            "cnn_5_probability": cnn_5_result,
            "cnn_10_probability": cnn_10_result,
            "cnn_lstm_probability": cnn_lstm_result,
            "total_probability": prob,
            "sentiment": response["sentiment"]
        }
    )

    if event.get('logs'):
        s3.Bucket(bucket_name).put_object(Key = log_path, Body = log)
    
    return response
