import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from text_pre_processing.text_pre_processing import PreProcessor

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
my_processor = PreProcessor(50, 400002)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
## @type: DataSource
## @args: [database = "hw4", table_name = "training", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "hw4", table_name = "training", transformation_ctx = "datasource0")
## @type: ApplyMapping
## @args: [mapping = [("col1", "string", "label", "string"), ("col2", "string", "text", "string")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]
applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("col1", "string", "label", "string"), ("col2", "string", "text", "string")], transformation_ctx = "applymapping1")

def map_function(record):
    label = record['label']
    tweet = record['text']
    #Remove the bad data point
    if label == 'Sentiment':
        return None
    feature = my_processor.pre_process(tweet)
    record['feature'] = feature
    return record
    
mapping1 = Map.apply(frame = applymapping1, f = map_function, transformation_ctx = "mapping1")

## @type: DataSink
## @args: [connection_type = "s3", connection_options = {"path": "s3://chenfei-bucket/HW4/ETL-output"}, format = "json", transformation_ctx = "datasink2"]
## @return: datasink2
## @inputs: [frame = applymapping1]
datasink2 = glueContext.write_dynamic_frame.from_options(frame = mapping1, connection_type = "s3", connection_options = {"path": "s3://chenfei-bucket/HW4/ETL-output"}, format = "json", transformation_ctx = "datasink2")
job.commit()
