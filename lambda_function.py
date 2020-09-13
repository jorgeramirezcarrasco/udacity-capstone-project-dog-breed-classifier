import boto3
import base64
import os
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

def lambda_handler(event, context):

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')
    
    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName = ENDPOINT_NAME,    # The name of the endpoint we created
                                       ContentType = 'application/json',                 # The data format that is expected
                                       Body = event['body'])                       # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8')

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 
                      'Access-Control-Allow-Origin' : '*',
                      'Access-Control-Allow-Headers': 'Content-Type',
                      'Access-Control-Allow-Methods': 'OPTIONS,GET,POST' },
        'body' : result
    }