import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import boto3

s3 = boto3.client('s3')
bucket = 'adithya-medical-llm-dataset'
prefix = 'general_test/'

os.makedirs('general_test', exist_ok=True)

response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
for obj in response.get('Contents', []):
    key = obj['Key']
    if key.endswith('.bin'):
        local_path = os.path.join('general_test', os.path.basename(key))
        s3.download_file(bucket, key, local_path)
        print(f"Downloaded {local_path}")