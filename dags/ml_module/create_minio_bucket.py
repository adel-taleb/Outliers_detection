import boto3
import os
import torch

def create_minio_bucket():
    print("Torch availability:", torch.cuda.is_available()) 
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
    )
    for bucket in ["databucket", "mlartifactbucket"]:
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"Bucket '{bucket}' already exists.")
        except:
            s3.create_bucket(Bucket=bucket)
            print(f"Bucket '{bucket}' created successfully.")


if __name__ == "__main__":
    create_minio_bucket()
