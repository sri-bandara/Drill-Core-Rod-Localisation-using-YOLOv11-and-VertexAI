import os 
import argparse
from google.cloud import storage
from ultralytics import YOLO
import yaml


def download_dataset(bucket_name, local_dir):
    print("Downloading dataset from GCS...")

    client = storage.Client() #connects to GCS using credentials from environment
    bucket = client.bucket(bucket_name) #connects to the specified bucket

    blobs = bucket.list_blobs(prefix="datasets/v1/") #blob is a file/object in GCS, lists all blobs with the specified prefix

    for blob in blobs: #loops over all files that look like "datasets/v1/..."
        if blob.name.endswith("/"): #skips directories, we only want files
            continue

        relative_path = blob.name.replace("datasets/v1/", "") #datasets/v1/train/image1.jpg -> train/image1.jpg
        local_path = os.path.join(local_dir, relative_path) #creates a local path like /tmp/dataset/train/image1.jpg

        os.makedirs(os.path.dirname(local_path), exist_ok=True) #creates the local directory if it doesn't exist, e.g. /tmp/dataset/train/

        blob.download_to_filename(local_path) #downloads the file from GCS to the local path

    print(f"Dataset ready at {local_dir}")


def upload_model(bucket_name, model_version):
    print("Uploading trained model to GCS...")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_model_path = "/tmp/runs/train/weights/best.pt"

    destination = f"models/{model_version}/best.pt"
    bucket.blob(destination).upload_from_filename(local_model_path) #creates a blob at the destination path and uploads the model to it

    print(f"Model uploaded to gs://{bucket_name}/{destination}")


def create_data_yaml(local_dir):
    yaml_content = {
        "train": f"{local_dir}/train",
        "val":   f"{local_dir}/val",
        "test":  f"{local_dir}/test",
        "nc":    1,
        "names": ["box"]
    }

    yaml_path = f"{local_dir}/data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    print(f"data.yaml written at {yaml_path}")
    return yaml_path


def main(args):
    local_dir = "/tmp/dataset"

    download_dataset(args.bucket_name, local_dir)

    yaml_path = create_data_yaml(local_dir)

    model = YOLO("yolo11s.pt")
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="/tmp/runs",
        name="train"
    )

    upload_model(args.bucket_name, args.model_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-name",   type=str, default="drill-core-sri")
    parser.add_argument("--epochs",        type=int, default=60)
    parser.add_argument("--imgsz",         type=int, default=640)
    parser.add_argument("--model-version", type=str, default="v1")
    args = parser.parse_args()

    main(args)