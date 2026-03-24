from kfp.dsl import component, Output, Metrics

# docker image
TRAINER_IMAGE = "australia-southeast1-docker.pkg.dev/drill-core-object-detection/drill-core-repo/trainer:v2"

@component(base_image=TRAINER_IMAGE) #running this component in the same image as training since it needs access to the same GCS libraries and credentials
def evaluate_model(
    bucket_name: str,
    model_version: str,
    metrics: Output[Metrics], #used to log evaluation metrics to Vertex AI Experiments
) -> float: #returns mAP50 as a float so the pipeline can decide whether to deploy
    """
    Downloads best.pt from GCS and runs it against test set
    Logs mAP50, precision and recall to Vertex AI Experiments
    Returns mAP50 as a float so the pipeline can decide whether to deploy
    """
    import os
    import yaml
    from google.cloud import storage
    from ultralytics import YOLO

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    print("Downloading trained model from GCS...")
    os.makedirs("/tmp/model", exist_ok=True)
    bucket.blob(f"models/{model_version}/best.pt").download_to_filename(
        "/tmp/model/best.pt"
    )

    print("Downloading test dataset from GCS...")
    blobs = bucket.list_blobs(prefix="datasets/v1/")
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        local_path = blob.name.replace("datasets/v1/", "/tmp/dataset/")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

    yaml_content = {
        "train": "/tmp/dataset/train",
        "val":   "/tmp/dataset/val",
        "test":  "/tmp/dataset/test",
        "nc":    1,
        "names": ["box"]
    }
    with open("/tmp/dataset/data.yaml", "w") as f:
        yaml.dump(yaml_content, f)

    print("Running evaluation on test set...")
    model = YOLO("/tmp/model/best.pt")
    results = model.val(data="/tmp/dataset/data.yaml", split="test")

    map50     = float(results.box.map50)
    precision = float(results.box.mp)
    recall    = float(results.box.mr)

    print(f"Results — mAP50: {map50:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}")

    #log metrics to Vertex AI Experiments
    metrics.log_metric("mAP50",     map50)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall",    recall)

    return map50