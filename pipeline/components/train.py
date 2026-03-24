from kfp.dsl import component

#docker image
TRAINER_IMAGE = "australia-southeast1-docker.pkg.dev/drill-core-object-detection/drill-core-repo/trainer:v2"

@component(base_image=TRAINER_IMAGE) #running this component in the same image as training since it needs access to the same GCS libraries and credentials
def train_model(
    bucket_name: str,
    model_version: str,
    epochs: int,
    imgsz: int,
):
    """
    Triggers the YOLO training job inside the Docker container.
    Calls trainer/train.py with the provided arguments
    """
    import subprocess

    print(f"Starting training — epochs={epochs}, imgsz={imgsz}")

    subprocess.run([ #running the training script using terminal command, passing the arguments from the component
        "python", "train.py",
        "--bucket-name",   bucket_name,
        "--epochs",        str(epochs),
        "--imgsz",         str(imgsz),
        "--model-version", model_version,
    ], check=True)

    print("Training complete!")