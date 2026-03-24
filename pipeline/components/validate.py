from kfp.dsl import component 

#docker image
TRAINER_IMAGE = "australia-southeast1-docker.pkg.dev/drill-core-object-detection/drill-core-repo/trainer:v2"

@component(base_image=TRAINER_IMAGE) #running this component in the same image as training since it needs access to the same GCS libraries and credentials
def validate_data(
    bucket_name: str, #ex: "drill-core-sri"
    dataset_prefix: str, #ex: "datasets/v1/"
) -> bool:
    """
    Makes sure train/val/test splits all have images
    """
    from google.cloud import storage #the dependency is imported in the environment where the component actually runs

    print("Starting data validation...")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    splits = ["train", "val", "test"]

    for split in splits:
        prefix = f"{dataset_prefix}{split}/images/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if len(blobs) == 0:
            print(f"ERROR: No images found in {split} split!")
            return False

        print(f"  {split}: {len(blobs)} images found")

    print("Data validation passed.")
    return True