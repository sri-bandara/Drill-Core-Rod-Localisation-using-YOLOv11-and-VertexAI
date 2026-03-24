from kfp.dsl import component

#docker image
TRAINER_IMAGE = "australia-southeast1-docker.pkg.dev/drill-core-object-detection/drill-core-repo/trainer:v2"

@component(base_image=TRAINER_IMAGE) #running this component in the same image as training since it needs access to the same GCS libraries and credentials
def register_model(
    bucket_name: str,
    model_version: str,
    project: str,
    location: str,
    map50: float,
):
    """
    Registers the trained model in Vertex AI Model Registry.
    Attaches the mAP50 score as metadata so you can compare versions
    """
    from google.cloud import aiplatform #vertex ai sdk

    aiplatform.init(project=project, location=location) #initialize the Vertex AI SDK with the project and location

    model_uri = f"gs://{bucket_name}/models/{model_version}" #pointing to trained model artifacts in GCS
    print(f"Registering model version {model_version}...")
    print(f"Model weights at: {model_uri}")

    model = aiplatform.Model.upload( #registers the model in Vertex AI Model Registry
        display_name=f"drill-core-yolo-{model_version}", #a human readable name for the model in the registry
        artifact_uri=model_uri, #the location of the model artifacts in GCS
        serving_container_image_uri="australia-southeast1-docker.pkg.dev/drill-core-object-detection/drill-core-repo/trainer:v2", #the container image to use for serving predictions with this model
        labels={
            "version": model_version,
            "framework": "yolo",
            "map50": str(round(map50, 4)).replace(".", "_"),
        },
    )

    print(f"Model successfully registered!")
    print(f"Resource name: {model.resource_name}")
    print(f"mAP50: {map50:.4f}")