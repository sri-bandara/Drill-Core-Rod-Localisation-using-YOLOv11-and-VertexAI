import google.cloud.aiplatform as aip #vertex ai python sdk
from kfp import compiler #kfp compiler to compile pipeline 
from kfp.dsl import pipeline, If #pipeline decorator and if for conditional execution

#imports components
from components.validate import validate_data 
from components.train import train_model
from components.evaluate import evaluate_model
from components.deploy import register_model

#configurations
PROJECT_ID      = "drill-core-object-detection"
LOCATION        = "australia-southeast1"
BUCKET_NAME     = "drill-core-sri"
PIPELINE_ROOT   = f"gs://{BUCKET_NAME}/pipeline-runs"
MODEL_VERSION   = "v1"
EPOCHS          = 60
IMGSZ           = 640
MAP50_THRESHOLD = 0.8


@pipeline(
    name="drill-core-yolo-pipeline",
    description="Validates data, trains YOLO, evaluates, and registers model (conditional) to model registry",
    pipeline_root=PIPELINE_ROOT,
)
def drill_core_pipeline(
    bucket_name:   str = BUCKET_NAME,
    model_version: str = MODEL_VERSION,
    epochs:        int = EPOCHS,
    imgsz:         int = IMGSZ,
):
    #validation
    validation_step = validate_data(
        bucket_name=bucket_name,
        dataset_prefix="datasets/v1/",
    ).set_memory_limit("4G").set_cpu_limit("2")

    #train (not using GPU for simplicity and to avoid quota issues)
    training_step = train_model(
        bucket_name=bucket_name,
        model_version=model_version,
        epochs=epochs,
        imgsz=imgsz,
    ).after(validation_step)\
     .set_memory_limit("16G")\
     .set_cpu_limit("8")

    #evaluate
    evaluation_step = evaluate_model(
        bucket_name=bucket_name,
        model_version=model_version,
    ).after(training_step)\
     .set_memory_limit("8G")\
     .set_cpu_limit("4")

    #register — only if mAP50 clears the threshold
    with If(
        condition=evaluation_step.outputs["Output"] > MAP50_THRESHOLD,
        name="check-map50-threshold"
    ):
        register_model(
            bucket_name=bucket_name,
            model_version=model_version,
            project=PROJECT_ID,
            location=LOCATION,
            map50=evaluation_step.outputs["Output"],
        )


if __name__ == "__main__":
    #compiles the pipeline to a YAML file because Vertex AI Pipelines runs a YAML specification of the pipeline
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=drill_core_pipeline,
        package_path="pipeline.yaml",
    )
    print("Pipeline compiled to pipeline.yaml")

    aip.init(project=PROJECT_ID, location=LOCATION) #initializes the Vertex AI SDK with the project and location

    job = aip.PipelineJob( #creates a PipelineJob object which represents the pipeline run
        display_name="drill-core-yolo-pipeline",
        template_path="pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "bucket_name":   BUCKET_NAME,
            "model_version": MODEL_VERSION,
            "epochs":        EPOCHS,
            "imgsz":         IMGSZ,
        },
    )

    job.submit() #submits the pipeline job to Vertex AI for execution
    print("Pipeline submitted! Check Vertex AI console to monitor progress.")
    print(f"https://console.cloud.google.com/vertex-ai/pipelines?project={PROJECT_ID}")