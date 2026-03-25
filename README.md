# Drill Core Rod Localisation using YOLOv11 🔍

End-to-end computer vision pipeline for detecting and localising drill core rods using YOLOv11, built as a reproducible Vertex AI pipeline and deployed as an interactive Gradio app on Hugging Face Spaces.

👉🏽 **Live demo:** [Drill Core Rod Localiser on Hugging Face Spaces](https://huggingface.co/spaces/sri-bandara/Drill-Core-Rod-Localiser-YOLOv11)

---

## Overview 📍

- **Input:** Drill core tray image  
- **Output:** Bounding boxes localising drill core rods  
- **Model:** YOLOv11 (Ultralytics)  
- **Pipeline:** Vertex AI Pipelines (Kubeflow Pipelines)  
- **Deployment:** Gradio, Hugging Face Spaces
- **Final Performance:**  
  - **mAP@50:** 0.965  
  - **mAP@50–95:** 0.677  
  - **Precision:** 0.942  
  - **Recall:** 0.946  

---

## Dataset 📊

The dataset used in this project was manually annotated using Label Studio and consists of 475 images in total, including 400 positive samples containing drill core rods and 75 negative samples with no target objects. The dataset is hosted in a Google Cloud Storage bucket and is structured into standard object detection splits. Images were captured under varying conditions, including different camera angles and perspectives, to improve the model’s ability to generalise across real-world scenarios. This variation allows the model to better handle differences in orientation, spacing, and tray layouts commonly observed in drill core imagery.

---

## Process ⚙️

Initial experimentation and model prototyping were conducted in a Google Colab notebook using an NVIDIA Tesla T4 GPU, allowing for rapid iteration and validation of the training approach. Once a suitable baseline was established, the project was migrated to Google Cloud Vertex AI to build a fully reproducible and scalable machine learning pipeline.

The pipeline was implemented using Kubeflow Pipelines (KFP) and consists of several stages. First, the dataset is validated to ensure that all required splits are present and contain data. Training is then executed using a custom training job, where the YOLOv11 model is trained on the dataset with configurable parameters. Following training, the model is evaluated using mAP50 to assess detection performance. If the model satisfies the predefined performance threshold, it is registered in the Vertex AI Model Registry, enabling versioning and traceability of model iterations.

For deployment, a lightweight and user-friendly inference interface was built using Gradio and hosted on Hugging Face Spaces, allowing users to upload or paste images and visualise detection results in real time.

---

## Assumptions and Limitations 🚧

The model assumes that input images are captured under relatively consistent conditions, particularly with adequate lighting and a near top-down (vertically straight) camera angle. Performance may degrade when images are taken at steep angles, as the model has limited exposure to such perspectives in the training data. Additionally, the model may struggle in scenarios where drill cores appear significantly rotated or partially occluded.


---

## Demo 🔮

<img width="1216" height="773" alt="pipeline" src="https://github.com/user-attachments/assets/f35a6a91-fba1-4161-b2b4-fb14a0913224" />

<img width="1456" height="830" alt="demo" src="https://github.com/user-attachments/assets/227435a9-15da-4dd9-92c3-d7ddf084245d" />

