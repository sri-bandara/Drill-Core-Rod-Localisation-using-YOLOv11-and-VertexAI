import os
from PIL import Image
import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")


def predict_image(img, conf_threshold, iou_threshold):
    if img is None:
        return None

    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        show_labels=True,
        show_conf=True,
    )

    for r in results:
        im_array = r.plot()
        return Image.fromarray(im_array[..., ::-1])

    return None


def load_example_image(example_path):
    return Image.open(example_path)


CUSTOM_CSS = """
h1 {
    text-align: center;
    font-size: 2.4rem;
    margin-bottom: 6px;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 1rem;
    color: white;
    margin-bottom: 20px;
}

.section-note {
    text-align: center;
    font-size: 14px;
    color: white;
    margin-top: 16px;
    opacity: 0.9;
}

.examples-title {
    text-align: center;
    font-size: 1rem;
    color: white;
    margin-top: 18px;
    margin-bottom: 12px;
    font-weight: 600;
}

.gradio-container {
    background-color: #C2452D !important;
}

.image-panel,
.control-panel {
    border: 2px solid #c2452D !important;
    border-radius: 12px !important;
    padding: 10px !important;
    box-sizing: border-box !important;
}

.button-panel .gr-button {
    border: 2px solid #c2452D !important;
    border-radius: 12px !important;
}

.control-panel {
    padding-top: 6px !important;
    padding-bottom: 6px !important;
}

.sample-row {
    justify-content: center !important;
    gap: 14px !important;
    margin-top: 6px;
    margin-bottom: 8px;
}

.sample-btn {
    min-width: 120px !important;
    max-width: 140px !important;
}
"""

EXAMPLES_DIR = "examples"
example_paths = []

if os.path.exists(EXAMPLES_DIR):
    for filename in sorted(os.listdir(EXAMPLES_DIR)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            example_paths.append(os.path.join(EXAMPLES_DIR, filename))

example_paths = example_paths[:3]

with gr.Blocks(title="Drill Core Rod Localisation", css=CUSTOM_CSS) as demo:

    gr.HTML("<h1>Drill Core Rod Localisation using YOLOv11 🔍</h1>")
    gr.HTML(
        "<div class='subtitle'>Upload or paste an image to detect and localise drill core rods.</div>"
    )

    with gr.Row():
        with gr.Column(elem_classes="image-panel"):
            image_input = gr.Image(
                type="pil",
                label="Input image",
                sources=["upload", "clipboard"],
                height=500
            )

        with gr.Column(elem_classes="image-panel"):
            output_image = gr.Image(
                type="pil",
                label="Detection result",
                height=500
            )

    with gr.Row():
        with gr.Column(elem_classes="control-panel"):
            conf_slider = gr.Slider(
                0.0, 1.0,
                value=0.30,
                step=0.01,
                label="Confidence"
            )

        with gr.Column(elem_classes="control-panel"):
            iou_slider = gr.Slider(
                0.0, 1.0,
                value=0.60,
                step=0.01,
                label="IoU"
            )

    with gr.Row(elem_classes="button-panel"):
        run_btn = gr.Button("Run", variant="primary")
        clear_btn = gr.ClearButton([image_input, output_image], value="Clear")

    if example_paths:
        gr.HTML("<div class='examples-title'>Try these sample images</div>")

        with gr.Row(elem_classes="sample-row"):
            for i, path in enumerate(example_paths):
                btn = gr.Button(f"Sample {i+1}", elem_classes="sample-btn")
                btn.click(
                    fn=load_example_image,
                    inputs=gr.State(path),
                    outputs=image_input
                )

    gr.HTML(
        """
        <div class="section-note">
            Built with YOLO + Gradio · Supports upload, paste, and sample images
        </div>
        """
    )

    run_btn.click(
        fn=predict_image,
        inputs=[image_input, conf_slider, iou_slider],
        outputs=output_image
    )

demo.launch()