# using a base image with Python 3.10 and CUDA 11.8 for GPU support
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu118.py310

# setting the working directory inside the container
WORKDIR /app

#installing libraries for OpenCV / YOLO
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# copying requirements.txt file into the container
COPY requirements.txt .

# installing the python packages
RUN pip install --no-cache-dir -r requirements.txt

# copying training script into the container
COPY train.py .

# running the training script when the container starts
ENTRYPOINT ["python", "train.py"]