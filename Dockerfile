# Reproducibility image for the RL-IDS experiments (Protocol #8).
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Datasets are mounted at run time (not shipped in the image):
#   docker run -e DATASET_DIR=/data -v /path/to/nsl_unsw:/data <image>
ENV DATASET_DIR=/data
CMD ["bash", "reproduce.sh"]
