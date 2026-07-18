# Deepfake Detector — GCP Training Project

Real-time deepfake detection system built with PyTorch, OpenCV and Google Cloud Platform.

## Features

- EfficientNet-B0
- Video processing
- Face extraction
- Vertex AI training
- FastAPI deployment

GCP-ready pipeline for the Kaggle `xdxd003/ff-c23` dataset.

## Pipeline
1. Download/upload `original` and `Deepfakes` videos.
2. Split videos 70/15/15 before frame extraction.
3. Extract frames to Cloud Storage.
4. Fine-tune EfficientNet-B0 on Vertex AI.
5. Save model and metrics to Cloud Storage.

## GCS layout
```
gs://BUCKET/raw/original/*.mp4
gs://BUCKET/raw/Deepfakes/*.mp4
gs://BUCKET/manifests/{train,validation,test}.csv
gs://BUCKET/frames/{train,validation,test}/{real,fake}/...
gs://BUCKET/outputs/best_model.pth
```

Copy `config/config.example.env` values into your environment. Run the numbered scripts in `scripts/`.

