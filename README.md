# DeepGuard

A deepfake video detector. Give it an `.mp4` and it tells you how much of it looks fake,
frame by frame.

The project covers the whole path: dataset handling in Cloud Storage, training on Vertex AI,
a local Streamlit tool, and a small web demo that runs the model on Vercel.

**Live demo:** https://deep-guard-deepfake-detector.vercel.app/

## What's here

| Path | What it does |
| --- | --- |
| `src/` | Dataset splitting, frame extraction, face cropping, training, evaluation |
| `app.py` | Local Streamlit app — upload a video, get a verdict |
| `web/` | Vercel deployment: Flask + ONNX Runtime API, plain JS frontend |
| `scripts/` | GCP helpers (download, upload, bucket setup) and `export_onnx.py` |
| `tests/` | 36 tests covering splitting, checkpoint format, face cropping, the web API |

## Results

Honest numbers first, because they aren't great yet.

Test set (15% of the videos, held out before any frames were extracted):

| Metric | Value |
| --- | --- |
| Accuracy | 0.513 |
| ROC-AUC (fake) | 0.548 |
| F1 (fake) | 0.330 |
| Precision (fake) | 0.529 |
| Recall (fake) | 0.240 |
| Loss | 0.714 |

Confusion matrix `[[54, 171], [48, 177]]` (rows = actual, columns = predicted).

That's barely above chance. Training accuracy climbed to 0.775 while validation accuracy
stalled around 0.518 — a textbook overfit on a small, noisy video set. The model and the
plumbing around it work end to end; the weights need more data, more frames per video, and
longer training before any of these numbers are worth quoting as a claim. I'd rather put the
real figures up front than dress up a 51% model.

## Run it locally

```bash
git clone git@github.com:alymkarim/DeepGuard-Deepfake-Detector.git
cd DeepGuard-Deepfake-Detector

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Needs `best_model.pth` in the repo root (it's committed, ~16 MB). Upload a video and hit
analyse — it samples frames, finds the face in each one, and scores them.

There's also a plain Python path if you don't want a UI:

```python
from src.predict_video import VideoDeepfakePredictor

predictor = VideoDeepfakePredictor("best_model.pth")
result = predictor.predict("some_video.mp4")
print(result["prediction"], result["fake_probability"])
```

## Web demo (Vercel)

The demo lives in `web/`. It's deliberately boring on the server side: a Flask function that
loads an ONNX graph and runs inference, and a static page that does everything else in the
browser.

```bash
pip install onnxruntime numpy flask
python scripts/export_onnx.py          # best_model.pth -> web/assets/deepguard.onnx

cd web
python api/predict.py                  # local check on 127.0.0.1:5000
npx vercel deploy                      # then vercel --prod when you like it
```

`export_onnx.py` checks the ONNX output against PyTorch before writing the file, so a bad
export fails loudly instead of shipping.

Face detection happens client-side with MediaPipe's BlazeFace, so only cropped 224×224
face patches ever leave the browser. The API accepts at most 16 of them per request.

## Training pipeline (GCP)

1. `scripts/01_download_kaggle.sh` — pull the Kaggle dataset (`xdxd003/ff-c23`).
2. `scripts/03_create_bucket.sh` — bucket, if you don't have one.
3. `scripts/02_upload_dataset.sh` — put videos in `gs://BUCKET/raw/`.
4. `python -m src.split_dataset` — 70/15/15 split **before** frame extraction, written to
   `manifests/`. Splitting first is the whole point: otherwise frames from the same video
   leak across train and test and the metrics lie.
5. Frames get extracted per manifest into `gs://BUCKET/frames/`.
6. `python scripts/submit_vertex_job.py` with `vertex_cpu_job.yaml` — trains on Vertex AI.

The training itself is two stages: freeze the EfficientNet-B0 backbone and fit the
classifier head, then unfreeze the last two blocks and fine-tune at a much lower learning
rate. Early stopping watches validation loss.

Checkpoint and metrics land in `gs://BUCKET/outputs/`.

## Face detection

OpenCV 5 removed the old Haar cascade classifiers, so face finding uses YuNet
(`assets/face_detection_yunet_2023mar.onnx`, Apache-2.0, from
[opencv_zoo](https://github.com/opencv/opencv_zoo)). The largest detected face is expanded
by 20% and squared off before it goes to the classifier; if nothing is detected the frame
falls back to a centre crop and the result is flagged in the UI.

## Limitations

- The model is near chance on the test set. Treat the demo as a demonstration of the
  pipeline, not a reliable detector.
- It was trained on face crops from one dataset (DFDC-style manipulations). It will happily
  misclassify anything outside that distribution — re-encodes, compression artifacts, other
  forgery methods, real footage that happens to be low quality.
- A single wrong face detection feeds the classifier background instead of a face, and the
  per-frame scores will be noise.
- No video-level temporal model. Frames are scored independently and averaged, so coherent
  but frame-wise-plausible forgeries aren't handled any better than the numbers suggest.

## Credits

- Dataset: [`xdxd003/ff-c23`](https://www.kaggle.com/datasets/xdxd003/ff-c23) on Kaggle.
- Face detection: YuNet from [opencv_zoo](https://github.com/opencv/opencv_zoo) (Apache-2.0).
- Backbone: EfficientNet-B0 via torchvision.
- Browser face detection: MediaPipe Tasks Vision.
