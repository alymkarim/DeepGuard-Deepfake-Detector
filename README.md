# DeepGuard

A deepfake video detector. Give it an `.mp4` and it tells you how much of it looks fake, frame by frame.

The project covers the whole path: splitting the dataset in Cloud Storage, training on Vertex AI, a local Streamlit tool, and a small web demo that runs the model on Vercel.

**Live demo:** https://deep-guard-deepfake-detector.vercel.app/

## What it does

* Scores every frame of a video and returns a real or fake verdict with a probability behind it
* Splits videos 70/15/15 before a single frame is extracted, so no video can land in both train and test
* Trains EfficientNet-B0 in two stages on Vertex AI, backbone frozen first and the last two blocks fine-tuned after
* Exports to ONNX only after checking the output against PyTorch, so a bad export fails before it ships
* Runs inference in a Flask function on Vercel, with face detection done in the browser so the video itself never gets uploaded
* Comes with 36 tests covering the split, the checkpoint format, face cropping and the web API

## What's here

* **`src/`**: dataset splitting, frame extraction, face cropping, training, evaluation
* **`app.py`**: the local Streamlit app
* **`web/`**: the Vercel deployment, a Flask and ONNX Runtime API with a plain JS frontend
* **`scripts/`**: GCP helpers for downloading, uploading and bucket setup, plus `export_onnx.py`
* **`tests/`**: tests for all of the above

## Results

Honest numbers first, because they aren't great yet.

Test set, 15% of the videos, held out before any frames were extracted:

* Accuracy: 0.513
* ROC-AUC (fake): 0.548
* F1 (fake): 0.330
* Precision (fake): 0.529
* Recall (fake): 0.240
* Loss: 0.714

Confusion matrix `[[54, 171], [48, 177]]`, rows are actual and columns are predicted.

That is barely above chance. Training accuracy climbed to 0.775 while validation accuracy stalled around 0.518, which is a textbook overfit on a small and noisy video set. The model and the plumbing around it work end to end. The weights need more data, more frames per video and longer training before any of these numbers are worth quoting as a claim. I would rather put the real figures up front than dress up a 51% model.

## Run it locally

```bash
git clone git@github.com:alymkarim/DeepGuard-Deepfake-Detector.git
cd DeepGuard-Deepfake-Detector

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Needs `best_model.pth` in the repo root. It is committed and about 16 MB. Upload a video and hit analyse. It samples frames, finds the face in each one and scores them.

There is also a plain Python path if you don't want a UI:

```python
from src.predict_video import VideoDeepfakePredictor

predictor = VideoDeepfakePredictor("best_model.pth")
result = predictor.predict("some_video.mp4")
print(result["prediction"], result["fake_probability"])
```

## Web demo (Vercel)

The demo lives in `web/`. It is deliberately boring on the server side: a Flask function loads an ONNX graph and runs inference, while a static page does everything else in the browser.

```bash
pip install onnxruntime numpy flask
python scripts/export_onnx.py          # best_model.pth -> web/assets/deepguard.onnx

cd web
python api/predict.py                  # local check on 127.0.0.1:5000
npx vercel deploy                      # then vercel --prod when you like it
```

`export_onnx.py` runs the ONNX graph and the PyTorch model over the same input and compares them before it writes the file, so it never quietly ships something that disagrees with the checkpoint.

Face detection happens client side with MediaPipe's BlazeFace, so only cropped 224x224 face patches ever leave the browser. The API accepts at most 16 of them per request.

## Training pipeline (GCP)

1. `scripts/01_download_kaggle.sh` pulls the Kaggle dataset (`xdxd003/ff-c23`).
2. `scripts/03_create_bucket.sh` makes a bucket if you don't have one.
3. `scripts/02_upload_dataset.sh` puts the videos in `gs://BUCKET/raw/`.
4. `python -m src.split_dataset` writes a 70/15/15 split into `manifests/` before any frames exist. Splitting first is the whole point. Do it the other way round and frames from the same video end up on both sides, and the metrics lie.
5. Frames are extracted per manifest into `gs://BUCKET/frames/`.
6. `python scripts/submit_vertex_job.py` with `vertex_cpu_job.yaml` trains on Vertex AI.

Training runs in two stages. The EfficientNet-B0 backbone starts frozen while the classifier head fits, then the last two blocks unfreeze and fine-tune at a much lower learning rate. Early stopping watches validation loss.

Checkpoints and metrics land in `gs://BUCKET/outputs/`.

## Face detection

OpenCV 5 removed the old Haar cascade classifiers, so face finding moved to YuNet (`assets/face_detection_yunet_2023mar.onnx`, Apache 2.0, from [opencv_zoo](https://github.com/opencv/opencv_zoo)). The largest detected face is expanded by 20% and squared off before it reaches the classifier. If nothing is detected the frame falls back to a centre crop and the result is flagged in the UI.

## Limitations

* The model is near chance on the test set. Treat the demo as a demonstration of the pipeline rather than a reliable detector.
* It was trained on face crops from a single dataset of DFDC style manipulations. It will happily misclassify anything outside that distribution: re-encodes, compression artifacts, other forgery methods, and real footage that happens to be low quality.
* One bad face detection hands the classifier background instead of a face, and the scores for that frame are noise from then on.
* There is no temporal model. Frames are scored one at a time and then averaged, so a forgery that holds up frame by frame but not across a sequence is not caught any better than the numbers suggest.

## Credits

* Dataset: [`xdxd003/ff-c23`](https://www.kaggle.com/datasets/xdxd003/ff-c23) on Kaggle
* Face detection: YuNet from [opencv_zoo](https://github.com/opencv/opencv_zoo), Apache 2.0
* Backbone: EfficientNet-B0 via torchvision
* Browser face detection: MediaPipe Tasks Vision
