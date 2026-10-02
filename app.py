import os
import tempfile
from pathlib import Path

import streamlit as st

from src.predict_video import VideoDeepfakePredictor

MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")

st.set_page_config(
    page_title="DeepGuard",
    page_icon="🛡️",
    layout="centered",
)

st.title("DeepGuard 🛡️")
st.write(
    "Upload a video to estimate whether it is real or manipulated."
)


@st.cache_resource
def load_predictor():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model weights not found at '{MODEL_PATH}'. "
            "Download them from Cloud Storage and set MODEL_PATH."
        )

    return VideoDeepfakePredictor(MODEL_PATH)


try:
    predictor = load_predictor()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Analyse video", type="primary"):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_file.name).suffix or ".mp4",
        ) as temporary_file:
            uploaded_file.seek(0)
            temporary_file.write(uploaded_file.read())
            video_path = temporary_file.name

        try:
            with st.spinner("Analysing sampled frames..."):
                result = predictor.predict(video_path)
        except ValueError as error:
            st.error(str(error))
        else:
            st.subheader(result["prediction"].upper())

            probability_column, frames_column = st.columns(2)
            probability_column.metric(
                "Fake probability",
                f"{result['fake_probability'] * 100:.1f}%",
            )
            frames_column.metric(
                "Frames analysed",
                result["frames_used"],
            )

            if result["faces_detected"] == 0:
                st.warning(
                    "No face was detected, so full frames were "
                    "scored instead. Treat this result with "
                    "extra caution."
                )

            st.progress(min(result["fake_probability"], 1.0))

            metadata = predictor.checkpoint_metadata

            if metadata.get("validation_roc_auc_fake") is not None:
                st.caption(
                    "Model: "
                    f"{metadata.get('architecture')} · "
                    f"stage `{metadata.get('training_stage')}` "
                    f"epoch {metadata.get('epoch')} · "
                    "validation ROC-AUC "
                    f"{metadata['validation_roc_auc_fake']:.3f}"
                )

            st.caption(
                "This is an experimental detector and should not "
                "be used as definitive evidence that a video is "
                "authentic or manipulated."
            )

        finally:
            Path(video_path).unlink(missing_ok=True)
