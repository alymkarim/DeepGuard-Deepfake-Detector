const CROP_SIZE = 224;
const FACE_MARGIN = 0.2;
const MAX_DETECTION_WIDTH = 960;
const MP_VERSION = "0.10.14";

const input = document.getElementById("video-input");
const dropzone = document.getElementById("dropzone");
const filename = document.getElementById("filename");
const preview = document.getElementById("preview");
const analyseButton = document.getElementById("analyse");
const frameCount = document.getElementById("frame-count");
const status = document.getElementById("status");
const results = document.getElementById("results");

const workCanvas = document.createElement("canvas");
const cropCanvas = document.createElement("canvas");
cropCanvas.width = CROP_SIZE;
cropCanvas.height = CROP_SIZE;

const workContext = workCanvas.getContext("2d", { willReadFrequently: true });
const cropContext = cropCanvas.getContext("2d", { willReadFrequently: true });

let videoUrl = null;
let detectorPromise = null;

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

/* ---------------------------------------------------------------- faces */

async function createDetector() {
  const vision = await import(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`
  );

  const fileset = await vision.FilesetResolver.forVisionTasks(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
  );

  return vision.FaceDetector.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_detector/" +
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
      delegate: "GPU",
    },
    runningMode: "IMAGE",
    minDetectionConfidence: 0.4,
  });
}

async function getDetector() {
  if (!detectorPromise) {
    detectorPromise = createDetector().catch((error) => {
      console.warn("Face detector unavailable, using full frames", error);
      return null;
    });
  }

  return detectorPromise;
}

function largestFace(detection) {
  let best = null;

  for (const face of detection.detections || []) {
    const box = face.boundingBox;
    if (!best || box.width * box.height > best.width * best.height) {
      best = box;
    }
  }

  return best;
}

/* ---------------------------------------------------------------- video */

function waitFor(eventTarget, eventName, timeout = 4000) {
  return new Promise((resolve, reject) => {
    let settled = false;

    const done = () => {
      if (settled) return;
      settled = true;
      eventTarget.removeEventListener(eventName, done);
      clearTimeout(timer);
      resolve();
    };

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      eventTarget.removeEventListener(eventName, done);
      reject(new Error(`Timed out waiting for "${eventName}".`));
    }, timeout);

    eventTarget.addEventListener(eventName, done);
  });
}

async function seekTo(time) {
  if (Math.abs(preview.currentTime - time) < 0.01) return;
  const waiting = waitFor(preview, "seeked");
  preview.currentTime = time;
  await waiting;
}

function drawCurrentFrame() {
  const scale = Math.min(
    1,
    MAX_DETECTION_WIDTH / Math.max(preview.videoWidth, preview.videoHeight)
  );

  workCanvas.width = Math.round(preview.videoWidth * scale);
  workCanvas.height = Math.round(preview.videoHeight * scale);
  workContext.drawImage(
    preview,
    0,
    0,
    workCanvas.width,
    workCanvas.height
  );
}

function cropToSquare(box) {
  const width = workCanvas.width;
  const height = workCanvas.height;

  const centerX = (box.x + box.width / 2) * width;
  const centerY = (box.y + box.height / 2) * height;
  const side =
    Math.max(box.width * width, box.height * height) *
    (1 + 2 * FACE_MARGIN);

  const x = Math.max(Math.round(centerX - side / 2), 0);
  const y = Math.max(Math.round(centerY - side / 2), 0);
  const x2 = Math.min(Math.round(centerX + side / 2), width);
  const y2 = Math.min(Math.round(centerY + side / 2), height);

  cropContext.drawImage(
    workCanvas,
    x,
    y,
    x2 - x,
    y2 - y,
    0,
    0,
    CROP_SIZE,
    CROP_SIZE
  );
}

function cropFullFrame() {
  cropContext.drawImage(
    workCanvas,
    0,
    0,
    CROP_SIZE,
    CROP_SIZE
  );
}

function encodeCrop() {
  const pixels = cropContext.getImageData(0, 0, CROP_SIZE, CROP_SIZE).data;
  const rgb = new Uint8Array(CROP_SIZE * CROP_SIZE * 3);

  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) {
    rgb[j] = pixels[i];
    rgb[j + 1] = pixels[i + 1];
    rgb[j + 2] = pixels[i + 2];
  }

  let binary = "";
  const chunk = 0x8000;

  for (let i = 0; i < rgb.length; i += chunk) {
    binary += String.fromCharCode.apply(null, rgb.subarray(i, i + chunk));
  }

  return btoa(binary);
}

async function captureFrames(count) {
  const detector = await getDetector();
  const frames = [];
  let facesFound = 0;

  for (let index = 0; index < count; index += 1) {
    const time = ((index + 0.5) / count) * preview.duration;
    await seekTo(Math.min(time, Math.max(preview.duration - 0.05, 0)));

    drawCurrentFrame();

    let box = null;

    if (detector) {
      try {
        box = largestFace(detector.detect(workCanvas));
      } catch (error) {
        console.warn("Detection failed for a frame", error);
      }
    }

    if (box) {
      cropToSquare(box);
      facesFound += 1;
    } else {
      cropFullFrame();
    }

    frames.push(encodeCrop());
    setStatus(`Sampling frame ${index + 1} of ${count}…`);
  }

  return { frames, facesFound };
}

/* -------------------------------------------------------------- request */

async function scoreFrames(frames, facesFound, sampled) {
  const started = performance.now();

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frames,
      width: CROP_SIZE,
      height: CROP_SIZE,
      faces_detected: facesFound,
      sampled_frames: sampled,
    }),
  });

  const body = await response.json();

  if (!response.ok) {
    throw new Error(body.error || `Request failed (${response.status}).`);
  }

  body.latency = Math.round(performance.now() - started);
  return body;
}

/* ---------------------------------------------------------------- render */

function stat(value, label) {
  const element = document.createElement("div");
  element.className = "stat";

  const strong = document.createElement("b");
  strong.textContent = value;

  const span = document.createElement("span");
  span.textContent = label;

  element.append(strong, span);
  return element;
}

function renderFrameBars(probabilities) {
  const container = document.getElementById("frame-bars");
  container.textContent = "";

  for (const probability of probabilities) {
    const bar = document.createElement("div");
    bar.className = "frame-bar" + (probability < 0.5 ? " real" : "");
    bar.title = `${(probability * 100).toFixed(1)}% fake`;

    const fill = document.createElement("i");
    fill.style.height = `${Math.max(probability * 100, 2)}%`;

    const label = document.createElement("em");
    label.textContent = (probability * 100).toFixed(0);

    bar.append(label, fill);
    container.append(bar);
  }
}

function renderResult(result) {
  const verdict = document.getElementById("verdict");
  verdict.textContent = result.prediction;
  verdict.className = `verdict ${result.prediction}`;

  const percentage = (result.fake_probability * 100).toFixed(1);
  document.getElementById("score").textContent = `${percentage}%`;
  document.getElementById("meter-fill").style.width = `${percentage}%`;

  const stats = document.getElementById("stats");
  stats.textContent = "";
  stats.append(
    stat(result.frames, "frames scored"),
    stat(result.faces_detected, "faces found"),
    stat(`${result.latency} ms`, "server time"),
    stat(result.model?.architecture || "efficientnet_b0", "model")
  );

  renderFrameBars(result.frame_probabilities);

  document.getElementById("face-warning").hidden =
    result.faces_detected > 0 || result.sampled_frames === 0;

  results.hidden = false;
}

function renderModelInfo(info) {
  const tbody = document.querySelector("#model-metrics tbody");
  const test = info.test_metrics;

  if (!test || !tbody) return;

  const rows = [
    ["Test accuracy", `${(test.accuracy * 100).toFixed(1)}%`],
    ["Test ROC-AUC", test.roc_auc_fake.toFixed(3)],
    ["Test F1 (fake)", test.f1_fake.toFixed(3)],
  ];

  tbody.textContent = "";

  for (const [label, value] of rows) {
    const row = document.createElement("tr");

    const th = document.createElement("th");
    th.textContent = label;

    const td = document.createElement("td");
    td.textContent = value;

    row.append(th, td);
    tbody.append(row);
  }
}

/* ----------------------------------------------------------------- init */

function acceptFile(file) {
  if (!file || !file.type.startsWith("video/")) {
    setStatus("Please choose a video file.", true);
    return;
  }

  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(file);

  preview.src = videoUrl;
  preview.hidden = false;
  filename.textContent = file.name;
  filename.hidden = false;
  results.hidden = true;
  analyseButton.disabled = false;
  setStatus("");
}

input.addEventListener("change", () => acceptFile(input.files[0]));

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("over");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("over"));

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("over");
  acceptFile(event.dataTransfer.files[0]);
});

analyseButton.addEventListener("click", async () => {
  analyseButton.disabled = true;
  results.hidden = true;
  setStatus("Preparing…");

  try {
    if (preview.readyState < 1) {
      await waitFor(preview, "loadedmetadata");
    }

    if (!Number.isFinite(preview.duration) || preview.duration <= 0) {
      throw new Error("Could not read the video duration.");
    }

    preview.pause();

    const count = Number.parseInt(frameCount.value, 10);
    const captured = await captureFrames(count);

    setStatus(`Scoring ${captured.frames.length} frames…`);

    const result = await scoreFrames(
      captured.frames,
      captured.facesFound,
      count
    );

    renderResult(result);
    setStatus(`Done in ${result.latency} ms.`);
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Something went wrong.", true);
  } finally {
    analyseButton.disabled = false;
  }
});

fetch("/api/model")
  .then((response) => (response.ok ? response.json() : null))
  .then((info) => {
    if (info) renderModelInfo(info);
  })
  .catch(() => {
    /* local preview without the API running */
  });
