/**
 * RoboEye Web — 瀏覽器端 CNN 即時年齡 / 性別預測
 *
 * 架構:
 *   face-api.js (SSD MobileNet) → 人臉偵測
 *   ONNX Runtime Web → 年齡 / 性別推論 (多模型切換)
 */

const GENDER_LABELS = ["Male", "Female"];
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const MODEL_INPUT_SIZE = 224;
const NUM_AGE_CLASSES = 101;

// Caffe model constants
const CAFFE_AGE_LABELS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
];
const CAFFE_MEAN = [78.4263377603, 87.7689143744, 114.895847746]; // BGR
const CAFFE_INPUT_SIZE = 227;

// face-api.js model URL
const FACE_API_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/";

// Model sessions
const sessions = {};
let isRunning = false;
let animFrameId = null;

// DOM
const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const statusEl = document.getElementById("status");
const fpsEl = document.getElementById("fps");
const faceCountEl = document.getElementById("face-count");

// ── Init ──────────────────────────────────────────────

async function init() {
    try {
        // Load face-api.js face detector
        statusEl.textContent = "載入人臉偵測模型...";
        try {
            await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL);
            console.log("[INFO] tinyFaceDetector 載入成功");
        } catch (e1) {
            console.warn("[WARN] CDN 載入失敗, 嘗試 ssdMobilenetv1...", e1.message);
            await faceapi.nets.ssdMobilenetv1.loadFromUri(FACE_API_MODEL_URL);
        }

        // Load ONNX age/gender models
        statusEl.textContent = "載入年齡/性別 CNN 模型...";

        await Promise.allSettled([
            loadModel("mobilenet", "models/roboeye.onnx"),
            loadModel("caffe_age", "models/caffe_age.onnx"),
            loadModel("caffe_gender", "models/caffe_gender.onnx"),
            loadModel("insightface", "models/insightface.onnx"),
        ]);

        const available = Object.keys(sessions);
        console.log("[INFO] 已載入模型:", available);

        if (available.length === 0) {
            statusEl.textContent = "沒有可用的模型，請先匯出 ONNX 模型至 docs/models/";
            return;
        }

        statusEl.textContent = `模型載入完成 (${available.join(", ")}) — 點擊「開始偵測」`;
        btnStart.disabled = false;
    } catch (err) {
        statusEl.textContent = `模型載入失敗: ${err.message}`;
        console.error("[ERROR] init failed:", err);
    }
}

async function loadModel(name, path) {
    try {
        sessions[name] = await ort.InferenceSession.create(path, {
            executionProviders: ["wasm"],
        });
        console.log(`[INFO] ${name} 模型載入成功`);
    } catch (err) {
        console.warn(`[WARN] ${name} 模型載入失敗 (${path}):`, err.message);
    }
}

// ── Webcam ────────────────────────────────────────────

async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
    });
    video.srcObject = stream;
    await new Promise((resolve) => (video.onloadedmetadata = resolve));
    video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
}

function stopWebcam() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach((t) => t.stop());
        video.srcObject = null;
    }
}

// ── Preprocessing: MobileNetV2 (match Python TRANSFORM) ──

function preprocessMobileNet(faceCanvas) {
    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = MODEL_INPUT_SIZE;
    resizeCanvas.height = MODEL_INPUT_SIZE;
    const rctx = resizeCanvas.getContext("2d");
    rctx.drawImage(faceCanvas, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

    const imageData = rctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    const { data } = imageData;

    // CHW float32 with ImageNet normalization
    const floatData = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
    const pixelCount = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

    for (let i = 0; i < pixelCount; i++) {
        const r = data[i * 4] / 255.0;
        const g = data[i * 4 + 1] / 255.0;
        const b = data[i * 4 + 2] / 255.0;

        floatData[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        floatData[pixelCount + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        floatData[2 * pixelCount + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }

    return new ort.Tensor("float32", floatData, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
}

// ── Preprocessing: Caffe models ───────────────────────

function preprocessCaffe(faceCanvas) {
    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = CAFFE_INPUT_SIZE;
    resizeCanvas.height = CAFFE_INPUT_SIZE;
    const rctx = resizeCanvas.getContext("2d");
    rctx.drawImage(faceCanvas, 0, 0, CAFFE_INPUT_SIZE, CAFFE_INPUT_SIZE);

    const imageData = rctx.getImageData(0, 0, CAFFE_INPUT_SIZE, CAFFE_INPUT_SIZE);
    const { data } = imageData;

    // Caffe: BGR, mean subtraction, no /255
    const floatData = new Float32Array(3 * CAFFE_INPUT_SIZE * CAFFE_INPUT_SIZE);
    const pixelCount = CAFFE_INPUT_SIZE * CAFFE_INPUT_SIZE;

    for (let i = 0; i < pixelCount; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];

        floatData[i] = b - CAFFE_MEAN[0];                       // B
        floatData[pixelCount + i] = g - CAFFE_MEAN[1];          // G
        floatData[2 * pixelCount + i] = r - CAFFE_MEAN[2];      // R
    }

    return new ort.Tensor("float32", floatData, [1, 3, CAFFE_INPUT_SIZE, CAFFE_INPUT_SIZE]);
}

// ── Postprocessing ────────────────────────────────────

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - maxLogit));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sumExp);
}

function expectedAge(ageLogits) {
    const probs = softmax(ageLogits);
    let age = 0;
    for (let i = 0; i < NUM_AGE_CLASSES; i++) {
        age += i * probs[i];
    }
    return age;
}

function predictGender(genderLogits) {
    const probs = softmax(genderLogits);
    const idx = probs[0] > probs[1] ? 0 : 1;
    return { label: GENDER_LABELS[idx], confidence: probs[idx] };
}

// ── Inference per method ──────────────────────────────

async function inferMobileNet(faceCanvas) {
    if (!sessions.mobilenet) return null;
    const inputTensor = preprocessMobileNet(faceCanvas);
    const results = await sessions.mobilenet.run({ input: inputTensor });
    const ageLogits = Array.from(results.age_logits.data);
    const genderLogits = Array.from(results.gender_logits.data);

    const age = expectedAge(ageLogits);
    const gender = predictGender(genderLogits);
    return { age: age.toFixed(1), gender: gender.label, confidence: gender.confidence };
}

async function inferCaffe(faceCanvas) {
    if (!sessions.caffe_age || !sessions.caffe_gender) return null;
    const inputTensor = preprocessCaffe(faceCanvas);

    const ageResult = await sessions.caffe_age.run({ data: inputTensor });
    const genderResult = await sessions.caffe_gender.run({ data: inputTensor });

    const ageProbs = Array.from(Object.values(ageResult)[0].data);
    const genderProbs = Array.from(Object.values(genderResult)[0].data);

    const ageIdx = ageProbs.indexOf(Math.max(...ageProbs));
    const genderIdx = genderProbs.indexOf(Math.max(...genderProbs));

    return {
        age: CAFFE_AGE_LABELS[ageIdx],
        gender: GENDER_LABELS[genderIdx],
        confidence: Math.max(...softmax(genderProbs)),
    };
}

async function inferInsightFace(faceCanvas) {
    if (!sessions.insightface) return null;
    // InsightFace uses 112x112 input, similar normalization
    const size = 112;
    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = size;
    resizeCanvas.height = size;
    const rctx = resizeCanvas.getContext("2d");
    rctx.drawImage(faceCanvas, 0, 0, size, size);

    const imageData = rctx.getImageData(0, 0, size, size);
    const { data } = imageData;
    const floatData = new Float32Array(3 * size * size);
    const pixelCount = size * size;

    for (let i = 0; i < pixelCount; i++) {
        floatData[i] = (data[i * 4] / 255.0 - 0.5) / 0.5;
        floatData[pixelCount + i] = (data[i * 4 + 1] / 255.0 - 0.5) / 0.5;
        floatData[2 * pixelCount + i] = (data[i * 4 + 2] / 255.0 - 0.5) / 0.5;
    }

    const inputTensor = new ort.Tensor("float32", floatData, [1, 3, size, size]);
    const results = await sessions.insightface.run({ input: inputTensor });

    // InsightFace outputs age and gender differently per model variant
    const output = Array.from(Object.values(results)[0].data);
    const age = output[2] != null ? output[2] : output[0];
    const genderIdx = output[0] > output[1] ? 0 : 1;
    return {
        age: age.toFixed(1),
        gender: GENDER_LABELS[genderIdx],
        confidence: softmax([output[0], output[1]])[genderIdx],
    };
}

async function infer(faceCanvas) {
    // currentMethod is defined in index.html
    const method = typeof currentMethod !== "undefined" ? currentMethod : "mobilenet";

    switch (method) {
        case "mobilenet": return inferMobileNet(faceCanvas);
        case "caffe": return inferCaffe(faceCanvas);
        case "insightface": return inferInsightFace(faceCanvas);
        default: return inferMobileNet(faceCanvas);
    }
}

// ── Extract face from video ───────────────────────────

function extractFace(box) {
    const { x, y, width, height } = box;
    const faceCanvas = document.createElement("canvas");
    faceCanvas.width = Math.max(1, Math.round(width));
    faceCanvas.height = Math.max(1, Math.round(height));
    const fctx = faceCanvas.getContext("2d");
    fctx.drawImage(
        video,
        Math.round(x), Math.round(y), Math.round(width), Math.round(height),
        0, 0, faceCanvas.width, faceCanvas.height
    );
    return faceCanvas;
}

// ── Detection loop ────────────────────────────────────

async function detectLoop() {
    if (!isRunning) return;

    const t0 = performance.now();

    // Use whichever face detector was loaded
    const faceOptions = faceapi.nets.tinyFaceDetector.isLoaded
        ? new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 })
        : new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });
    const detections = await faceapi.detectAllFaces(video, faceOptions);

    ctx.clearRect(0, 0, overlay.width, overlay.height);
    faceCountEl.textContent = detections.length;

    for (const det of detections) {
        const box = det.box;
        const faceCanvas = extractFace(box);
        if (faceCanvas.width < 10 || faceCanvas.height < 10) continue;

        const result = await infer(faceCanvas);

        // Mirror x-coordinate (video is CSS-flipped, overlay is not)
        const x = overlay.width - Math.round(box.x) - Math.round(box.width);
        const y = Math.round(box.y);
        const w = Math.round(box.width);
        const h = Math.round(box.height);

        ctx.strokeStyle = "#00e676";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        if (result) {
            const label = `${result.gender} (${(result.confidence * 100).toFixed(0)}%), ${result.age}y`;
            ctx.font = "bold 14px sans-serif";
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
            ctx.fillRect(x, y - 22, textWidth + 8, 20);

            ctx.fillStyle = "#00e676";
            ctx.fillText(label, x + 4, y - 7);
        } else {
            ctx.font = "bold 12px sans-serif";
            ctx.fillStyle = "#ff5252";
            ctx.fillText("model not loaded", x + 4, y - 7);
        }
    }

    const elapsed = performance.now() - t0;
    fpsEl.textContent = Math.round(1000 / elapsed);

    animFrameId = requestAnimationFrame(detectLoop);
}

// ── Button handlers ───────────────────────────────────

btnStart.disabled = true;

btnStart.addEventListener("click", async () => {
    try {
        await startWebcam();
        isRunning = true;
        btnStart.disabled = true;
        btnStop.disabled = false;
        statusEl.textContent = "偵測中...";
        detectLoop();
    } catch (err) {
        statusEl.textContent = `攝影機錯誤: ${err.message}`;
    }
});

btnStop.addEventListener("click", () => {
    isRunning = false;
    if (animFrameId) cancelAnimationFrame(animFrameId);
    stopWebcam();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    btnStart.disabled = false;
    btnStop.disabled = true;
    statusEl.textContent = "已停止";
    fpsEl.textContent = "0";
    faceCountEl.textContent = "0";
});

// ── Start ─────────────────────────────────────────────

init();
