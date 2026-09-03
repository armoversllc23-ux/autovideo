// AutoVideo frontend — three screens, three API calls, no timelines, no
// settings panels. All "creative" decisions happen server-side (see
// ARCHITECTURE.md); this file only orchestrates which screen is visible
// and polls job status.

const API_BASE = ""; // same-origin: FastAPI serves this file too.

const screens = {
  describe: document.getElementById("screen-describe"),
  progress: document.getElementById("screen-progress"),
  result: document.getElementById("screen-result"),
};

function showScreen(name) {
  Object.values(screens).forEach((el) => el.classList.remove("screen--active"));
  screens[name].classList.add("screen--active");
}

// ---- Screen 1: Describe ---------------------------------------------------

const descriptionEl = document.getElementById("description");
const addMediaBtn = document.getElementById("add-media-btn");
const mediaInput = document.getElementById("media-input");
const mediaCountEl = document.getElementById("media-count");
const createBtn = document.getElementById("create-btn");
const describeError = document.getElementById("describe-error");

addMediaBtn.addEventListener("click", () => mediaInput.click());

mediaInput.addEventListener("change", () => {
  const n = mediaInput.files.length;
  mediaCountEl.textContent = n > 0 ? `${n} file${n > 1 ? "s" : ""} selected` : "";
});

// ---- Shape (platform) picker ----------------------------------------------
// Entirely optional: if the person never taps a shape, we send no `platform`
// field at all and the backend infers one from the description text (see
// `_apply_platform_override` in main.py, which only overrides when a value
// is actually provided). Tapping the same shape again deselects it, going
// back to "let AutoVideo decide".

const platformButtons = Array.from(document.querySelectorAll(".shape-btn"));
let selectedPlatform = null;

platformButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const value = btn.dataset.platform;
    selectedPlatform = selectedPlatform === value ? null : value;
    platformButtons.forEach((b) => {
      const isSelected = b.dataset.platform === selectedPlatform;
      b.classList.toggle("shape-btn--selected", isSelected);
      b.setAttribute("aria-pressed", String(isSelected));
    });
  });
});

const voiceToggle = document.getElementById("voice-toggle");

createBtn.addEventListener("click", async () => {
  const description = descriptionEl.value.trim();
  describeError.hidden = true;

  if (!description) {
    describeError.textContent = "Please describe the video you want to create.";
    describeError.hidden = false;
    return;
  }

  createBtn.disabled = true;
  createBtn.textContent = "Starting…";

  try {
    const form = new FormData();
    form.append("description", description);
    if (selectedPlatform) {
      form.append("platform", selectedPlatform);
    }
    // Narration defaults on server-side too; we only need to say something
    // when the person has switched it off.
    form.append("voice", voiceToggle.checked ? "on" : "off");
    for (const file of mediaInput.files) {
      form.append("media", file);
    }

    const res = await fetch(`${API_BASE}/api/videos`, { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || "Could not start your video.");
    }
    const { job_id } = await res.json();
    showScreen("progress");
    pollStatus(job_id);
  } catch (err) {
    describeError.textContent = err.message || "Something went wrong. Please try again.";
    describeError.hidden = false;
  } finally {
    createBtn.disabled = false;
    createBtn.textContent = "Create video";
  }
});

// ---- Screen 2: Progress ---------------------------------------------------

const progressMessageEl = document.getElementById("progress-message");
const progressFillEl = document.getElementById("progress-fill");

// Friendly copy per backend status, in the order the pipeline runs. Width
// gives the user a sense of motion even though we don't know exact timing.
const STAGE_ORDER = ["queued", "parsing", "storyboarding", "selecting_media", "rendering", "done"];
// "Recording narration..." is sent as a `rendering`-status progress message
// (see renderer.py's `_notify("Recording narration...")`), so it's handled
// by `renderingWidthPercent` below rather than needing its own stage here.
const STAGE_WIDTH = {
  queued: "8%",
  parsing: "20%",
  storyboarding: "45%",
  selecting_media: "68%",
  rendering: "68%", // fallback only — see renderingWidthPercent for the real, per-scene value
  done: "100%",
};

// The "rendering" status covers everything from the first scene clip to the
// final muxed file, and the backend now sends per-scene progress messages
// ("Rendering scene 2 of 4...", "Combining scenes...", "Adding music...")
// instead of one flat message for the whole stage. It may also send
// "Recording narration..." right after the last scene (see renderer.py's
// `_build_narration_track`), when spoken narration is on. This maps those
// messages onto a width within [RENDERING_START, RENDERING_END] so the bar
// keeps moving smoothly (and never backward) through what's usually the
// longest stage, rather than sitting at a fixed 90% for the whole render.
const RENDERING_START = 68;
const RENDERING_END = 96;

function renderingWidthPercent(message) {
  const span = RENDERING_END - RENDERING_START;
  const sceneMatch = /Rendering scene (\d+) of (\d+)/.exec(message || "");
  if (sceneMatch) {
    const current = Number(sceneMatch[1]);
    const total = Number(sceneMatch[2]) || 1;
    // Per-scene rendering is the bulk of the work; leave the tail of the
    // range for the narration/combine/music steps that follow the last scene.
    const sceneShare = 0.7;
    const fraction = Math.min(1, (current - 1) / total) * sceneShare;
    return RENDERING_START + fraction * span;
  }
  if (/Recording narration/i.test(message || "")) return RENDERING_START + span * 0.78;
  if (/Combining scenes/i.test(message || "")) return RENDERING_START + span * 0.85;
  if (/Adding music/i.test(message || "")) return RENDERING_START + span * 0.95;
  return RENDERING_START;
}

let pollTimer = null;

function pollStatus(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(() => checkStatus(jobId), 1200);
  checkStatus(jobId); // immediate first check
}

async function checkStatus(jobId) {
  try {
    const res = await fetch(`${API_BASE}/api/videos/${jobId}`);
    if (!res.ok) throw new Error("Lost track of your video.");
    const status = await res.json();

    progressMessageEl.textContent = status.progress_message;
    if (status.status === "rendering") {
      progressFillEl.style.width = `${renderingWidthPercent(status.progress_message)}%`;
    } else {
      progressFillEl.style.width = STAGE_WIDTH[status.status] || "50%";
    }

    if (status.status === "done") {
      clearInterval(pollTimer);
      showResult(jobId);
    } else if (status.status === "failed") {
      clearInterval(pollTimer);
      showScreen("describe");
      describeError.textContent = status.error || "We couldn't create your video. Please try again.";
      describeError.hidden = false;
    }
  } catch (err) {
    clearInterval(pollTimer);
    showScreen("describe");
    describeError.textContent = "Lost connection while creating your video. Please try again.";
    describeError.hidden = false;
  }
}

// ---- Screen 3: Result ------------------------------------------------------

const resultVideo = document.getElementById("result-video");
const downloadBtn = document.getElementById("download-btn");
const shareBtn = document.getElementById("share-btn");
const variantBtn = document.getElementById("variant-btn");
const startOverBtn = document.getElementById("start-over-btn");
const resultError = document.getElementById("result-error");

let currentJobId = null;

function showResult(jobId) {
  currentJobId = jobId;
  const fileUrl = `${API_BASE}/api/videos/${jobId}/file?t=${Date.now()}`;
  resultVideo.src = fileUrl;
  downloadBtn.href = fileUrl;
  resultError.hidden = true;
  showScreen("result");
}

shareBtn.addEventListener("click", async () => {
  const fileUrl = `${window.location.origin}/api/videos/${currentJobId}/file`;
  if (navigator.share) {
    try {
      await navigator.share({ title: "My video", url: fileUrl });
    } catch (_) {
      /* user cancelled share sheet — no action needed */
    }
  } else {
    await navigator.clipboard.writeText(fileUrl);
    shareBtn.textContent = "Link copied!";
    setTimeout(() => (shareBtn.textContent = "Share"), 1500);
  }
});

variantBtn.addEventListener("click", async () => {
  variantBtn.disabled = true;
  resultError.hidden = true;
  try {
    const res = await fetch(`${API_BASE}/api/videos/${currentJobId}/variant`, { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || "Could not create a new version.");
    }
    showScreen("progress");
    pollStatus(currentJobId);
  } catch (err) {
    resultError.textContent = err.message || "Something went wrong.";
    resultError.hidden = false;
  } finally {
    variantBtn.disabled = false;
  }
});

startOverBtn.addEventListener("click", () => {
  descriptionEl.value = "";
  mediaInput.value = "";
  mediaCountEl.textContent = "";
  describeError.hidden = true;
  selectedPlatform = null;
  platformButtons.forEach((b) => {
    b.classList.remove("shape-btn--selected");
    b.setAttribute("aria-pressed", "false");
  });
  voiceToggle.checked = true;
  showScreen("describe");
});
