const form = document.getElementById("generateForm");
const topicInput = document.getElementById("topicInput");
const generateBtn = document.getElementById("generateBtn");
const generateBtnText = document.getElementById("generateBtnText");
const progressPanel = document.getElementById("progressPanel");
const progressLabel = document.getElementById("progressLabel");
const progressPercent = document.getElementById("progressPercent");
const progressBar = document.getElementById("progressBar");
const stageList = document.getElementById("stageList");
const videoPlayer = document.getElementById("videoPlayer");
const emptyPreview = document.getElementById("emptyPreview");
const previewTitle = document.getElementById("previewTitle");
const previewMeta = document.getElementById("previewMeta");
const downloadBtn = document.getElementById("downloadBtn");
const refreshBtn = document.getElementById("refreshBtn");
const historyGrid = document.getElementById("historyGrid");
const emptyHistory = document.getElementById("emptyHistory");
const historyCount = document.getElementById("historyCount");
const toastViewport = document.getElementById("toastViewport");

const stages = [
    { label: "Generating script", ceiling: 24 },
    { label: "Generating voice", ceiling: 48 },
    { label: "Generating subtitles", ceiling: 72 },
    { label: "Rendering video", ceiling: 92 }
];

const state = {
    videos: [],
    progress: 0,
    activeStage: 0,
    progressTimer: null,
    isGenerating: false
};

form.addEventListener("submit", handleGenerate);
refreshBtn.addEventListener("click", () => loadVideos());

loadVideos({ silent: true });

async function handleGenerate(event) {
    event.preventDefault();

    const topic = topicInput.value.trim();
    if (!topic) {
        showToast("Add a topic before generating.", "error");
        topicInput.focus();
        return;
    }

    setGenerating(true);
    startProgress();

    try {
        const response = await fetch("/generate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ topic })
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            const step = result.step ? `${result.step}: ` : "";
            throw new Error(step + (result.error || result.detail || "Generation failed."));
        }

        finishProgress();

        const generatedVideo = {
            filename: result.filename,
            topic: result.topic,
            created_at: new Date().toISOString(),
            url: result.video_url,
            download_url: result.download_url
        };

        setPreview(generatedVideo, true);

        if (Array.isArray(result.videos)) {
            state.videos = result.videos;
            renderHistory();
        } else {
            await loadVideos({ silent: true });
        }

        showToast("Video generated successfully.", "success");
    } catch (error) {
        failProgress();
        showToast(error.message || "Generation failed.", "error");
    } finally {
        window.setTimeout(() => {
            if (!state.isGenerating) {
                return;
            }
            setGenerating(false);
        }, 700);
    }
}

async function loadVideos(options = {}) {
    const silent = Boolean(options.silent);

    try {
        const response = await fetch("/videos", { cache: "no-store" });
        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.detail || result.error || "Could not load videos.");
        }

        state.videos = Array.isArray(result.videos) ? result.videos : [];
        renderHistory();

        if (state.videos.length > 0 && !videoPlayer.src) {
            setPreview(state.videos[0], false);
        }

        if (!silent) {
            showToast("History refreshed.", "success");
        }
    } catch (error) {
        if (!silent) {
            showToast(error.message || "Could not load videos.", "error");
        }
    }
}

function setGenerating(isGenerating) {
    state.isGenerating = isGenerating;
    generateBtn.disabled = isGenerating;
    generateBtnText.textContent = isGenerating ? "Generating..." : "Generate Video";

    if (!isGenerating) {
        window.clearInterval(state.progressTimer);
        state.progressTimer = null;
    }
}

function startProgress() {
    window.clearInterval(state.progressTimer);
    progressPanel.hidden = false;
    state.progress = 3;
    state.activeStage = 0;
    renderProgress();

    state.progressTimer = window.setInterval(() => {
        const active = stages[state.activeStage];
        const increment = state.activeStage === stages.length - 1
            ? Math.random() * 1.2 + 0.35
            : Math.random() * 3.5 + 1.2;

        state.progress = Math.min(state.progress + increment, active.ceiling);

        if (state.progress >= active.ceiling - 0.4 && state.activeStage < stages.length - 1) {
            state.activeStage += 1;
            state.progress = Math.max(state.progress, stages[state.activeStage - 1].ceiling + 1);
        }

        renderProgress();
    }, 850);
}

function finishProgress() {
    window.clearInterval(state.progressTimer);
    state.progressTimer = null;
    state.progress = 100;
    state.activeStage = stages.length - 1;
    renderProgress("Video ready");
}

function failProgress() {
    window.clearInterval(state.progressTimer);
    state.progressTimer = null;
    progressLabel.textContent = "Generation failed";
    setGenerating(false);
}

function renderProgress(labelOverride) {
    const rounded = Math.round(state.progress);
    progressBar.style.width = `${rounded}%`;
    progressPercent.textContent = `${rounded}%`;
    progressLabel.textContent = labelOverride || stages[state.activeStage].label;

    Array.from(stageList.children).forEach((item, index) => {
        item.classList.toggle("is-active", index === state.activeStage && state.progress < 100);
        item.classList.toggle("is-complete", index < state.activeStage || state.progress === 100);
    });
}

function setPreview(video, autoplay) {
    if (!video || !video.url) {
        return;
    }

    videoPlayer.pause();
    videoPlayer.src = video.url;
    videoPlayer.load();
    videoPlayer.classList.add("is-visible");
    emptyPreview.classList.add("is-hidden");

    previewTitle.textContent = video.topic || "Generated video";
    previewMeta.textContent = formatMeta(video);

    downloadBtn.href = video.download_url || "#";
    downloadBtn.setAttribute("download", video.filename || "video.mp4");
    downloadBtn.setAttribute("aria-disabled", "false");
    downloadBtn.classList.remove("is-disabled");

    if (autoplay) {
        videoPlayer.play().catch(() => {
            videoPlayer.controls = true;
        });
    }
}

function renderHistory() {
    historyGrid.replaceChildren();
    historyCount.textContent = `${state.videos.length} ${state.videos.length === 1 ? "video" : "videos"}`;
    emptyHistory.classList.toggle("is-visible", state.videos.length === 0);

    state.videos.forEach((video) => {
        const card = document.createElement("article");
        card.className = "video-card";

        const thumb = document.createElement("div");
        thumb.className = "video-thumb";

        const thumbVideo = document.createElement("video");
        thumbVideo.src = video.url;
        thumbVideo.muted = true;
        thumbVideo.playsInline = true;
        thumbVideo.preload = "metadata";
        thumb.appendChild(thumbVideo);

        const title = document.createElement("p");
        title.className = "card-title";
        title.textContent = video.topic || "Generated video";

        const meta = document.createElement("p");
        meta.className = "card-meta";
        meta.textContent = formatMeta(video);

        const actions = document.createElement("div");
        actions.className = "card-actions";

        const previewButton = document.createElement("button");
        previewButton.type = "button";
        previewButton.className = "small-button";
        previewButton.textContent = "Preview";
        previewButton.addEventListener("click", () => setPreview(video, true));

        const downloadLink = document.createElement("a");
        downloadLink.className = "small-button";
        downloadLink.href = video.download_url;
        downloadLink.download = video.filename || "video.mp4";
        downloadLink.textContent = "Download";

        actions.append(previewButton, downloadLink);
        card.append(thumb, title, meta, actions);
        historyGrid.appendChild(card);
    });
}

function formatMeta(video) {
    const dateLabel = formatDate(video.created_at);
    const sizeLabel = typeof video.size_mb === "number" ? `${video.size_mb} MB` : "";
    return [dateLabel, sizeLabel].filter(Boolean).join(" / ");
}

function formatDate(value) {
    if (!value) {
        return "Just now";
    }

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return "Just now";
    }

    return new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit"
    }).format(date);
}

function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast is-${type}`;
    toast.textContent = message;
    toastViewport.appendChild(toast);

    window.setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(8px)";
    }, 4200);

    window.setTimeout(() => {
        toast.remove();
    }, 4600);
}
