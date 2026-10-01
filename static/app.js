let uploadedVideoPath = null;
let fpsChart = null;
let confChart = null;

// ---- Tab switching ----
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "history") loadHistory();
    if (btn.dataset.tab === "versions") loadVersions();
  });
});

// ---- Upload ----
async function uploadVideo() {
  const fileInput = document.getElementById("videoFile");
  if (!fileInput.files.length) {
    alert("Choose a video first.");
    return;
  }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  document.getElementById("uploadStatus").innerText = "Uploading...";
  const res = await fetch("/api/upload", { method: "POST", body: formData });
  const data = await res.json();
  uploadedVideoPath = data.video_path;
  document.getElementById("uploadStatus").innerText = `Uploaded: ${uploadedVideoPath}`;
  document.getElementById("videoPathDisplay").innerText = `Video ready: ${uploadedVideoPath}`;
}

// ---- Comparison run ----
async function startComparison() {
  if (!uploadedVideoPath) {
    alert("Upload a video first (Dashboard tab).");
    return;
  }
  const checked = Array.from(document.querySelectorAll(".checkboxes input:checked")).map(el => el.value);
  if (!checked.length) {
    alert("Pick at least one algorithm.");
    return;
  }
  const epochs = document.getElementById("epochs").value;
  const maxFrames = document.getElementById("maxFrames").value;

  const formData = new FormData();
  formData.append("video_path", uploadedVideoPath);
  formData.append("algorithms", checked.join(","));
  formData.append("epochs", epochs);
  formData.append("max_frames", maxFrames);
  formData.append("device", "cpu");

  const res = await fetch("/api/compare/start", { method: "POST", body: formData });
  const data = await res.json();
  document.getElementById("runStatus").innerText = `Job started: ${data.job_id}`;

  pollStatus(data.job_id);
}

async function pollStatus(jobId) {
  const statusEl = document.getElementById("runStatus");
  const interval = setInterval(async () => {
    const res = await fetch(`/api/compare/status/${jobId}`);
    const data = await res.json();

    if (data.status === "running" || data.status === "queued") {
      statusEl.innerText = `Status: ${data.status}...`;
    } else if (data.status === "done") {
      clearInterval(interval);
      statusEl.innerText = "Done!";
      renderResults(data.result);
      renderLatestOnDashboard(data.result);
    } else if (data.status === "error") {
      clearInterval(interval);
      statusEl.innerText = `Error: ${data.error}`;
    }
  }, 2000);
}

function buildRows(algorithms) {
  let rows = "";
  const labels = [], fpsData = [], confData = [];
  for (const [name, metrics] of Object.entries(algorithms)) {
    if (metrics.error) {
      rows += `<tr><td>${name}</td><td colspan="4">Error: ${metrics.error}</td></tr>`;
      continue;
    }
    rows += `<tr>
      <td>${name}</td>
      <td>${metrics.avg_fps}</td>
      <td>${metrics.avg_inference_ms}</td>
      <td>${metrics.avg_detections_per_frame}</td>
      <td>${metrics.avg_confidence}</td>
    </tr>`;
    labels.push(name);
    fpsData.push(metrics.avg_fps);
    confData.push(metrics.avg_confidence);
  }
  return { rows, labels, fpsData, confData };
}

function renderResults(result) {
  const { rows, labels, fpsData, confData } = buildRows(result.algorithms);
  document.querySelector("#resultsTable tbody").innerHTML = rows;
  drawChart("fpsChart", fpsChart, labels, fpsData, "Avg FPS (higher = faster)", (c) => fpsChart = c);
  drawChart("confChart", confChart, labels, confData, "Avg Confidence", (c) => confChart = c);
}

function renderLatestOnDashboard(result) {
  const { rows } = buildRows(result.algorithms);
  document.querySelector("#latestTable tbody").innerHTML = rows;
  document.getElementById("statLastRun").innerText = result.run_id;
}

function drawChart(canvasId, existingChart, labels, data, title, setter) {
  if (existingChart) existingChart.destroy();
  const ctx = document.getElementById(canvasId).getContext("2d");
  const chart = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ label: title, data, backgroundColor: "#2563eb" }] },
    options: { responsive: true, plugins: { title: { display: true, text: title } } },
  });
  setter(chart);
}

// ---- History ----
async function loadHistory() {
  const res = await fetch("/api/compare/history");
  const runs = await res.json();

  document.getElementById("statRuns").innerText = runs.length;
  if (runs.length) document.getElementById("statLastRun").innerText = runs[0].run_id;

  const el = document.getElementById("historyList");
  el.innerHTML = runs.map(r => `
    <div style="border-bottom:1px solid #eee; padding:8px 0;">
      <b>${r.run_id}</b> — ${r.video} — ${r.started_at}<br>
      Algorithms: ${Object.keys(r.algorithms).join(", ")}
    </div>
  `).join("") || "<p>No runs yet.</p>";
}

// ---- Versions ----
async function loadVersions() {
  const res = await fetch("/api/versions");
  const data = await res.json();

  const envRows = Object.entries(data)
    .filter(([k]) => k !== "algorithms")
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
    .join("");
  document.querySelector("#envTable tbody").innerHTML = envRows;

  const algoRows = Object.entries(data.algorithms || {})
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
    .join("");
  document.querySelector("#algoVersionTable tbody").innerHTML = algoRows;
}

// ---- Explain with AI ----
function getTabContext(tab) {
  const tableId = { dashboard: "latestTable", comparison: "resultsTable",
                     history: "historyList", about: null, versions: "envTable" }[tab];
  if (!tableId) return document.getElementById(tab)?.innerText?.slice(0, 2000) || "";
  const el = document.getElementById(tableId);
  return el ? el.innerText.slice(0, 2000) : "";
}

async function explainTab(tab) {
  const box = document.getElementById(`explain-${tab}`);
  box.innerText = "Thinking...";

  const context = getTabContext(tab);
  const formData = new FormData();
  formData.append("tab", tab);
  formData.append("context", context);

  try {
    const res = await fetch("/api/explain", { method: "POST", body: formData });
    const data = await res.json();
    const prefix = data.agent ? `[${data.agent}]\n\n` : "";
    box.innerText = data.error ? data.error : prefix + (data.explanation || "No response.");
  } catch (e) {
    box.innerText = "Request failed: " + e;
  }
}

// Load history + stats on first page load
loadHistory();
