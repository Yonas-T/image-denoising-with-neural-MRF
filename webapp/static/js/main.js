/**
 * Neural MRF Image Denoiser — Frontend Logic
 *
 * Handles file upload (drag-and-drop + click), noise configuration,
 * denoising request, result display, metric highlighting, and
 * image download.
 */

// ── DOM Elements ─────────────────────────────────────
const uploadZone     = document.getElementById('upload-zone');
const fileInput      = document.getElementById('file-input');
const preview        = document.getElementById('upload-preview');
const filenameLbl    = document.getElementById('upload-filename');
const noiseType      = document.getElementById('noise-type');
const noiseSlider    = document.getElementById('noise-level');
const sliderLabel    = document.getElementById('slider-label');
const sliderValue    = document.getElementById('slider-value');
const denoiseBtn     = document.getElementById('denoise-btn');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');

// Image elements
const imgOriginal = document.getElementById('img-original');
const imgNoisy    = document.getElementById('img-noisy');
const imgNmrf     = document.getElementById('img-nmrf');
const imgBaseline = document.getElementById('img-baseline');
const imgPotts = document.getElementById('img-potts');

// Download buttons
const dlOriginal = document.getElementById('dl-original');
const dlNoisy    = document.getElementById('dl-noisy');
const dlNmrf     = document.getElementById('dl-nmrf');
const dlBaseline = document.getElementById('dl-baseline');
const dlPotts = document.getElementById('dl-potts');

// State
let uploadedFile = null;

// ── Upload: Drag & Drop ─────────────────────────────
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragover');
});
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) handleFile(files[0]);
});
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Please upload an image file (PNG, JPEG, WebP).');
    return;
  }
  uploadedFile = file;

  // Preview
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.classList.add('visible');
    filenameLbl.textContent = file.name;
    filenameLbl.classList.add('visible');
  };
  reader.readAsDataURL(file);

  denoiseBtn.disabled = false;
}

// ── Noise Controls ───────────────────────────────────
function updateSliderUI() {
  const type = noiseType.value;
  if (type === 'gaussian') {
    sliderLabel.textContent = 'Sigma (σ)';
    noiseSlider.min = 5;
    noiseSlider.max = 100;
    noiseSlider.value = 25;
    sliderValue.textContent = 'σ = 25';
  } else {
    sliderLabel.textContent = 'Lambda (λ)';
    noiseSlider.min = 5;
    noiseSlider.max = 100;
    noiseSlider.value = 30;
    sliderValue.textContent = 'λ = 30';
  }
}

noiseType.addEventListener('change', updateSliderUI);
noiseSlider.addEventListener('input', () => {
  const prefix = noiseType.value === 'gaussian' ? 'σ' : 'λ';
  sliderValue.textContent = `${prefix} = ${noiseSlider.value}`;
});

// ── Denoise ──────────────────────────────────────────
denoiseBtn.addEventListener('click', denoise);

async function denoise() {
  if (!uploadedFile) {
    showToast('Please upload an image first.');
    return;
  }

  // Show loading
  loadingOverlay.classList.add('visible');
  denoiseBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('image', uploadedFile);
    formData.append('noise_type', noiseType.value);
    formData.append('noise_level', noiseSlider.value);

    const resp = await fetch('/denoise', {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || `Server error: ${resp.status}`);
    }

    const data = await resp.json();
    displayResults(data);
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    loadingOverlay.classList.remove('visible');
    denoiseBtn.disabled = false;
  }
}

// ── Display Results ──────────────────────────────────
function displayResults(data) {
  // Set images
  imgOriginal.src = 'data:image/png;base64,' + data.original;
  imgNoisy.src    = 'data:image/png;base64,' + data.noisy;
  imgNmrf.src     = 'data:image/png;base64,' + data.nmrf_denoised;
  imgPotts.src    = 'data:image/png;base64,' + data.potts_denoised;
  imgBaseline.src = 'data:image/png;base64,' + data.baseline_denoised;

  // Fill metrics
  const m = data.metrics;
  setMetric('m-nmrf-psnr',     m.nmrf.psnr,     'psnr');
  setMetric('m-nmrf-ssim',     m.nmrf.ssim,     'ssim');
  setMetric('m-nmrf-mae',      m.nmrf.mae,      'mae');
  setMetric('m-baseline-psnr', m.baseline.psnr, 'psnr');
  setMetric('m-baseline-ssim', m.baseline.ssim, 'ssim');
  setMetric('m-baseline-mae',  m.baseline.mae,  'mae');
  setMetric('m-potts-psnr', m.potts.psnr, 'psnr'); 
  setMetric('m-potts-ssim', m.potts.ssim, 'ssim'); 
  setMetric('m-potts-mae',  m.potts.mae,  'mae'); 

  // Highlight best between NMRF, Potts, and baseline
  highlightBest('psnr', m.nmrf.psnr, m.potts.psnr, m.baseline.psnr, true);   // higher is better
  highlightBest('ssim', m.nmrf.ssim, m.potts.ssim, m.baseline.ssim, true);
  highlightBest('mae',  m.nmrf.mae, m.potts.mae, m.baseline.mae,  false);  // lower is better


  // Setup download buttons
  setupDownload(dlOriginal, data.original,          'original.png');
  setupDownload(dlNoisy,    data.noisy,             'noisy.png');
  setupDownload(dlNmrf,     data.nmrf_denoised,     'nmrf_denoised.png');
  setupDownload(dlBaseline, data.baseline_denoised, 'baseline_denoised.png');
  setupDownload(dlPotts,    data.potts_denoised,    'potts_denoised.png'); 

  // Show results
  resultsSection.classList.add('visible');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setMetric(id, value, type) {
  const el = document.getElementById(id);
  if (type === 'psnr') {
    el.textContent = value.toFixed(2) + ' dB';
  } else if (type === 'ssim') {
    el.textContent = value.toFixed(4);
  } else {
    el.textContent = value.toFixed(4);
  }
  el.classList.remove('metric-best');
}

function highlightBest(metric, nmrfVal, pottsVal, baselineVal, higherIsBetter) {
  const nmrfEl     = document.getElementById('m-nmrf-' + metric);
  const pottsEl    = document.getElementById('m-potts-' + metric);
  const baselineEl = document.getElementById('m-baseline-' + metric);

  nmrfEl.classList.remove('metric-best');
  pottsEl.classList.remove('metric-best');
  baselineEl.classList.remove('metric-best');

  let bestVal = higherIsBetter 
    ? Math.max(nmrfVal, pottsVal, baselineVal) 
    : Math.min(nmrfVal, pottsVal, baselineVal);

  if (nmrfVal === bestVal) nmrfEl.classList.add('metric-best');
  else if (pottsVal === bestVal) pottsEl.classList.add('metric-best');
  else if (baselineVal === bestVal) baselineEl.classList.add('metric-best');
}

function setupDownload(btn, base64Data, filename) {
  btn.onclick = () => {
    const link = document.createElement('a');
    link.href = 'data:image/png;base64,' + base64Data;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
}

// ── Toast Notifications ──────────────────────────────
function showToast(message) {
  // Remove existing toasts
  document.querySelectorAll('.toast').forEach(t => t.remove());

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => {
    if (toast.parentNode) toast.remove();
  }, 5000);
}

// ── Init ─────────────────────────────────────────────
denoiseBtn.disabled = true;
updateSliderUI();
