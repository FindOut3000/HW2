'use strict';

const MAX_BYTES = 10 * 1024 * 1024;

const el = {
  drop: document.getElementById('drop'),
  fileInput: document.getElementById('fileInput'),
  previewBox: document.getElementById('previewBox'),
  preview: document.getElementById('preview'),
  fileMeta: document.getElementById('fileMeta'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  resetBtn: document.getElementById('resetBtn'),
  error: document.getElementById('error'),
  result: document.getElementById('result'),
  badge: document.getElementById('badge'),
  topBreed: document.getElementById('topBreed'),
  topBreedEn: document.getElementById('topBreedEn'),
  message: document.getElementById('message'),
  bars: document.getElementById('bars'),
  detailMeta: document.getElementById('detailMeta'),
  disclaimer: document.getElementById('disclaimer'),
  barsNote: document.getElementById('barsNote'),
  status: document.getElementById('status'),
  modelInfo: document.getElementById('modelInfo'),
  dropSub: document.getElementById('dropSub'),
};

let selectedFile = null;
let previewUrl = null;
let busy = false;

/* ---------- helpers ---------- */

function showError(msg) {
  el.error.textContent = msg;
  el.error.hidden = false;
  el.result.hidden = true;
}

function clearError() {
  el.error.hidden = true;
  el.error.textContent = '';
}

function humanSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(2) + ' MB';
}

function setMeta(node, rows) {
  node.innerHTML = '';
  for (const [k, v] of rows) {
    const dt = document.createElement('dt');
    dt.textContent = k;
    const dd = document.createElement('dd');
    dd.textContent = v;
    node.append(dt, dd);
  }
}

/* ---------- model status ---------- */

async function loadHealth() {
  try {
    const res = await fetch('/api/health');
    const h = await res.json();
    if (h.ready) {
      el.status.textContent = `모델 준비 완료 · ${h.breed_count}개 라벨 · ${h.device.toUpperCase()} 추론`;
      el.status.className = 'status good';
    } else {
      el.status.textContent = '모델을 사용할 수 없습니다: ' + (h.error || '준비 중');
      el.status.className = 'status bad';
      el.analyzeBtn.disabled = true;
    }
    el.modelInfo.textContent = `품종: ${h.models.breed} · 고양이 판별: ${h.models.gate}`;
    if (h.allowed_formats) {
      el.dropSub.textContent =
        h.allowed_formats.join(' · ') + ` / 최대 ${h.max_upload_mb}MB`;
    }
  } catch (e) {
    el.status.textContent = '서버에 연결할 수 없습니다. 터미널에서 서버가 떠 있는지 확인하세요.';
    el.status.className = 'status bad';
  }
}

/* ---------- file selection + preview ---------- */

function selectFile(file) {
  clearError();
  el.result.hidden = true;

  if (!file) return;

  if (!file.type.startsWith('image/')) {
    showError(`이미지 파일이 아닙니다 (${file.type || '형식 불명'}). JPG, PNG, WEBP 등을 선택해주세요.`);
    return;
  }
  if (file.size === 0) {
    showError('빈 파일입니다. 다른 이미지를 선택해주세요.');
    return;
  }
  if (file.size > MAX_BYTES) {
    showError(`파일이 너무 큽니다 (${humanSize(file.size)}). 최대 10MB까지 업로드할 수 있습니다.`);
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);

  el.preview.onload = () => {
    setMeta(el.fileMeta, [
      ['파일명', file.name],
      ['형식', file.type],
      ['크기', humanSize(file.size)],
      ['해상도', `${el.preview.naturalWidth} × ${el.preview.naturalHeight}`],
    ]);
  };
  el.preview.onerror = () => {
    showError('이미지를 미리 볼 수 없습니다. 파일이 손상되었을 수 있습니다.');
    resetAll();
  };
  el.preview.src = previewUrl;
  el.previewBox.hidden = false;
  el.analyzeBtn.disabled = false;
  el.resetBtn.disabled = false;
}

function resetAll() {
  selectedFile = null;
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  el.preview.removeAttribute('src');
  el.previewBox.hidden = true;
  el.result.hidden = true;
  el.fileInput.value = '';
  el.analyzeBtn.disabled = true;
  el.resetBtn.disabled = true;
  clearError();
}

/* ---------- analyze ---------- */

async function analyze() {
  if (!selectedFile || busy) return;
  busy = true;
  clearError();
  el.result.hidden = true;
  el.analyzeBtn.disabled = true;
  el.analyzeBtn.innerHTML = '<span class="spinner"></span>분석 중…';

  const body = new FormData();
  body.append('file', selectedFile, selectedFile.name);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body });
    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error(`서버가 예상하지 못한 응답을 보냈습니다 (HTTP ${res.status}).`);
    }
    if (!res.ok) throw new Error(data.message || `분석에 실패했습니다 (HTTP ${res.status}).`);
    render(data);
  } catch (err) {
    const offline = err instanceof TypeError;
    showError(offline
      ? '서버에 연결할 수 없습니다. 터미널에서 서버가 실행 중인지 확인하세요.'
      : err.message);
  } finally {
    busy = false;
    el.analyzeBtn.disabled = false;
    el.analyzeBtn.textContent = '분석하기';
  }
}

function render(data) {
  const top = data.predictions[0];

  if (data.status === 'no_cat') {
    el.badge.textContent = '고양이 미검출';
    el.badge.className = 'badge bad';
    el.topBreed.textContent = '판정 불가';
    el.topBreedEn.textContent = '';
  } else if (data.status === 'wild_cat') {
    el.badge.textContent = '집고양이 아님 (추정)';
    el.badge.className = 'badge warn';
    el.topBreed.textContent = '야생 고양이과로 추정';
    el.topBreedEn.textContent = data.cat_detection.gate_top_label;
  } else {
    el.badge.textContent = `추정 · 확신도 ${data.confidence.label}`;
    el.badge.className = 'badge' + (data.confidence.level === 'low' ? ' warn' : '');
    el.topBreed.textContent = (top.breed_ko || top.breed) +
      ` · ${(top.probability * 100).toFixed(1)}%`;
    el.topBreedEn.textContent = top.breed + (top.is_pattern ? ' (품종이 아닌 털색/무늬 라벨)' : '');
  }

  el.message.textContent = data.message;

  // 고양이가 아니거나 야생 고양이과이면 품종 후보는 의미가 없으므로 흐리게 표시한다.
  const unreliable = data.status !== 'ok';
  el.bars.className = 'bars' + (unreliable ? ' unreliable' : '');
  el.barsNote.hidden = !unreliable;
  if (unreliable) {
    el.barsNote.textContent =
      '아래 수치는 입력이 집고양이라고 가정했을 때의 계산 결과일 뿐이며, ' +
      '이 사진에서는 의미가 없습니다. 참고용으로만 표시합니다.';
  }

  el.bars.innerHTML = '';
  for (const p of data.predictions) {
    const pct = (p.probability * 100).toFixed(1);
    const li = document.createElement('li');

    const head = document.createElement('div');
    head.className = 'bar-head';

    const left = document.createElement('span');
    const name = document.createElement('span');
    name.className = 'bar-name';
    name.textContent = p.breed_ko || p.breed;
    left.appendChild(name);

    const en = document.createElement('span');
    en.className = 'bar-en';
    en.textContent = ' ' + p.breed;
    left.appendChild(en);

    if (p.is_pattern) {
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = '무늬/털색';
      left.appendChild(tag);
    }

    const pctEl = document.createElement('span');
    pctEl.className = 'bar-pct';
    pctEl.textContent = pct + '%';

    head.append(left, pctEl);

    const track = document.createElement('div');
    track.className = 'track';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.width = Math.max(1.5, p.probability * 100) + '%';
    track.appendChild(fill);

    li.append(head, track);
    el.bars.appendChild(li);
  }

  setMeta(el.detailMeta, [
    ['집고양이 점수', (data.cat_detection.domestic_cat_score * 100).toFixed(1) + '%'],
    ['야생 고양이과 점수', (data.cat_detection.wild_feline_score * 100).toFixed(1) + '%'],
    ['판별 모델 1순위', data.cat_detection.gate_top_label],
    ['추론 시간', data.elapsed_ms + ' ms'],
    ['입력 해상도', `${data.image.width} × ${data.image.height}`],
    ['품종 모델', data.model.breed],
    ['모델 참고', data.model.breed_note],
    ['연산 장치', data.model.device.toUpperCase()],
    // 어느 배포에서 나온 결과인지 화면에서 바로 확인할 수 있게 한다.
    ['전처리 모드', data.model.preprocess
      ? `${data.model.preprocess} (게이트 ${data.model.views.gate}뷰 · 품종 ${data.model.views.breed}뷰)`
      : 'legacy (1뷰)'],
    ['배포 버전', data.version ? data.version.build : '-'],
  ]);

  el.disclaimer.textContent = data.disclaimer;
  el.result.hidden = false;
  el.result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ---------- events ---------- */

el.drop.addEventListener('click', () => el.fileInput.click());
el.drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.fileInput.click(); }
});
el.fileInput.addEventListener('change', (e) => selectFile(e.target.files[0]));

['dragenter', 'dragover'].forEach((t) =>
  el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) =>
  el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove('over'); }));
el.drop.addEventListener('drop', (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) selectFile(f);
});

el.analyzeBtn.addEventListener('click', analyze);
el.resetBtn.addEventListener('click', resetAll);

loadHealth();
