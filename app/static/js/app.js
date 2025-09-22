// Stimulus AI — Adapter Card JS (P1 refactor)
// NOTE: Maintains existing conventions, IDs, and function names (minus Live toggle usage)

window.addEventListener('DOMContentLoaded', () => {
  const intro = document.getElementById('intro');
  const app   = document.getElementById('app');
  setTimeout(() => { intro.style.display = 'none'; app.classList.add('ready'); }, 5000);
});

async function pingHealth(){
  const badge = document.getElementById('health');
  try {
    const r = await fetch('/health');
    badge.className = r.ok ? 'badge ok' : 'badge';
    badge.querySelector('span:nth-child(2)').textContent = r.ok ? 'healthy' : 'checking…';
  } catch {
    badge.className = 'badge';
    badge.querySelector('span:nth-child(2)').textContent = 'checking…';
  }
}
pingHealth(); setInterval(pingHealth, 8000);

const $ = (id) => document.getElementById(id);
const STATE = { schema: null, claims: [], resume_path: "", transcript_path: "", audio_meta: null, video_path: "", video_meta: null };

// --- Processing overlay / button gating ---
const OVERLAY_MIN_MS = 8000; // two loops of 4s gif
function disableControls(disabled){
  const ids = ["compileBtn","clearBtn","loadBtn","resumeBtn","audioBtn","videoBtn","evidenceBtn","cand","candSelect"];
  ids.forEach(id => { const el = $(id); if (el) el.disabled = !!disabled; });
}
function showProcessing(label){
  disableControls(true);
  const res = $('result'); if (res) res.style.opacity = '0.12';
  $('processingLabel').textContent = label || 'Processing…';
  $('processingOverlay').style.display = 'flex';
}
function hideProcessing(){
  $('processingOverlay').style.display = 'none';
  const res = $('result'); if (res) res.style.opacity = '';
  disableControls(false);
}
async function gateWithSpinner(promise, label){
  showProcessing(label);
  const minWait = new Promise(r => setTimeout(r, OVERLAY_MIN_MS));
  try{
    const [result] = await Promise.all([promise, minWait]);
    return result;
  } finally { hideProcessing(); }
}

function esc(s){
  return (s ?? '').toString().replace(/[&<>"']/g, (m) => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[m]));
}
const setMsg = (t) => { $('artifactsMsg').textContent = t; };
const setMsgHTML = (h) => { $('artifactsMsg').innerHTML = h; };

const postForm = async (url, fd) => {
  const r = await fetch(url, { method:'POST', body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};
const humanSizeMB = (m) => (m == null ? '' : `${m.toFixed(1)} MB`);
const mmss = (s) => {
  s = Math.max(0, Math.floor(+s||0));
  const m = Math.floor(s/60), ss = s%60;
  return `${m}:${ss.toString().padStart(2,'0')}`;
};

// Badges
function renderVideoBadge(vpath, meta){
  if (!vpath || !meta) return '';
  return `<span class="badge-mini" title="video ✓ ${mmss(meta.duration_sec)} · ${meta.codec} · ${humanSizeMB(meta.filesize_mb)}">
    <video src="${vpath}"></video>
    <span>video ✓ ${mmss(meta.duration_sec)} · ${meta.codec} · ${humanSizeMB(meta.filesize_mb)}</span>
  </span>`;
}
function renderAudioBadge(meta){
  if (!meta) return '';
  return `<span class="badge-mini" title="audio ✓ ${mmss(meta.duration_sec)} · ${meta.codec} · ${humanSizeMB(meta.filesize_mb)}">
    <span style="display:inline-flex;align-items:center;gap:6px;">
      <!-- Headphones icon (unchanged) -->
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M12 3a9 9 0 0 0-9 9v5a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H5a7 7 0 0 1 14 0h-1a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-5a9 9 0 0 0-9-9z"></path>
      </svg>
      <span>audio ✓ ${mmss(meta.duration_sec)} · ${meta.codec} · ${humanSizeMB(meta.filesize_mb)}</span>
    </span>
  </span>`;
}
function renderResumeBadge(path){
  if (!path) return '';
  return `<span class="badge-mini" title="resume attached">resume ✓</span>`;
}

function enforceButtons(latest){
  const hasAudio = (!!latest.audio_meta || (!!latest.transcript_chars && !latest.video_path));
  const hasVideo = !!latest.video_path;

  $('audioBtn').title = hasVideo
    ? "Attach audio (will offer to replace existing video)"
    : "Attach audio";

  $('videoBtn').title = hasAudio
    ? "Attach video (will offer to replace existing audio)"
    : "Attach video";
}

// --- Candidates API (preserve placeholder; do NOT auto-select first candidate) ---
async function refreshCandidates(){
  try {
    const r = await fetch('/artifacts/candidates');
    if (!r.ok) return;
    const { candidates } = await r.json();
    const sel = $('candSelect');
    const current = sel.value;

    // Keep placeholder at index 0
    const placeholder = '<option value="">— pick candidate —</option>';
    sel.innerHTML = placeholder;

    for (const c of (candidates || [])) {
      const opt = document.createElement('option');
      opt.value = c; opt.textContent = c;
      sel.appendChild(opt);
    }
    if (current && (candidates || []).includes(current)) sel.value = current;
  } catch {}
}
refreshCandidates();

// Keep cand & select in sync without forcing a default
$('cand').addEventListener('input', () => {
  const v = ($('cand').value || '').trim();
  const sel = $('candSelect');
  const match = [...sel.options].find(o => o.value === v);
  sel.value = match ? v : '';
});
$('candSelect').addEventListener('change', () => {
  const v = $('candSelect').value;
  if (v) $('cand').value = v;
});

// Load existing artifacts (explicit candidate)
$('loadBtn').addEventListener('click', async () => {
  const cand = ($('candSelect').value || $('cand').value || '').trim();
  if (!cand) { alert('Pick or type a candidate first.'); return; }
  setMsg('Checking artifacts…');

  try {
    const r = await gateWithSpinner(fetch(`/artifacts/for/${encodeURIComponent(cand)}`), 'Artifacts');
    if (!r.ok) { setMsg('Load failed'); return; }
    const data = await r.json();
    const latest = data.latest || {};
    if (!latest) { setMsg('No artifacts found'); return; }

    STATE.resume_path = latest.resume_path || "";
    STATE.transcript_path = latest.transcript_path || "";
    STATE.audio_meta = latest.audio_meta || null;
    STATE.video_path = latest.video_path || "";
    STATE.video_meta = latest.video_meta || null;

    const vbadge = renderVideoBadge(STATE.video_path, STATE.video_meta);
    const abadge = renderAudioBadge(STATE.audio_meta);
    const rbadge = renderResumeBadge(STATE.resume_path);

    const prev = esc(latest.transcript_preview || '');
    const previewHTML = prev
      ? ` <details style="display:inline-block;margin-left:8px;">
           <summary style="cursor:pointer;color:#e8edf2;display:inline;">Transcript preview</summary>
           <pre style="white-space:pre-wrap;margin:.35rem 0 0;color:#a8b2bd;">${prev}</pre>
         </details>` : '';

    const bits = [];
    if (STATE.resume_path) bits.push(rbadge);
    if (STATE.audio_meta) bits.push(abadge);
    if (STATE.video_meta) bits.push(vbadge);

    setMsgHTML(bits.join(' · ') + previewHTML);
    enforceButtons(latest);
  } catch (e) { setMsg('Load failed'); }
});

// Compile JD (always Live: ?dry=0) — show plain headings (no receipts note) until after build
$('compileBtn').addEventListener('click', async () => {
  const text = $('jd').value.trim();
  if (!text) { alert('Paste a JD first.'); return; }
  const qs = '?dry=0';
  const r = await gateWithSpinner(fetch('/jd/60s' + qs, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text }) }), 'Compile');
  if (!r.ok) { alert('Compile failed.'); return; }
  const data = await r.json();
  const s = data.schema || {};
  STATE.schema = s;

  const companyRaw = (s.company ?? '').toString().trim();
  const hideCompany = !companyRaw || /^(n\/?a|none|null|not specified|unspecified)$/i.test(companyRaw);

  $('role').textContent = s.role_title || '—';
  $('company').textContent = hideCompany ? '' : companyRaw;
  $('company').style.display = hideCompany ? 'none' : 'block';

  const grid = $('themes'); grid.innerHTML = '';
  (s.themes || []).forEach(t => {
    const div = document.createElement('div');
    div.className = 'theme';
    const reqList = (t.requirements || []).map(x => `<li>${esc(x)}</li>`).join('');
    const siList  = (t.success_indicators || []).map(x => `<li>${esc(x)}</li>`).join('');
    div.innerHTML = `
      <h3>${t.name || 'Theme'}</h3>
      <strong>Requirements</strong>
      <ul class="reqs">${reqList}</ul>
      <strong>Success Indicators</strong>
      <ul class="reqs">${siList}</ul>
    `;
    grid.appendChild(div);
  });

  setMsg('Compiled ✓ now attach artifacts or build evidence');
  $('coverage').style.display = 'none';
  $('covOverall').textContent = '';
  $('covThemes').innerHTML = '';
  $('result').style.display = 'block';
});

// Clear UI
$('clearBtn').addEventListener('click', () => {
  $('jd').value = '';
  $('themes').innerHTML = '';
  $('role').textContent = '';
  $('company').textContent = '';
  $('company').style.display = 'none';
  $('coverage').style.display = 'none';
  $('covOverall').textContent = '';
  $('covThemes').innerHTML = '';
  setMsg('');
  STATE.schema = null;
});

// Attach resume
$('resumeBtn').addEventListener('click', () => $('resumeFile').click());
$('resumeFile').addEventListener('change', async () => {
  const f = $('resumeFile').files[0]; if (!f) return;
  const cand = ($('candSelect').value || $('cand').value || '').trim() || 'anon';
  const qs = '?dry=0';
  const fd = new FormData(); fd.append('candidate_id', cand); fd.append('file', f);

  setMsg('Uploading resume…');
  try {
    const res = await gateWithSpinner(postForm('/artifacts/resume' + qs, fd), 'Upload');
    STATE.resume_path = res.path || "";
    const vbadge = renderVideoBadge(STATE.video_path, STATE.video_meta);
    const abadge = renderAudioBadge(STATE.audio_meta);
    const rbadge = renderResumeBadge(STATE.resume_path);
    setMsgHTML(`${rbadge}${STATE.audio_meta ? ' · '+abadge : ''}${STATE.video_meta ? ' · '+vbadge : ''}`);
  } catch (e){ setMsg('Resume upload failed'); }
  finally { $('resumeFile').value = ''; }
});

// Helpers
const fetchLatestFor = async (cand) => {
  const r = await fetch(`/artifacts/for/${encodeURIComponent(cand)}`);
  if (!r.ok) return null;
  return r.json();
};

// Attach audio (with replace flow + success alert)
$('audioBtn').addEventListener('click', () => $('audioFile').click());
$('audioFile').addEventListener('change', async () => {
  const f = $('audioFile').files[0]; if (!f) return;
  const cand = ($('candSelect').value || $('cand').value || '').trim() || 'anon';
  const hasVideo = !!STATE.video_path;
  let qs = '?dry=0';
  let replacing = false;
  if (hasVideo){
    const ok = window.confirm('Replace existing video with this audio? This will remove the current video + transcript.');
    if (!ok){ $('audioFile').value=''; return; }
    qs += '&replace=1';
    replacing = true;
  }
  const fd = new FormData(); fd.append('candidate_id', cand); fd.append('file', f);

  setMsg('Transcribing…');
  try {
    const res = await gateWithSpinner(postForm('/artifacts/audio' + qs, fd), 'Transcribe');
    try {
      const latest = await fetchLatestFor(cand);
      if (latest?.latest) {
        STATE.resume_path = latest.latest.resume_path || STATE.resume_path;
      }
    } catch {}
    STATE.video_path = ""; STATE.video_meta = null; // replaced by audio
    STATE.transcript_path = res.transcript_path || "";
    STATE.audio_meta = res.audio_meta || null;

    const vbadge = renderVideoBadge(STATE.video_path, STATE.video_meta);
    const rbadge = renderResumeBadge(STATE.resume_path);
    setMsgHTML(`${rbadge ? rbadge + ' · ' : ''}${renderAudioBadge(STATE.audio_meta)}${res.preview ? ` · <span class="muted-tip">${esc(res.preview)}</span>` : ''}`);

    await refreshCandidates(); $('cand').value = cand; $('candSelect').value = cand;
    enforceButtons({ transcript_chars: res.transcript_chars, video_path: STATE.video_path, audio_meta: STATE.audio_meta });

    if (replacing) window.alert('Replaced existing video with this audio. Transcript refreshed.');
  } catch (e){ setMsg('Audio upload failed: ' + (e?.message||'')); }
  finally { $('audioFile').value = ''; }
});

// Attach video (with replace flow + success alert)
$('videoBtn').addEventListener('click', () => $('videoFile').click());
$('videoFile').addEventListener('change', async () => {
  const f = $('videoFile').files[0]; if (!f) return;
  const cand = ($('candSelect').value || $('cand').value || '').trim() || 'anon';

  const hasAudio = (!!STATE.transcript_path && !STATE.video_path) || !!STATE.audio_meta;

  let qs = '?dry=0';
  let replacing = false;
  if (hasAudio){
    const ok = window.confirm('Replace existing audio with this video? This will remove the current audio + transcript.');
    if (!ok){ $('videoFile').value=''; return; }
    qs += '&replace=1';
    replacing = true;
  }

  const fd = new FormData(); fd.append('candidate_id', cand); fd.append('file', f);

  setMsg('Processing video… (extracting audio → transcribing)');
  try {
    const res = await gateWithSpinner(postForm('/artifacts/video' + qs, fd), 'FFmpeg · Whisper');
    try {
      const latest = await fetchLatestFor(cand);
      if (latest?.latest) {
        STATE.resume_path = latest.latest.resume_path || STATE.resume_path;
      }
    } catch {}
    STATE.transcript_path = res.transcript_path || "";
    STATE.audio_meta = null;
    STATE.video_path = res.video_path || "";
    STATE.video_meta = res.video_meta || null;

    const prev = esc(res.preview || '').trim();
    const vbadge = renderVideoBadge(STATE.video_path, STATE.video_meta);
    const rbadge = renderResumeBadge(STATE.resume_path);
    setMsgHTML(`${rbadge ? rbadge + ' · ' : ''}${vbadge}${prev ? ` · <span class="muted-tip">${prev}</span>` : ''}`);

    await refreshCandidates(); $('cand').value = cand; $('candSelect').value = cand;
    enforceButtons({ transcript_chars: res.transcript_chars, video_path: STATE.video_path, audio_meta: STATE.audio_meta });

    if (replacing) window.alert('Replaced existing audio with this video. Transcript refreshed.');
  } catch (e){ setMsg('Video upload failed: ' + (e?.message||'')); }
  finally { $('videoFile').value = ''; }
});

// Evidence builder

// Highlight terms on raw text, escape segments; merge overlaps
function highlightTerms(rawSnippet, terms){
  const text = (rawSnippet ?? '').toString();
  const uniq = Array.from(new Set((terms || []).map(t => (t||'').toString()).filter(Boolean)));
  uniq.sort((a,b) => (b.length||0)-(a.length||0));
  const lower = text.toLowerCase();
  const ranges = [];
  for (const term of uniq){
    const t = term.toLowerCase(); if (!t) continue;
    let idx = 0;
    while (idx < lower.length){
      const at = lower.indexOf(t, idx);
      if (at === -1) break;
      ranges.push([at, at + t.length]);
      idx = at + Math.max(1, t.length);
    }
  }
  ranges.sort((a,b)=>a[0]-b[0]);
  const merged = [];
  for (const r of ranges){
    if (!merged.length || r[0] > merged[merged.length-1][1]) merged.push(r.slice());
    else merged[merged.length-1][1] = Math.max(merged[merged.length-1][1], r[1]);
  }
  let out = ''; let prev = 0;
  for (const [s,e] of merged){
    const before = text.slice(prev, s);
    const hit = text.slice(s, e);
    out += esc(before) + '<span class="hit">' + esc(hit) + '</span>';
    prev = e;
  }
  out += esc(text.slice(prev));
  return out;
}

$('evidenceBtn').addEventListener('click', async () => {
  if (!STATE.schema) { alert('Compile the JD first.'); return; }
  const cand = ($('candSelect').value || $('cand').value || '').trim() || 'anon';
  const payload = { candidate_id: cand, schema: STATE.schema, claims: STATE.claims || [], transcript_path: STATE.transcript_path || null };

  setMsg('Building evidence map…');
  const qs = '?dry=0';
  try {
    const r = await gateWithSpinner(fetch('/evidence/build' + qs, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) }), 'Build');
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    renderEvidence(data);
  } catch (e) {
    const ok = window.confirm('Build failed. Open console details?');
    if (ok) console.error(e);
    setMsg('Build failed');
  }
});

// Render evidence (post-build headings include receipts note)
function renderEvidence(data){
  $('coverage').style.display = 'flex';
  $('covOverall').textContent = `Overall: ${data.coverage?.overall ?? '?'}%`;
  const ct = $('covThemes'); ct.innerHTML = '';
  const by = data.coverage?.by_theme || {};
  Object.keys(by).forEach(name => { const pill = document.createElement('span'); pill.className='pill'; pill.textContent = `${name}: ${by[name]}%`; ct.appendChild(pill); });

  const grid = $('themes'); grid.innerHTML = '';
  const adapter = data.adapter || {};
  const schemaThemes = STATE.schema?.themes || [];

  Object.keys(adapter).forEach(tname => {
    const rows = adapter[tname] || [];
    const div = document.createElement('div');
    div.className = 'theme';

    const reqList = rows.map(row => {
      const evid = (row.evidence || []).map(ev => {
        const src = (ev.source_ref?.kind || 'artifact');
        const line = ev.source_ref?.line != null ? ` #${ev.source_ref.line}` : '';
        const snippetHTML = highlightTerms(ev.snippet, ev.hit_terms || []);
        return `<li class="muted">[${src}${line}] <span class="receipt__snippet">${snippetHTML}</span></li>`;
      }).join('');
      const receipt = evid ? `<details class="receipt"><summary>Receipts</summary><ul>${evid}</ul></details>` : '';

      const oqs = (row.open_questions || []);
      const oqHTML = oqs.length ? `<div class="open-qs"><em>Open questions</em><ul>${oqs.map(q => `<li class="muted">${esc(q)}</li>`).join('')}</ul></div>` : '';

      return `<li><span>${esc(row.requirement || '')}</span>${receipt}${oqHTML}</li>`;
    }).join('');

    const sch = schemaThemes.find(th => (th.name || 'Theme') === tname);
    let siArray = Array.isArray(sch?.success_indicators) ? sch.success_indicators.slice() : [];
    if (!siArray.length) {
      const fromRows = rows.flatMap(r => r.success_indicators || []);
      const seen = new Set();
      siArray = fromRows.filter(x => (x = (x||'').toString().trim()) && !seen.has(x) && seen.add(x));
    }
    const siList = siArray.map(si => `<li><span>${esc(si)}</span></li>`).join('');

    div.innerHTML = `
      <h3>${tname}</h3>
      <strong>Requirements (matched items show receipts)</strong>
      <ul class="reqs">${reqList}</ul>
      <strong>Success Indicators</strong>
      <ul class="reqs">${siList}</ul>
    `;
    grid.appendChild(div);
  });

  setMsg('Evidence ✓  coverage updated');
  document.querySelector('#result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
