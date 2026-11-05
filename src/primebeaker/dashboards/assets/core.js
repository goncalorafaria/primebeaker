// Shared presentation primitives for dashboard views. No run names or API keys.
export const el = id => document.getElementById(id);
export const escape = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const number = value => value == null ? '—' : Number(value).toLocaleString(undefined, {maximumFractionDigits: 1});
export const duration = value => value == null ? '—' : value >= 60 ? (value / 60).toFixed(1) + ' min' : value.toFixed(2) + ' s';
export const palette = ['#56B4E9', '#E69F00', '#009E73', '#CC79A7', '#F0E442', '#D55E00'];

export async function getJSON(url) {
  const response = await fetch(url + '?t=' + Date.now(), {cache: 'no-store'});
  if (!response.ok) {
    const message = await response.json().catch(() => ({}));
    throw Error(message.error || ('HTTP ' + response.status));
  }
  return response.json();
}

export function renderPlot(target, series, {label = 'Metric', scale = 1, xMode = 'step', zeroFloor = false} = {}) {
  series = series.map((s, i) => ({...s, color: s.color || palette[i % palette.length], points: (s.points || [])
    .filter(p => Number.isFinite(p.step)).map(p => ({x: p.step, y: Number.isFinite(p.value) ? p.value / scale : null, raw: p.value}))}));
  const points = series.flatMap(s => s.points).filter(p => p.y !== null);
  if (!points.length) {
    target.innerHTML = '<p class="hint">No recorded points yet.</p>';
    return;
  }
  let xmin = Math.min(...points.map(p => p.x)), xmax = Math.max(...points.map(p => p.x));
  let ymin = Math.min(...points.map(p => p.y)), ymax = Math.max(...points.map(p => p.y));
  if (xmin === xmax) { xmin = Math.max(0, xmin - 1); xmax += 1; }
  const pad = (ymax - ymin) * .12 || Math.max(Math.abs(ymax) * .08, .1);
  ymin -= pad; ymax += pad;
  if (zeroFloor) ymin = 0;
  const xLabel = xMode === 'time' ? 'Time (local)' : 'Training step (W&B)';
  const xText = value => xMode === 'time' ? new Date(value).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) : value;
  const w = 560, h = 310, left = 65, right = 16, top = 20, bottom = 48;
  const px = x => left + (x - xmin) / (xmax - xmin) * (w - left - right);
  const py = y => h - bottom - (y - ymin) / (ymax - ymin) * (h - top - bottom);
  let svg = '<svg viewBox="0 0 ' + w + ' ' + h + '" role="img" aria-label="' + escape(label + ' against ' + xLabel) + '">';
  for (let i = 0; i <= 4; i++) {
    const value = ymin + (ymax - ymin) * i / 4, y = py(value);
    svg += '<line x1="' + left + '" y1="' + y + '" x2="' + (w - right) + '" y2="' + y + '" stroke="#2d3b50"/>';
    svg += '<text x="' + (left - 9) + '" y="' + (y + 4) + '" text-anchor="end" fill="#a6b4c9" font-size="13">' + number(value) + '</text>';
  }
  const ticks = [...new Set(Array.from({length: 5}, (_, i) => Math.round(xmin + (xmax - xmin) * i / 4)))];
  for (const value of ticks) svg += '<text x="' + px(value) + '" y="' + (h - bottom + 24) + '" text-anchor="middle" fill="#a6b4c9" font-size="13">' + escape(xText(value)) + '</text>';
  svg += '<text x="' + ((w + left - right) / 2) + '" y="' + (h - 5) + '" text-anchor="middle" fill="#a6b4c9" font-size="13">' + escape(xLabel) + '</text>';
  for (const source of series) {
    let segment = [];
    const flush = () => {
      if (segment.length > 1) svg += '<polyline fill="none" stroke="' + source.color + '" stroke-width="2.5" points="' + segment.map(p => px(p.x) + ',' + py(p.y)).join(' ') + '"/>';
      segment = [];
    };
    for (const p of source.points) {
      if (p.y === null) { flush(); continue; }
      segment.push(p);
      svg += '<circle cx="' + px(p.x) + '" cy="' + py(p.y) + '" r="' + (xMode === 'time' ? 2 : 4) + '" fill="' + source.color + '"><title>' + escape(source.name + ' · ' + (xMode === 'time' ? new Date(p.x).toLocaleString() : 'step ' + p.x) + ' · ' + number(p.raw)) + '</title></circle>';
    }
    flush();
  }
  target.innerHTML = svg + '</svg>';
}
