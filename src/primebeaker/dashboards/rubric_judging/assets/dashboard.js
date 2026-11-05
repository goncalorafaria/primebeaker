import {el, escape, number, duration, palette, getJSON, renderPlot} from './core.js';

let config = null, snapshot = null;
function freshness() {
  if (!snapshot) return;
  const seconds = (Date.now() - Date.parse(snapshot.timestamp)) / 1000;
  el('freshness').textContent = 'Snapshot ' + new Date(snapshot.timestamp).toLocaleString() + ' · ' + Math.max(0, Math.round(seconds)) + ' seconds ago';
  el('freshness').className = seconds > Math.max(45, snapshot.refresh_seconds * 3) ? 'stale' : 'fresh';
}

function runState(id) {
  return snapshot.jobs.find(j => j.role === id)?.state || 'unknown';
}

function render(s) {
  snapshot = s;
  freshness();
  const runs = config.runs, primary = s.runs[runs[0].id]?.trainer || {};
  el('step').textContent = primary.completed_optimizer_step ?? (runState(runs[0].id) === 'running' ? 'First step pending' : runState(runs[0].id));
  el('batch').textContent = 'Next rollout batch ' + (primary.train_batch || 'pending') + ' · ' + number(primary.episodes_inflight) + ' inflight · ' + number(primary.buffered_episodes) + ' buffered';
  if (runs[1]) {
    const second = s.runs[runs[1].id]?.trainer || {};
    el('base').textContent = second.completed_optimizer_step ?? runState(runs[1].id);
    el('basebatch').textContent = 'Rollout batch ' + (second.train_batch || 'pending');
  } else {
    el('base').textContent = 'Not configured';
    el('basebatch').textContent = '';
  }
  const judges = (s.inference.judge || []).filter(e => e.available);
  el('replicas').textContent = judges.length + ' / ' + (s.judge_pool.requested ?? '—');
  el('load').textContent = judges.length ? number(judges.reduce((a, e) => a + e.active, 0)) + ' / ' + number(judges.reduce((a, e) => a + e.queued, 0)) : '—';
  const knownRates = judges.filter(e => e.output_tokens_per_second != null);
  el('throughput').textContent = knownRates.length ? number(knownRates.reduce((a, e) => a + e.output_tokens_per_second, 0)) + ' output tokens/s across ' + knownRates.length + ' sampled judges' : 'Throughput unavailable';
  let rows = '';
  for (const item of [...runs, {id: 'judge', label: config.judge_label}]) {
    let endpoints = s.inference[item.id] || [];
    if (!endpoints.length) endpoints = [{available: false, error: item.id === 'judge' ? 'No registered endpoints' : runState(item.id)}];
    for (const e of endpoints) rows += '<tr><td>' + escape(item.label) + '</td><td>' + escape(e.available ? e.endpoint : e.error) + '</td><td class="number">' + number(e.active) + '</td><td class="number">' + number(e.queued) + '</td><td class="number">' + number(e.output_tokens_per_second) + '</td><td class="number">' + (e.kv_cache_percent_mean == null ? '—' : number(e.kv_cache_percent_mean) + '%') + '</td><td class="number">' + duration(e.request_latency_seconds_interval_mean) + '</td><td class="number">' + duration(e.queue_seconds_interval_mean) + '</td></tr>';
  }
  el('inference').innerHTML = rows;
  let jobs = '';
  for (const job of s.jobs) {
    let state = job.state;
    if (job.role === 'judge' && state === 'running') state = judges.some(e => job.metrics_endpoint && e.endpoint === job.metrics_endpoint) ? 'Serving' : 'Running / initializing';
    jobs += '<tr><td>' + escape(job.label) + '</td><td><span class="badge">' + escape(state) + '</span></td><td>' + escape(job.host) + '</td><td><a target="_blank" rel="noopener" href="' + escape(job.url) + '">Open experiment</a></td></tr>';
  }
  el('jobs').innerHTML = jobs;
  let training = '';
  for (const run of runs) {
    const metrics = s.runs[run.id]?.trainer || {}, times = metrics.timing_seconds || {};
    const throughput = (metrics.latest_completed_step_line || '').match(/Throughput ([\d.]+) tokens\/s/);
    training += '<tr><td>' + escape(run.label) + '</td><td class="number">' + number(metrics.completed_optimizer_step) + '</td><td class="number">' + escape(times.step != null ? duration(times.step) : metrics.latest_step_duration_reported || '—') + '</td><td class="number">' + duration(times.wait_for_batch) + '</td><td class="number">' + duration(times.forward_backward) + '</td><td class="number">' + duration(times.broadcast_weights) + '</td><td class="number">' + number(throughput ? Number(throughput[1]) : null) + '</td></tr>';
  }
  el('training').innerHTML = training;
  el('timing').textContent = 'Step time includes rollout-batch waiting. Detailed timing is retained when sparse trainer rows appear in the file monitor.';
  const errors = Object.entries(s.source_errors || {}).map(([source, error]) => source + ': ' + error);
  el('error').textContent = errors.join(' · ');
  el('error').style.display = errors.length ? 'block' : 'none';
}

async function update() {
  try { render(await getJSON('/api/snapshot')); }
  catch (error) { el('error').textContent = 'Cannot refresh: ' + error.message; el('error').style.display = 'block'; }
}

async function curves() {
  try {
    const data = await getJSON('/api/curves');
    for (const [kind, suffix, scale] of [['time_per_step', 'time', 60], ['reward_mean', 'reward', 1], ['total_tokens_mean', 'tokens', 1]]) {
      const series = config.runs.map((run, i) => ({name: run.label, color: palette[i % palette.length], points: data.runs[run.id]?.curves?.[kind]?.points || []}));
      renderPlot(el('plot-' + suffix), series, {label: kind, scale});
      el('source-' + suffix).textContent = [...new Set(config.runs.map(r => data.runs[r.id]?.curves?.[kind]?.metric).filter(Boolean))].join(' · ');
    }
    el('curves-status').textContent = 'W&B checked ' + new Date(data.fetched_at).toLocaleString() + ' · ' + config.runs.map(run => run.label + ': ' + (data.runs[run.id]?.available ? 'history loaded' : data.runs[run.id]?.curves ? 'cached history; refresh unavailable' : 'no history available yet')).join(' · ');
  } catch (error) { el('curves-status').textContent = error.message; }
}

async function start() {
  try {
    config = await getJSON('/api/config');
    el('title').textContent = config.title;
    document.title = config.title;
    el('primary-label').textContent = config.runs[0].label;
    el('comparison-label').textContent = config.runs[1]?.label || 'Comparison policy';
    document.querySelector('.legend').innerHTML = config.runs.map((run, i) => '<span style="color:' + palette[i % palette.length] + '">' + escape(run.label) + '</span>').join('');
    el('refresh').onclick = () => { update(); curves(); };
    await update();
    curves();
    setInterval(update, 5000);
    setInterval(curves, 15000);
    setInterval(freshness, 1000);
  } catch (error) { el('error').textContent = error.message; el('error').style.display = 'block'; }
}
start();
