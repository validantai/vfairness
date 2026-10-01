/* vfairness Library Usage Map — interactive matrix, filters, flow.
 *
 * Consumes docs/site/data/usage_map.json (schema v2 with tracks, categories,
 * and SVG-template attribution).
 */
(function () {
  'use strict';

  const DATA_URL = '../data/usage_map.json';
  // There is deliberately NO second data URL here (2026-09-10). This page is
  // served unauthenticated, and it used to fetch a sibling JSON under ../data/
  // that is generated from the private platform checkout. Measured that day
  // against the live site: that object answered HTTP 200 with 366447 bytes,
  // disclosing the checkout name, a commit sha, the branch and 82 internal
  // handler paths, and the rendered DOM carried all four. Removing the pointer
  // is the control: a visitor's browser never asks for it, whether or not the
  // object is still there. Do not reintroduce a fetch of it; if those views are
  // wanted again they need a surface that is not public.
  //
  // The artifact is not named here on purpose. Its filename is on the export
  // leak denylist, so writing it in this comment would abort the export on this
  // very file. The export script's EXCLUDES block names it in full.
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));

  const STATUS = {
    active:        { label: 'Active',       cls: 'um-tag um-tag--active' },
    'library-only':{ label: 'Library-only', cls: 'um-tag um-tag--lib' },
    unused:        { label: 'Unused',       cls: 'um-tag um-tag--unused' },
  };

  const state = {
    data: null,
    rows: [],
    filtered: [],
    sort: { key: 'symbol', dir: 1 },
    filters: { category: '', track: '', svg: '', status: '', q: '', step: '' },
    pipelineKey: 'predictive',
    // 'navigator' | 'pulse' | 'modules'
    surface: 'navigator',
    platform: null,
    laneKey: 'tabular',
  };

  async function init() {
    try {
      const r = await fetch(DATA_URL, { cache: 'no-cache' });
      state.data = await r.json();
    } catch (e) {
      $('#um-root').innerHTML = '<p class="um-error">Could not load <code>data/usage_map.json</code>. Run <code>python scripts/build_usage_map.py</code> first.</p>';
      return;
    }
    // state.platform stays null. It is kept as a field, not deleted, because
    // every surface guard below reads it: null is what makes this page report
    // the Navigator view only, rather than an empty Pulse view that would look
    // like a measurement of nothing.

    state.rows = state.data.rows;
    renderMeta();
    renderSummary();
    renderTrackSelector();
    bindPipelineToggle();
    renderFlow();
    renderTrackList();
    renderSvgCoverage();
    bindFilters();
    bindSurfaceToggle();
    applyFilters();
    applySurfaceFromHash();
  }

  // ── Surfaces ────────────────────────────────────────────────────────────

  function surfaceData(key) {
    if (key === 'navigator') return null;
    return ((state.platform || {}).surfaces || {})[key] || null;
  }

  function bindSurfaceToggle() {
    const btns = $$('.um-surface-btn');
    const available = !!state.platform;
    btns.forEach(b => {
      const key = b.dataset.surfaceBtn;
      if (key !== 'navigator' && !available) {
        b.disabled = true;
        b.title = 'This view is not published here.';
        b.style.opacity = '0.45';
        b.style.cursor = 'not-allowed';
        return;
      }
      b.addEventListener('click', () => setSurface(key));
    });
  }

  function applySurfaceFromHash() {
    const h = (location.hash || '').replace('#', '');
    if (h === 'pulse' || h === 'modules') setSurface(h);
    else setSurface('navigator');
  }

  function setSurface(key) {
    if (key !== 'navigator' && !surfaceData(key)) key = 'navigator';
    state.surface = key;
    $$('.um-surface-btn').forEach(b => {
      const on = b.dataset.surfaceBtn === key;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    $$('[data-surface]').forEach(sec => {
      sec.hidden = !sec.dataset.surface.split(' ').includes(key);
    });
    // Track / category filters only describe Navigator tracks.
    const navFilters = $('#um-nav-filters');
    if (navFilters) navFilters.style.display = (key === 'navigator') ? 'contents' : 'none';
    if (key !== 'navigator') { state.filters.category = ''; state.filters.track = ''; state.filters.step = ''; }

    state.rows = (key === 'navigator') ? state.data.rows : surfaceData(key).rows;
    renderSummary();
    renderMeta();
    renderMatrixCopy();
    applyFilters();
  }

  const SURFACE_COPY = {
    navigator: {
      lede: 'A live map of every public <strong>library symbol</strong> — that is, every function or class the <code>vfairness</code> package exports (such as <code>CorrelationReducer</code> or <code>demographic_parity_difference</code>) — against the Navigator tracks in the library repo that consume it.',
      scope: 'symbols invoked by at least one Navigator track',
      matrixTitle: 'Every library symbol, every call site',
      matrixLede: 'Each row is a <strong>symbol</strong> — a public function or class exported by the <code>vfairness</code> package. Filter by track, SVG template or status; sort any column.',
      stepCol: 'Navigator step(s)',
      tracksCol: 'Tracks',
      tracksLabel: 'Tracks scanned',
    },
  };

  function renderMatrixCopy() {
    // Fall back rather than throw. state.surface can only be 'navigator'
    // today, but a missing copy block must not blank the whole page.
    const c = SURFACE_COPY[state.surface] || SURFACE_COPY.navigator;
    const set = (sel, html) => { const el = $(sel); if (el) el.innerHTML = html; };
    set('#um-hero-lede', c.lede);
    set('#um-coverage-scope', c.scope);
    set('#um-matrix-title', c.matrixTitle);
    set('#um-matrix-lede', c.matrixLede);
    set('#um-th-step', c.stepCol);
    set('#um-th-tracks', c.tracksCol);
    const tl = $('#um-tracks-label');
    if (tl) tl.textContent = c.tracksLabel;
  }

  function bindPipelineToggle() {
    $$('.um-pipeline-btn').forEach(b => {
      b.addEventListener('click', () => {
        const key = b.dataset.pipeline;
        if (key === state.pipelineKey) return;
        state.pipelineKey = key;
        $$('.um-pipeline-btn').forEach(x => {
          x.classList.toggle('is-active', x.dataset.pipeline === key);
          x.setAttribute('aria-selected', x.dataset.pipeline === key ? 'true' : 'false');
        });
        renderFlow();
      });
    });
  }

  function renderMeta() {
    const d = state.data;
    if (state.surface === 'navigator') {
      $('#um-meta').textContent = `Library v${d.library_version} · ${d.summary.tracks_scanned} Navigator track(s) scanned · Generated ${d.generated_at}`;
      return;
    }
    const pf = state.platform || {};
    const s = surfaceData(state.surface).summary;
    const what = state.surface === 'pulse'
      ? `${s.lanes} Pulse pathway(s) · ${s.engine_files} engine file(s)`
      : `${s.modules_engine_backed} of ${s.modules_total} modules reach the engine`;
    $('#um-meta').textContent =
      `Library v${pf.library_version || d.library_version} · ${what} · Platform ${(pf.platform||{}).root || 'checkout'} @ ${(pf.platform||{}).commit || '?'} · Generated ${pf.generated_at}`;
  }

  function renderSummary() {
    const s = (state.surface === 'navigator')
      ? state.data.summary
      : surfaceSummaryShape(surfaceData(state.surface).summary, state.surface);
    $('#um-coverage-bar').style.setProperty('--um-pct', s.coverage_pct + '%');
    $('#um-coverage-pct').textContent = s.coverage_pct + '%';
    $('#um-active').textContent = s.active;
    const a2 = $('#um-active2'); if (a2) a2.textContent = s.active;
    $('#um-total').textContent = s.total_symbols;
    $('#um-libonly').textContent = s.library_only;
    $('#um-unused').textContent = s.unused;
    $('#um-tracks').textContent = s.tracks_scanned;
    $('#um-svgs').textContent = `${s.svg_templates_active} / ${s.svg_templates_total}`;
    renderCombinedReach();
  }

  /** No single surface reaches the whole library, so a per-surface percentage
   *  reads as a coverage gap that is not there. This line states the union and
   *  names how many symbols THIS view calls unused while another view uses. */
  function renderCombinedReach() {
    const el = $('#um-combined');
    if (!el) return;
    if (!state.platform) { el.textContent = ''; return; }
    const sets = {
      navigator: new Set(state.data.rows.filter(r => r.status === 'active').map(r => r.symbol)),
      pulse: new Set((surfaceData('pulse') || { rows: [] }).rows.filter(r => r.status === 'active').map(r => r.symbol)),
      modules: new Set((surfaceData('modules') || { rows: [] }).rows.filter(r => r.status === 'active').map(r => r.symbol)),
    };
    const union = new Set([...sets.navigator, ...sets.pulse, ...sets.modules]);
    const total = state.data.rows.length;
    const here = sets[state.surface] || new Set();
    const elsewhere = [...union].filter(x => !here.has(x)).length;
    el.innerHTML =
      `Across all three surfaces, <strong>${union.size}</strong> of <strong>${total}</strong> symbols are reached ` +
      `(${Math.round(100 * union.size / total)}%). <strong>${elsewhere}</strong> of them are reached by a surface other than this one, ` +
      `so a symbol marked unused here is not necessarily unused in the product.`;
  }

  /** Map a platform-surface summary onto the shape the hero tiles read. */
  function surfaceSummaryShape(s, key) {
    return {
      coverage_pct: s.coverage_pct,
      active: s.active,
      total_symbols: s.total_symbols,
      library_only: 0,
      unused: s.unused,
      tracks_scanned: (key === 'pulse') ? s.lanes : s.modules_engine_backed,
      svg_templates_active: s.svg_templates_active,
      svg_templates_total: s.svg_templates_total,
    };
  }

  function renderCategoryChips() {
    const cats = state.data.categories;
    $('#um-categories').innerHTML = cats.map(c => `
      <button type="button" class="um-cat-chip" data-category="${esc(c.label)}" role="tab">
        <span class="um-cat-chip__dot" style="background:${categoryColor(c.label)}"></span>
        <span>${esc(c.label)}</span>
        <span class="um-cat-chip__count">${c.active_symbols}</span>
      </button>
    `).join('');
    $$('.um-cat-chip').forEach(btn => {
      btn.addEventListener('click', () => {
        const v = btn.dataset.category;
        state.filters.category = (state.filters.category === v) ? '' : v;
        $('#um-filter-category').value = state.filters.category;
        reflectChips();
        applyFilters();
      });
    });
  }

  function renderTrackSelector() {
    const catSel = $('#um-filter-category');
    const trkSel = $('#um-filter-track');
    const svgSel = $('#um-filter-svg');
    catSel.innerHTML = '<option value="">All categories</option>' +
      state.data.categories.map(c => `<option value="${esc(c.label)}">${esc(c.label)}</option>`).join('');
    trkSel.innerHTML = '<option value="">All tracks</option>' +
      state.data.tracks.map(t => `<option value="${t.id}" data-category="${esc(t.category)}">${esc(t.label)}</option>`).join('');
    svgSel.innerHTML = '<option value="">All SVG templates</option>' +
      state.data.svg_templates.map(s => `<option value="${esc(s.template)}">${esc(s.template)}${s.active ? '' : ' · unused'}</option>`).join('');
  }

  // The Navigator, Pulse and Modules views all draw the same three-column
  // Sankey. Only the left column changes meaning (Navigator step / Pulse stage
  // / platform module), so the renderer takes a config instead of reading the
  // Navigator pipeline directly.
  function renderFlow(cfg) {
    if (!cfg) {
      const pipeline = (state.data.pipelines || {})[state.pipelineKey];
      if (!pipeline) { $('#um-flow').innerHTML = '<p class="um-dim">No pipeline data.</p>'; return; }
      cfg = {
        mount: '#um-flow',
        flow: pipeline,
        rows: state.data.rows,
        axis: ['Navigator step', 'Library module', 'SVG template'],
        colorFn: (sid) => NAV_STEP_COLORS[sid] || '#1a3a4f',
        tracks: state.data.tracks,
        svgTemplates: state.data.svg_templates,
      };
    }
    const mount = $(cfg.mount);
    if (!mount) return;
    const flow = cfg.flow || {};
    const rows = cfg.rows || [];
    const steps   = flow.steps || [];
    const s2m     = flow.step_to_module || [];
    const m2s     = flow.module_to_svg || [];
    const modules = Array.from(new Set(s2m.map(e => e.module))).sort();
    const svgs    = Array.from(new Set(m2s.map(e => e.svg))).sort();
    const colorFn = cfg.colorFn || (() => '#1a3a4f');
    const axis    = cfg.axis || ['Step', 'Library module', 'SVG template'];

    if (!steps.length) {
      mount.innerHTML = '<p class="um-dim">No steps defined for this view.</p>';
      return;
    }

    const W = 1100;
    const padL = 16, padR = 16;
    const colW = 210;
    const xStep = padL;
    const xMod  = (W - colW) / 2;
    const xSvg  = W - padR - colW;
    const rowH = 30;
    const H = 60 + Math.max(steps.length, modules.length, svgs.length, 1) * rowH;

    const stepIds = steps.map(s => s.id);
    const stepY = (id) => 40 + stepIds.indexOf(id) * rowH + rowH/2;
    const modY  = (m)  => 40 + modules.indexOf(m) * rowH + rowH/2;
    const svgY  = (s)  => 40 + svgs.indexOf(s) * rowH + rowH/2;

    const maxW1 = Math.max(...s2m.map(e => e.weight), 1);
    const maxW2 = Math.max(...m2s.map(e => e.weight), 1);

    // Build edge → symbol indices restricted to the rows reachable here.
    const stepSymbolSets = {};
    steps.forEach(s => stepSymbolSets[s.id] = new Set(s.symbols || []));
    const activeSymbols = new Set();
    steps.forEach(s => (s.symbols || []).forEach(x => activeSymbols.add(x)));

    const idxStepMod = {}, idxModSvg = {};
    const idxStep = {}, idxMod = {}, idxSvg = {};
    rows.filter(r => r.status === 'active' && activeSymbols.has(r.symbol)).forEach(r => {
      const modTop = ((r.module || '').split('.')[0] || '(other)').replace(/^vfairness\./, '');
      steps.forEach(s => {
        if (!stepSymbolSets[s.id].has(r.symbol)) return;
        (idxStepMod[s.id + '|' + modTop] = idxStepMod[s.id + '|' + modTop] || []).push(r);
        (idxStep[s.id] = idxStep[s.id] || new Set()).add(r.symbol);
      });
      (idxMod[modTop] = idxMod[modTop] || new Set()).add(r.symbol);
      (r.svg_templates || []).forEach(tpl => {
        (idxModSvg[modTop + '|' + tpl] = idxModSvg[modTop + '|' + tpl] || []).push(r);
        (idxSvg[tpl] = idxSvg[tpl] || new Set()).add(r.symbol);
      });
    });

    const stepColor = colorFn;

    const edges1 = s2m.map(e => {
      if (!stepIds.includes(e.step_id) || modules.indexOf(e.module) < 0) return '';
      const y1 = stepY(e.step_id), y2 = modY(e.module);
      const w = 1 + (e.weight / maxW1) * 8;
      const cx1 = xStep + colW + (xMod - xStep - colW) * 0.4;
      const cx2 = xStep + colW + (xMod - xStep - colW) * 0.6;
      return `<path d="M${xStep+colW},${y1} C${cx1},${y1} ${cx2},${y2} ${xMod},${y2}"
              class="um-edge" data-edge-type="step-mod"
              data-step="${esc(e.step_id)}" data-step-label="${esc(e.step_label)}" data-mod="${esc(e.module)}" data-weight="${e.weight}"
              stroke="${stepColor(e.step_id)}" stroke-opacity="0.45" stroke-width="${w}" fill="none"></path>`;
    }).join('');

    const edges2 = m2s.map(e => {
      if (modules.indexOf(e.module) < 0 || svgs.indexOf(e.svg) < 0) return '';
      const y1 = modY(e.module), y2 = svgY(e.svg);
      const w = 1 + (e.weight / maxW2) * 6;
      const cx1 = xMod + colW + (xSvg - xMod - colW) * 0.4;
      const cx2 = xMod + colW + (xSvg - xMod - colW) * 0.6;
      return `<path d="M${xMod+colW},${y1} C${cx1},${y1} ${cx2},${y2} ${xSvg},${y2}"
              class="um-edge um-edge--svg" data-edge-type="mod-svg"
              data-mod="${esc(e.module)}" data-svg="${esc(e.svg)}" data-weight="${e.weight}"
              stroke-width="${w}" fill="none"></path>`;
    }).join('');

    const nodeRow = (x, y, label, color, sub, dataset) => `
      <g class="um-flow-node" ${dataset}>
        <rect x="${x}" y="${y-11}" width="${colW}" height="22" fill="#fff" stroke="${color || '#cbd5e1'}"></rect>
        <text x="${x+10}" y="${y+3.5}" class="um-flow-label" font-weight="300">${esc(label)}</text>
        ${sub ? `<text x="${x+colW-10}" y="${y+3.5}" text-anchor="end" class="um-flow-sub" font-weight="300">${esc(sub)}</text>` : ''}
      </g>`;
    const nodeRowEx = (x, y, label, color, sub, dataset, empty) => `
      <g class="um-flow-node ${empty ? 'is-empty' : ''}" ${dataset}>
        <rect x="${x}" y="${y-11}" width="${colW}" height="22" fill="${empty ? '#fafbfc' : '#fff'}" stroke="${empty ? '#e2e8f0' : color}" stroke-dasharray="${empty ? '3 3' : '0'}"></rect>
        <text x="${x+10}" y="${y+3.5}" class="um-flow-label ${empty ? 'is-empty' : ''}" font-weight="300">${esc(label)}</text>
        ${sub ? `<text x="${x+colW-10}" y="${y+3.5}" text-anchor="end" class="um-flow-sub ${empty ? 'is-empty' : ''}" font-weight="300">${esc(sub)}</text>` : ''}
      </g>`;

    const stepNodes = steps.map(s => {
      const symCount = (idxStep[s.id] || new Set()).size;
      const emptyClass = s.active ? '' : ' is-empty';
      const dataset = `data-node-type="step" data-step="${esc(s.id)}" data-step-label="${esc(s.label)}" data-count="${symCount}" class-extra="${emptyClass}"`;
      return nodeRowEx(xStep, stepY(s.id), s.label, stepColor(s.id), s.active ? String(symCount) : 'empty', dataset, !s.active);
    }).join('');
    const modNodes = modules.map(m => {
      const symCount = (idxMod[m] || new Set()).size;
      return nodeRow(xMod, modY(m), m, '#42b7de', String(symCount), `data-node-type="mod" data-mod="${esc(m)}" data-count="${symCount}"`);
    }).join('');
    const svgNodes = svgs.map(s => {
      const symCount = (idxSvg[s] || new Set()).size;
      return nodeRow(xSvg, svgY(s), s, '#14b8a6', String(symCount), `data-node-type="svg" data-svg="${esc(s)}" data-count="${symCount}"`);
    }).join('');

    mount.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" class="um-flow-svg" role="img" aria-label="Flow from ${esc(axis[0])} to library module to SVG template">
        <text x="${xStep}" y="24" class="um-flow-axis">${esc(axis[0])}</text>
        <text x="${xMod}" y="24" class="um-flow-axis">${esc(axis[1])}</text>
        <text x="${xSvg}" y="24" class="um-flow-axis">${esc(axis[2])}</text>
        ${edges1}
        ${edges2}
        ${stepNodes}
        ${modNodes}
        ${svgNodes}
      </svg>
      <div class="um-tip" role="tooltip" aria-hidden="true"></div>
    `;

    bindFlowHover(mount, { idxStepMod, idxModSvg, idxStep, idxMod, idxSvg }, cfg);
  }

  const NAV_STEP_COLORS = {
    // Predictive AI
    'p1-validate':   '#0284c7',
    'p2-detect':     '#d4a017',
    'p3-feature':    '#0891b2',
    'p4-train':      '#7c3aed',
    'p5-calibrate':  '#9333ea',
    'p6-metrics':    '#0d9488',
    'p7-cicd':       '#ea580c',
    'p8-monitor':    '#16a34a',
    'p9-report':     '#1a3a4f',
    // Generative AI
    'g1-validate':   '#0284c7',
    'g2-detect':     '#d4a017',
    'g3-probes':     '#dc2626',
    'g4-calibrate':  '#9333ea',
    'g5-metrics':    '#0d9488',
    'g6-regression': '#7c3aed',
    'g7-cicd':       '#ea580c',
    'g8-monitor':    '#16a34a',
    'g9-report':     '#1a3a4f',
  };

  // ── Sankey hover: tooltip + dimming non-related edges ───────────────────
  function bindFlowHover(flow, indexes, cfg) {
    cfg = cfg || {};
    const tip = flow.querySelector('.um-tip');
    const svg = flow.querySelector('svg');
    if (!svg || !tip) return;
    const colorFn = cfg.colorFn || (() => '#1a3a4f');
    const tracks = cfg.tracks || [];
    const rows = cfg.rows || [];
    const svgTemplates = cfg.svgTemplates || [];

    const showTip = (html, x, y) => {
      tip.innerHTML = html;
      tip.classList.add('is-visible');
      tip.setAttribute('aria-hidden', 'false');
      const r = flow.getBoundingClientRect();
      const tr = tip.getBoundingClientRect();
      let px = x - r.left + 14;
      let py = y - r.top + 14;
      if (px + tr.width > r.width - 8) px = x - r.left - tr.width - 14;
      if (py + tr.height > r.height + 200) py = y - r.top - tr.height - 14;
      tip.style.transform = `translate(${Math.max(8, px)}px, ${Math.max(8, py)}px)`;
    };
    const hideTip = () => { tip.classList.remove('is-visible'); tip.setAttribute('aria-hidden','true'); };

    const dim = (predicate) => {
      svg.querySelectorAll('.um-edge').forEach(e => {
        e.classList.toggle('is-dim', !predicate(e));
        e.classList.toggle('is-hot', predicate(e));
      });
      svg.querySelectorAll('.um-flow-node').forEach(n => {
        n.classList.toggle('is-dim', !nodePred(n, predicate));
      });
    };
    const clearDim = () => {
      svg.querySelectorAll('.is-dim, .is-hot').forEach(e => e.classList.remove('is-dim', 'is-hot'));
    };
    const nodePred = (n, predicate) => {
      const t = n.dataset.nodeType;
      if (!t) return false;
      const sel = t === 'step' ? `[data-step="${n.dataset.step}"]`
                : t === 'mod'  ? `[data-mod="${n.dataset.mod}"]`
                               : `[data-svg="${n.dataset.svg}"]`;
      const edges = svg.querySelectorAll(`.um-edge${sel}`);
      for (const e of edges) if (predicate(e)) return true;
      return false;
    };

    svg.querySelectorAll('.um-edge').forEach(edge => {
      edge.addEventListener('mouseenter', ev => {
        const type = edge.dataset.edgeType;
        const w = edge.dataset.weight;
        let html = '';
        if (type === 'step-mod') {
          const sid = edge.dataset.step, slabel = edge.dataset.stepLabel, mod = edge.dataset.mod;
          const rows = indexes.idxStepMod[sid + '|' + mod] || [];
          html = renderTipFromRows(`${esc(slabel)} → <code>${esc(mod)}</code>`, `${w} symbol(s) flowing on this link`, rows);
          dim(e => e.dataset.step === sid && e.dataset.mod === mod);
        } else {
          const mod = edge.dataset.mod, tpl = edge.dataset.svg;
          const rows = indexes.idxModSvg[mod + '|' + tpl] || [];
          html = renderTipFromRows(`<code>${esc(mod)}</code> → ${esc(tpl)}`, `${w} symbol(s) produce this template from this module`, rows, tpl);
          dim(e => (e.dataset.mod === mod && e.dataset.svg === tpl));
        }
        showTip(html, ev.clientX, ev.clientY);
      });
      edge.addEventListener('mousemove', ev => showTip(tip.innerHTML, ev.clientX, ev.clientY));
      edge.addEventListener('mouseleave', () => { hideTip(); clearDim(); });
    });

    svg.querySelectorAll('.um-flow-node').forEach(node => {
      node.addEventListener('mouseenter', ev => {
        const t = node.dataset.nodeType;
        let html = '';
        if (t === 'step' && node.classList.contains('is-empty')) {
          html = `
            <div class="um-tip__head">${esc(node.dataset.stepLabel)}</div>
            <div class="um-tip__sub">Step defined in the canonical pipeline but no library symbol is wired to it yet.</div>
            <div class="um-tip__hint">Nothing in the library is wired to this step yet.</div>`;
          showTip(html, ev.clientX, ev.clientY);
          return;
        }
        if (t === 'step') {
          const sid = node.dataset.step, slabel = node.dataset.stepLabel;
          const syms = Array.from(indexes.idxStep[sid] || []).sort();
          const tracksAtStep = tracks.filter(tk =>
            rows.some(r => (r.nav_steps || []).some(s => s.id === sid) && (r.tracks || []).some(rt => rt.id === tk.id))
          );
          const stepMeta = ((cfg.flow || {}).steps || []).find(x => x.id === sid) || {};
          html = `
            <div class="um-tip__head"><span class="um-tip__dot" style="background:${colorFn(sid)}"></span>${esc(slabel)}</div>
            <div class="um-tip__sub">${syms.length} active symbol(s)${tracksAtStep.length ? ` · ${tracksAtStep.length} track(s) touch this step` : ''}</div>
            ${stepMeta.meta ? `<div class="um-tip__sub"><code>${esc(stepMeta.meta)}</code></div>` : ''}
            ${tracksAtStep.length ? `<div class="um-tip__sect">Tracks active at this step</div>
            <ul class="um-tip__list">${tracksAtStep.slice(0,6).map(tk => `<li>${esc(tk.label)}</li>`).join('')}${tracksAtStep.length>6?`<li class="um-dim">+ ${tracksAtStep.length-6} more</li>`:''}</ul>` : ''}
            <div class="um-tip__sect">Symbols invoked</div>
            <div class="um-tip__chips">${chipList(syms)}</div>`;
          dim(e => e.dataset.step === sid);
        } else if (t === 'mod') {
          const mod = node.dataset.mod;
          const syms = Array.from(indexes.idxMod[mod] || []).sort();
          html = `
            <div class="um-tip__head"><span class="um-tip__dot" style="background:#42b7de"></span><code>${esc(mod)}</code></div>
            <div class="um-tip__sub">vfairness module · ${syms.length} active symbol(s)</div>
            <div class="um-tip__sect">Symbols imported from this module</div>
            <div class="um-tip__chips">${chipList(syms)}</div>`;
          dim(e => e.dataset.mod === mod);
        } else if (t === 'svg') {
          const tpl = node.dataset.svg;
          const syms = Array.from(indexes.idxSvg[tpl] || []).sort();
          const producers = svgTemplates.find(s => s.template === tpl) || { active_tracks: [] };
          html = `
            <div class="um-tip__head"><span class="um-tip__dot" style="background:#14b8a6"></span><code>${esc(tpl)}</code></div>
            <div class="um-tip__sub">SVG template · ${(producers.active_tracks || []).length} producing track(s)</div>
            <div class="um-tip__preview"><img src="${svgPreviewURL(tpl)}" alt="" loading="lazy" onerror="this.parentElement.style.display='none'"></div>
            <div class="um-tip__sect">Producer symbols</div>
            <div class="um-tip__chips">${chipList(syms)}</div>`;
          dim(e => e.dataset.svg === tpl);
        }
        showTip(html, ev.clientX, ev.clientY);
      });
      node.addEventListener('mousemove', ev => showTip(tip.innerHTML, ev.clientX, ev.clientY));
      node.addEventListener('mouseleave', () => { hideTip(); clearDim(); });
      node.addEventListener('click', () => {
        const t = node.dataset.nodeType;
        if (t === 'step')     { state.filters.step = node.dataset.step; }
        else if (t === 'svg') { state.filters.svg = node.dataset.svg;  const el = $('#um-filter-svg'); if (el) el.value = node.dataset.svg; }
        else if (t === 'mod') { state.filters.q = node.dataset.mod;    const el = $('#um-search'); if (el) el.value = node.dataset.mod; }
        reflectChips(); applyFilters(); scrollToMatrix();
      });
    });
  }


  // ── Platform surfaces: deliberately not rendered here ───────────────────
  //
  // The Pulse and Platform-modules renderers lived here until 2026-09-10, with
  // renderPlatformViews() as their single entry point. They are gone with the
  // fetch that fed them (see DATA_URL above). Two reasons, and the second is
  // the one that mattered: they could no longer run at all once state.platform
  // was permanently null, and their markup carried private-platform source
  // paths as string literals, which this file SERVES to every visitor whether
  // or not a code path reaches them. Client-side hiding is not a control; the
  // only control is not shipping the strings.
  //
  // The surface machinery around them (surfaceData, bindSurfaceToggle,
  // setSurface) is kept: it is what makes any stray '#pulse' link fall back to
  // the Navigator view instead of throwing.

  function chipList(symbols) {
    if (!symbols.length) return '<span class="um-dim">none</span>';
    const max = 14;
    const head = symbols.slice(0, max).map(s => `<code>${esc(s)}</code>`).join('');
    const tail = symbols.length > max ? `<span class="um-dim">+ ${symbols.length - max} more</span>` : '';
    return head + tail;
  }

  function renderTipFromRows(title, sub, rows, svgTpl) {
    const syms = Array.from(new Set(rows.map(r => r.symbol))).sort();
    const preview = svgTpl
      ? `<div class="um-tip__preview"><img src="${svgPreviewURL(svgTpl)}" alt="" loading="lazy" onerror="this.parentElement.style.display='none'"></div>`
      : '';
    return `
      <div class="um-tip__head">${title}</div>
      <div class="um-tip__sub">${sub}</div>
      ${preview}
      <div class="um-tip__sect">Symbols on this link</div>
      <div class="um-tip__chips">${chipList(syms)}</div>`;
  }

  function svgPreviewURL(template) {
    // Prefer the explanation-free variant for a cleaner thumbnail; fall back via onerror.
    return `../img/svg-gallery/${template}_no_explanation.svg`;
  }

  function renderTrackList() {
    const tracks = state.data.tracks;
    const wrap = $('#um-tracks-list');
    if (!tracks.length) { wrap.innerHTML = '<p class="um-dim">No tracks discovered.</p>'; return; }
    wrap.innerHTML = tracks.map(t => `
      <button type="button" class="um-track-row" data-track="${t.id}">
        <span class="um-track-row__dot" style="background:${categoryColor(t.category)}"></span>
        <div class="um-track-row__meta">
          <div class="um-track-row__label">${esc(t.label)}</div>
          <div class="um-track-row__sub">
            <span class="um-track-row__cat">${esc(t.category)}</span>
            <code>${esc(t.path)}</code>
          </div>
        </div>
        <span class="um-track-row__num">${t.symbol_count} sym · ${t.usage_count} uses</span>
      </button>
    `).join('');
    $$('.um-track-row').forEach(b => {
      b.addEventListener('click', () => {
        const v = b.dataset.track;
        state.filters.track = (state.filters.track === v) ? '' : v;
        $('#um-filter-track').value = state.filters.track;
        reflectChips();
        applyFilters();
        scrollToMatrix();
      });
    });
  }

  function renderSvgCoverage() {
    const items = state.data.svg_templates;
    const active = items.filter(t => t.active);
    const orphans = items.filter(t => !t.active);
    const pct = items.length ? Math.round(100 * active.length / items.length) : 0;
    $('#um-svg-pct').textContent = pct + '%';
    $('#um-svg-active').textContent = active.length;
    $('#um-svg-total').textContent = items.length;
    $('#um-svg-bar').style.setProperty('--um-svg-pct', pct + '%');

    const wrap = $('#um-orphans');
    if (!orphans.length) {
      wrap.innerHTML = `<p class="um-orphans__empty">✓ Every SVG template is produced by at least one Navigator track.</p>`;
      return;
    }
    wrap.innerHTML = `
      <p class="um-orphans__title">Orphaned templates · ${orphans.length}</p>
      <ul>${orphans.map(t => `
        <li>
          <span>${esc(t.template)}</span>
          <span class="um-dim">${t.symbols.length ? t.symbols.length + ' producer symbol(s)' : 'no producer symbol'}</span>
        </li>`).join('')}</ul>
      <p class="um-orphans__hint">These templates ship with the library but no track renders them. Either wire them into a Navigator entry point or schedule them for retirement.</p>
    `;
  }

  function bindFilters() {
    $('#um-filter-category').addEventListener('change', e => { state.filters.category = e.target.value; reflectChips(); applyFilters(); });
    $('#um-filter-track').addEventListener('change',    e => { state.filters.track    = e.target.value; reflectChips(); applyFilters(); });
    $('#um-filter-svg').addEventListener('change',      e => { state.filters.svg      = e.target.value; reflectChips(); applyFilters(); });
    $('#um-search').addEventListener('input',           e => { state.filters.q        = e.target.value.trim().toLowerCase(); applyFilters(); });

    $$('.um-status-chip').forEach(c => c.addEventListener('click', () => {
      const v = c.dataset.status;
      state.filters.status = (state.filters.status === v) ? '' : v;
      $$('.um-status-chip').forEach(b => b.classList.toggle('is-active', b.dataset.status === state.filters.status));
      applyFilters();
    }));

    $('#um-clear').addEventListener('click', () => {
      state.filters = { category: '', track: '', svg: '', status: '', q: '', step: '' };
      state.rows = (state.surface === 'navigator')
        ? state.data.rows
        : surfaceData(state.surface).rows;
      $('#um-filter-category').value = '';
      $('#um-filter-track').value = '';
      $('#um-filter-svg').value = '';
      $('#um-search').value = '';
      $$('.um-status-chip, .um-cat-chip, .um-track-row').forEach(b => b.classList.remove('is-active'));
      applyFilters();
    });

    $$('#um-table th[data-sort]').forEach(th => {
      th.addEventListener('click', () => {
        const k = th.dataset.sort;
        if (state.sort.key === k) state.sort.dir *= -1;
        else { state.sort.key = k; state.sort.dir = 1; }
        $$('#um-table th[data-sort]').forEach(x => x.removeAttribute('data-active'));
        th.setAttribute('data-active', state.sort.dir > 0 ? 'asc' : 'desc');
        renderTable();
      });
    });
  }

  function reflectChips() {
    $$('.um-cat-chip').forEach(b => b.classList.toggle('is-active', b.dataset.category === state.filters.category && !!state.filters.category));
    $$('.um-track-row').forEach(b => b.classList.toggle('is-active', b.dataset.track === state.filters.track && !!state.filters.track));
  }

  function applyFilters() {
    const f = state.filters;
    state.filtered = state.rows.filter(r => {
      if (f.status   && r.status   !== f.status) return false;
      if (f.category && !(r.categories || []).includes(f.category)) return false;
      if (f.track    && !(r.tracks || []).some(t => t.id === f.track)) return false;
      if (f.step     && !(r.nav_steps || []).some(s => s.id === f.step)) return false;
      if (f.svg      && !(r.svg_templates || []).includes(f.svg)) return false;
      if (f.q) {
        const sites = (r.sites || []).map(x => (x.where || '') + ' ' + (x.source || '')).join(' ');
        const hay = (r.symbol + ' ' + r.module + ' ' + r.capability + ' ' +
                     (r.svg_templates || []).join(' ') + ' ' + sites).toLowerCase();
        if (!hay.includes(f.q)) return false;
      }
      return true;
    });
    renderTable();
  }

  function siteCount(r) {
    if (r.tracks) return r.tracks.length;
    return new Set((r.sites || []).map(x => x.where)).size;
  }

  function renderTable() {
    const tbody = $('#um-table tbody');
    const { key, dir } = state.sort;
    const cmp = (a, b) => {
      let av, bv;
      switch (key) {
        case 'use_count': av = a.use_count; bv = b.use_count; break;
        case 'status':    av = a.status; bv = b.status; break;
        case 'module':    av = a.module; bv = b.module; break;
        case 'tracks':    av = siteCount(a); bv = siteCount(b); break;
        case 'svg':       av = (a.svg_templates || []).length; bv = (b.svg_templates || []).length; break;
        default:          av = a.symbol.toLowerCase(); bv = b.symbol.toLowerCase();
      }
      return av < bv ? -dir : av > bv ? dir : 0;
    };
    const rows = state.filtered.slice().sort(cmp);
    $('#um-count').textContent = `${rows.length} of ${state.rows.length} symbols`;
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="um-empty">No symbols match the current filters.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(r => {
      const sm = STATUS[r.status] || STATUS.unused;

      // Column 3/4 read differently per surface: the Navigator has canonical
      // pipeline steps and tracks; Pulse and Modules have call SITES (which
      // stage / which module reached the symbol), so those are rendered from
      // `sites` rather than faking a step id that does not exist.
      let stepCell, tracksCell;
      if (state.surface === 'navigator') {
        const navSteps = (r.nav_steps || []);
        stepCell = navSteps.length
          ? `<div class="um-step-pills">${navSteps.map(s => {
              const c = NAV_STEP_COLORS[s.id] || '#42b7de';
              return `<span class="um-step-pill" style="--cat:${c};--cat-bg:${hexAlpha(c,0.08)};--cat-fg:${darken(c)}">${esc(s.label)}</span>`;
            }).join('')}</div>`
          : '<span class="um-dim">—</span>';
        tracksCell = (r.tracks || []).length
          ? `<div class="um-tracks-cell" title="${esc(r.tracks.map(t => t.label).join(' · '))}">
               <span class="um-track-dots">${r.tracks.slice(0,5).map(t => `<span style="background:${categoryColor(t.category)}"></span>`).join('')}</span>
               <span class="um-tracks-cell__count">${r.tracks.length}</span>
             </div>`
          : '<span class="um-dim">—</span>';
      } else {
        const sites = r.sites || [];
        const wheres = Array.from(new Set(sites.map(x => x.where)));
        stepCell = wheres.length
          ? `<div class="um-step-pills">${wheres.slice(0, 4).map(w => {
              const c = '#42b7de';
              return `<span class="um-step-pill" style="--cat:${c};--cat-bg:${hexAlpha(c,0.08)};--cat-fg:${darken(c)}" title="${esc(w)}">${esc(w)}</span>`;
            }).join('')}${wheres.length > 4 ? `<span class="um-dim">+ ${wheres.length - 4}</span>` : ''}</div>`
          : '<span class="um-dim">—</span>';
        const sources = Array.from(new Set(sites.map(x => x.source).filter(Boolean)));
        tracksCell = wheres.length
          ? `<div class="um-tracks-cell" title="${esc(sources.join(' · '))}">
               <span class="um-tracks-cell__count">${wheres.length}</span>
             </div>`
          : '<span class="um-dim">—</span>';
      }

      // SVG templates — count + first two names + remainder.
      const svgCell = (r.svg_templates || []).length
        ? `<div class="um-svg-cell">
             <span class="um-svg-cell__count">${r.svg_templates.length} template${r.svg_templates.length === 1 ? '' : 's'}</span>
             ${r.svg_templates.slice(0,2).map(s => `<code title="${esc(s)}">${esc(s)}</code>`).join('')}
             ${r.svg_templates.length > 2 ? `<span class="um-dim">+ ${r.svg_templates.length - 2} more</span>` : ''}
           </div>`
        : '<span class="um-dim">—</span>';

      return `
        <tr data-status="${r.status}">
          <td>
            <div class="um-cell-symbol">
              <span class="um-cell-symbol__name">${esc(r.symbol)}</span>
              ${r.kind ? `<span class="um-cell-symbol__kind">${esc(r.kind)}</span>` : ''}
            </div>
          </td>
          <td><span class="um-mod">${esc(r.module || '—')}</span></td>
          <td>${stepCell}</td>
          <td>${tracksCell}</td>
          <td>${svgCell}</td>
          <td class="um-num">${r.use_count}</td>
          <td><span class="${sm.cls}">${sm.label}</span></td>
        </tr>`;
    }).join('');
  }

  // ── Color utils for step pills ──
  function hexAlpha(hex, a) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || '');
    if (!m) return 'rgba(66,183,222,' + a + ')';
    return `rgba(${parseInt(m[1],16)},${parseInt(m[2],16)},${parseInt(m[3],16)},${a})`;
  }
  function darken(hex) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || '');
    if (!m) return '#0f172a';
    const f = (v) => Math.max(0, Math.round(parseInt(v,16) * 0.55)).toString(16).padStart(2,'0');
    return `#${f(m[1])}${f(m[2])}${f(m[3])}`;
  }

  function scrollToMatrix() {
    const el = $('#um-matrix'); if (el) el.scrollIntoView({behavior:'smooth', block:'start'});
  }

  // ── Category colors — stable palette aligned with Blanco accents.
  const CATEGORY_COLORS = {
    'Predictive AI · End-to-End':   '#1a3a4f',
    'Pre-Training Bias Auditing':   '#d4a017',
    'Preprocessing':                '#0891b2',
    'In-Processing':                '#7c3aed',
    'Post-Processing':              '#9333ea',
    'Evaluation & Metrics':         '#0d9488',
    'Monitoring':                   '#16a34a',
    'Reporting':                    '#0284c7',
    'Experimentation':              '#dc2626',
    'Operations · CI/CD':           '#ea580c',
    'Rendering · SVG':              '#14b8a6',
    'Library Validation':           '#6b7280',
  };
  function categoryColor(label) {
    return CATEGORY_COLORS[label] || '#42b7de';
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
