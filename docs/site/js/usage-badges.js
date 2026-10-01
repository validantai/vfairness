/* Inline Navigator-usage badges.
 *
 * Loaded by the API Reference page. For every heading (h2/h3/h4) that contains
 * a <code>SymbolName</code> matching a row in usage_map.json, this script
 * appends a small badge row indicating which Navigator step(s) call it,
 * or marks it as library-only / unused.
 *
 * Falls back silently if the manifest is missing.
 */
(function () {
  'use strict';

  // Resolve data path relative to current page (works from /api-reference/, /sample-assessment/, etc.)
  const DATA_URL = new URL('../data/usage_map.json', document.baseURI).toString();

  fetch(DATA_URL, { cache: 'no-cache' })
    .then(r => r.ok ? r.json() : null)
    .then(data => { if (data) decorate(data); })
    .catch(() => { /* silent */ });

  function decorate(data) {
    const bySymbol = {};
    data.rows.forEach(r => { bySymbol[r.symbol] = r; });
    const usageMapHref = relativeTo('../usage-map/index.html');

    const headings = document.querySelectorAll('h2, h3, h4');
    const seen = new Set();
    headings.forEach(h => {
      // Find the first <code> child that matches a known symbol.
      const codes = h.querySelectorAll('code');
      for (const c of codes) {
        const name = c.textContent.trim();
        const row = bySymbol[name];
        if (!row || seen.has(name)) continue;
        seen.add(name);
        appendBadges(h, row, usageMapHref);
        break;
      }
    });
  }

  function appendBadges(h, row, mapHref) {
    const wrap = document.createElement('span');
    wrap.className = 'um-badges';

    if (row.status === 'active') {
      row.steps.forEach(s => {
        const a = document.createElement('a');
        a.className = 'um-badge um-badge--active';
        a.href = mapHref + '?step=' + encodeURIComponent(s.id);
        a.title = 'Used in ' + s.label + ' (' + row.use_count + ' call site' + (row.use_count === 1 ? '' : 's') + ')';
        a.textContent = s.label;
        wrap.appendChild(a);
      });
    } else if (row.status === 'library-only') {
      const s = document.createElement('span');
      s.className = 'um-badge um-badge--lib';
      s.textContent = 'Library-only';
      s.title = 'Exposed by vfairness; not called by the Navigator.';
      wrap.appendChild(s);
    } else {
      const s = document.createElement('span');
      s.className = 'um-badge um-badge--unused';
      s.textContent = 'Unused';
      s.title = 'No call sites detected in the Navigator.';
      wrap.appendChild(s);
    }
    h.appendChild(document.createTextNode(' '));
    h.appendChild(wrap);
  }

  function relativeTo(href) {
    try { return new URL(href, document.baseURI).toString(); }
    catch (e) { return href; }
  }
})();
