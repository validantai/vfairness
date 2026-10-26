/**
 * The two interactive demos on the docs landing page.
 *
 *  1. A real run (#hero-run, in the 'A real run' section): types a short script, streams its output and
 *     draws the selection rate of every subgroup, with the one group the
 *     library withheld shown as withheld, not as a bar.
 *  2. The threshold explorer (#threshold-explorer): move a decision threshold
 *     and read the fairness verdict at that threshold.
 *
 * NOTHING HERE COMPUTES A FAIRNESS NUMBER. Every value is looked up in
 * data/landing-demos.json, which scripts/build_landing_demos.py writes by
 * running vfairness on the UCI Adult test split. The hero's code is the exact
 * text that script executes, and its output lines are that code's stdout.
 * A value the library did not produce (null) is shown as "could not check",
 * never as a zero.
 */
(function () {
    'use strict';

    var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    var src = (document.querySelector('[data-demos-src]') || {}).getAttribute && document.querySelector('[data-demos-src]').getAttribute('data-demos-src');
    if (!src) return;

    function el(tag, cls, text) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text != null) e.textContent = text;
        return e;
    }
    function fmt(v, d) { return v == null ? 'could not check' : v.toFixed(d == null ? 3 : d); }
    function pct(v) { return v == null ? 'could not check' : (v * 100).toFixed(1) + '%'; }
    function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

    // Minimal Python colouring for a short, known snippet.
    function highlight(code) {
        return code.split('\n').map(function (line) {
            var out = esc(line);
            out = out.replace(/(&quot;[^&]*?&quot;|"[^"]*")/g, '<span class="tk-s">$1</span>');
            out = out.replace(/\b(import|from|as)\b/g, '<span class="tk-k">$1</span>');
            out = out.replace(/\b(print)\b/g, '<span class="tk-f">$1</span>');
            out = out.replace(/\b(\d+)\b(?![^<]*<\/span>)/g, '<span class="tk-n">$1</span>');
            return out;
        }).join('\n');
    }

    fetch(src, { cache: 'no-cache' }).then(function (r) { return r.ok ? r.json() : null; }).then(function (data) {
        if (!data) return;
        try { initHero(data); } catch (e) { /* the static fallback stays */ }
        try { initExplorer(data); } catch (e) { /* the section says it could not load */ }
    }).catch(function () {});

    // ---------------------------------------------------------------- hero
    function initHero(data) {
        var root = document.getElementById('hero-run');
        if (!root) return;
        var h = data.hero;
        var codeEl = root.querySelector('.hero-run__code code');
        var outEl = root.querySelector('.hero-run__out');
        var chart = root.querySelector('.hero-run__chart');
        var replay = root.querySelector('.hero-run__replay');
        var ver = root.querySelector('[data-hr="version"]');
        if (ver) ver.textContent = 'vfairness ' + data.vfairness_version;
        root.classList.add('is-live');

        var measured = h.groups.filter(function (g) { return g.state === 'measured'; });
        var max = Math.max.apply(null, measured.map(function (g) { return g.rate; }));
        var scale = Math.ceil(max * 10 + 0.5) / 10; // axis to the next 0.1 above the top bar

        // Chart rows, built once; animation only toggles classes and widths.
        chart.innerHTML = '';
        var axis = el('div', 'hr-axis');
        axis.innerHTML = '<span>selection rate</span><span>0</span><span>' + scale.toFixed(1) + '</span>';
        chart.appendChild(axis);
        h.groups.forEach(function (g) {
            var row = el('div', 'hr-row' + (g.state === 'withheld' ? ' is-withheld' : '') +
                (g.name === h.gap.high ? ' is-high' : '') + (g.name === h.gap.low ? ' is-low' : ''));
            var lab = el('span', 'hr-label');
            lab.innerHTML = esc(g.name) + ' <small>' + g.n.toLocaleString('en-US') + '</small>';
            var track = el('span', 'hr-track');
            if (g.state === 'withheld') {
                track.appendChild(el('span', 'hr-withheld', 'withheld: ' + g.n + ' people, under the floor of ' + h.floor));
            } else {
                var bar = el('span', 'hr-bar');
                bar.style.setProperty('--w', (g.rate / scale * 100).toFixed(2) + '%');
                track.appendChild(bar);
                track.appendChild(el('span', 'hr-val', g.rate.toFixed(3)));
            }
            row.appendChild(lab); row.appendChild(track);
            chart.appendChild(row);
        });
        var gap = el('p', 'hr-gap');
        gap.innerHTML = 'Largest gap <b>' + fmt(h.gap.value) + '</b> (demographic parity difference), limit ' +
            fmt(h.gap.threshold, 2) + '. <span>The withheld group is left out of the verdict, not counted as fair.</span>';
        chart.appendChild(gap);

        // Accessible twin of the chart.
        var tbl = root.querySelector('.hero-run__table');
        if (tbl) {
            tbl.innerHTML = '<caption>Selection rate per subgroup, computed by vfairness</caption><thead><tr><th>Subgroup</th><th>People</th><th>Selection rate</th></tr></thead><tbody>' +
                h.groups.map(function (g) {
                    return '<tr><td>' + esc(g.name) + '</td><td>' + g.n + '</td><td>' +
                        (g.state === 'withheld' ? 'withheld, under the group-size floor of ' + h.floor : g.rate.toFixed(3)) + '</td></tr>';
                }).join('') + '</tbody>';
        }

        var timers = [];
        function later(fn, ms) { timers.push(setTimeout(fn, ms)); }
        function clear() { timers.forEach(clearTimeout); timers = []; }

        function finalState() {
            clear();
            codeEl.innerHTML = highlight(h.code);
            outEl.innerHTML = '<div class="hr-prompt">$ python audit.py</div>' + h.stdout.map(function (l) { return '<div class="hr-line">' + esc(l) + '</div>'; }).join('');
            root.classList.add('is-done');
            chart.classList.add('is-shown');
            if (replay) replay.hidden = false;
        }

        function play() {
            clear();
            root.classList.remove('is-done');
            chart.classList.remove('is-shown');
            outEl.innerHTML = '';
            codeEl.innerHTML = '';
            if (replay) replay.hidden = true;
            var text = h.code, i = 0, step = 3;
            function type() {
                i = Math.min(text.length, i + step);
                codeEl.innerHTML = highlight(text.slice(0, i)) + '<span class="hr-caret"></span>';
                if (i < text.length) later(type, 16);
                else later(run, 380);
            }
            function run() {
                codeEl.innerHTML = highlight(text);
                var p = el('div', 'hr-prompt', '$ python audit.py');
                outEl.appendChild(p);
                var line = el('div', 'hr-line');
                outEl.appendChild(line);
                var words = h.stdout.join('\n').split(' '), w = 0;
                function stream() {
                    w = Math.min(words.length, w + 3);
                    line.textContent = words.slice(0, w).join(' ');
                    if (w < words.length) later(stream, 28);
                    else later(function () { chart.classList.add('is-shown'); root.classList.add('is-done'); later(function () { if (replay) replay.hidden = false; }, 900); }, 250);
                }
                later(stream, 420);
            }
            type();
        }

        if (replay) replay.addEventListener('click', function () { reduce ? finalState() : play(); });
        if (reduce) { finalState(); return; }
        // It sits below the fold now, so it plays when it comes into view,
        // once; Replay runs it again.
        finalState();
        root.classList.remove('is-done'); chart.classList.remove('is-shown');
        if ('IntersectionObserver' in window) {
            var io = new IntersectionObserver(function (es) {
                if (es.some(function (e) { return e.isIntersecting; })) { io.disconnect(); play(); }
            }, { threshold: 0, rootMargin: '0px 0px -25% 0px' });
            io.observe(root);
        } else { play(); }
    }

    // ------------------------------------------------------------ explorer
    function initExplorer(data) {
        var root = document.getElementById('threshold-explorer');
        if (!root) return;
        var T = data.threshold;
        var G = T.groups; // ["Female", "Male"]
        var q = function (s) { return root.querySelector(s); };
        var uni = q('#te-t'), sf = q('#te-tf'), sm = q('#te-tm');
        var modeBtns = root.querySelectorAll('[data-te-mode]');
        var fixBtn = q('.te-fix');
        var svg = q('.te-plot svg');
        var mode = 'one', fixed = false;

        uni.max = T.uniform.length - 1;
        var mid = T.uniform.findIndex(function (r) { return Math.abs(r.t - 0.5) < 1e-6; });
        uni.value = mid >= 0 ? mid : Math.floor(T.uniform.length / 2);
        [sf, sm].forEach(function (s) { s.max = T.split_grid.length - 1; s.value = T.split_grid.indexOf(0.5); });

        // Histograms: drawn once, per group, as shares of the group.
        var W = 800, rowH = 150, top = 20, gapY = 40;
        var H = top + G.length * (rowH + gapY);
        svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
        var edges = T.hist.edges, nb = edges.length - 1, bw = W / nb;
        var peak = 0;
        // Square-root height scale: most people score near 0, and a linear
        // scale flattens the high-score tail where the threshold matters.
        G.forEach(function (g) { for (var i = 0; i < nb; i++) peak = Math.max(peak, T.hist[g].pos[i] + T.hist[g].neg[i]); });
        var sq = function (v) { return Math.sqrt(v / peak) * rowH; };
        var parts = [];
        G.forEach(function (g, gi) {
            var y0 = top + gi * (rowH + gapY) + rowH;
            parts.push('<text class="te-glabel" x="0" y="' + (y0 - rowH - 4) + '">' + g + ' &middot; ' + T.n[g].toLocaleString('en-US') + ' people</text>');
            parts.push('<rect class="te-shade" data-g="' + gi + '" x="0" y="' + (y0 - rowH) + '" width="0" height="' + rowH + '"></rect>');
            for (var i = 0; i < nb; i++) {
                var tot = sq(T.hist[g].pos[i] + T.hist[g].neg[i]), share = (T.hist[g].pos[i] + T.hist[g].neg[i]) || 1;
                var hp = tot * T.hist[g].pos[i] / share, hn = tot - hp;
                parts.push('<rect class="te-neg" x="' + (i * bw + 0.5).toFixed(1) + '" y="' + (y0 - hn - hp).toFixed(1) + '" width="' + (bw - 1).toFixed(1) + '" height="' + hn.toFixed(1) + '"></rect>');
                parts.push('<rect class="te-pos" x="' + (i * bw + 0.5).toFixed(1) + '" y="' + (y0 - hp).toFixed(1) + '" width="' + (bw - 1).toFixed(1) + '" height="' + hp.toFixed(1) + '"></rect>');
            }
            parts.push('<line class="te-base" x1="0" x2="' + W + '" y1="' + y0 + '" y2="' + y0 + '"></line>');
            parts.push('<line class="te-line" data-g="' + gi + '" x1="0" x2="0" y1="' + (y0 - rowH - 2) + '" y2="' + (y0 + 4) + '"></line>');
        });
        parts.push('<text class="te-tick" x="0" y="' + (H - 2) + '">score 0</text><text class="te-tick" x="' + W + '" y="' + (H - 2) + '" text-anchor="end">1</text>');
        svg.innerHTML = parts.join('');

        function current() {
            if (fixed) return { row: T.optimizer, th: [T.optimizer.thresholds[G[0]], T.optimizer.thresholds[G[1]]] };
            if (mode === 'one') { var r = T.uniform[+uni.value]; return { row: r, th: [r.t, r.t] }; }
            var tf = T.split_grid[+sf.value], tm = T.split_grid[+sm.value];
            return { row: T.split[tf.toFixed(2) + '|' + tm.toFixed(2)], th: [tf, tm] };
        }

        function state(v, lim) { return v == null ? 'na' : (v <= lim ? 'ok' : 'over'); }

        function render() {
            var c = current(), r = c.row;
            G.forEach(function (g, gi) {
                var x = c.th[gi] * W;
                var line = svg.querySelector('.te-line[data-g="' + gi + '"]');
                line.setAttribute('x1', x); line.setAttribute('x2', x);
                var sh = svg.querySelector('.te-shade[data-g="' + gi + '"]');
                sh.setAttribute('x', x); sh.setAttribute('width', Math.max(0, W - x));
            });
            q('[data-te="t"]').textContent = c.th[0].toFixed(fixed ? 3 : 2);
            q('[data-te="tf"]').textContent = c.th[0].toFixed(fixed ? 3 : 2);
            q('[data-te="tm"]').textContent = c.th[1].toFixed(fixed ? 3 : 2);
            G.forEach(function (g) {
                q('[data-te="sel-' + g + '"]').textContent = pct(r.sel[g]);
                q('[data-te="bar-' + g + '"]').style.width = (r.sel[g] == null ? 0 : r.sel[g] * 100) + '%';
            });
            var dpS = state(r.dp, T.thresholds.dp), eoS = state(r.eo, T.thresholds.eo);
            var dpT = q('[data-te-tile="dp"]'), eoT = q('[data-te-tile="eo"]');
            dpT.setAttribute('data-state', dpS); eoT.setAttribute('data-state', eoS);
            q('[data-te="dp"]').textContent = fmt(r.dp);
            q('[data-te="eo"]').textContent = fmt(r.eo);
            q('[data-te="dp-s"]').textContent = dpS === 'ok' ? 'within limit' : dpS === 'over' ? 'over limit' : 'could not check';
            q('[data-te="eo-s"]').textContent = eoS === 'ok' ? 'within limit' : eoS === 'over' ? 'over limit' : 'could not check';
            q('[data-te="acc"]').textContent = pct(r.acc);
            var total = r.passed + r.failed + r.na;
            q('[data-te="verdict"]').textContent = r.passed + ' of ' + total + ' metrics within thresholds' + (r.na ? ', ' + r.na + ' could not be checked' : '');
            var note = q('.te-note');
            if (fixed) {
                note.innerHTML = '<b>GroupThresholdOptimizer</b> (demographic parity, tolerance ' + T.optimizer.tolerance + ') chose ' +
                    G[0] + ' ' + T.optimizer.thresholds[G[0]].toFixed(3) + ' and ' + G[1] + ' ' + T.optimizer.thresholds[G[1]].toFixed(3) +
                    '. Parity is met, and the equal-opportunity gap ' + (eoS === 'over' ? 'is now over its limit' : 'stays within its limit') +
                    ': <a href="concepts/index.html#impossibility">fairness criteria can conflict</a>.';
            } else if (dpS === 'ok' && eoS === 'over') {
                note.innerHTML = 'Parity is met here, but equal opportunity is not: closing one gap opened another. <a href="concepts/index.html#impossibility">Fairness criteria can conflict.</a>';
            } else if (dpS === 'over' && eoS === 'ok') {
                note.innerHTML = 'Equal opportunity is within its limit, demographic parity is not. Which one matters is a decision about the use case, not a setting.';
            } else if (dpS === 'ok' && eoS === 'ok') {
                note.innerHTML = 'Both gaps are within their limits at this threshold. Check what it costs: accuracy is ' + pct(r.acc) + '.';
            } else {
                note.textContent = 'Both gaps are over their limits at this threshold.';
            }
            uni.setAttribute('aria-valuetext', 'threshold ' + T.uniform[+uni.value].t.toFixed(2) + ': ' + q('[data-te="verdict"]').textContent);
        }

        function setMode(m) {
            mode = m;
            root.setAttribute('data-mode', m);
            modeBtns.forEach(function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-te-mode') === m ? 'true' : 'false'); });
        }
        modeBtns.forEach(function (b) { b.addEventListener('click', function () { fixed = false; root.classList.remove('is-fixed'); setMode(b.getAttribute('data-te-mode')); render(); }); });
        [uni, sf, sm].forEach(function (s) { s.addEventListener('input', function () { fixed = false; root.classList.remove('is-fixed'); render(); }); });
        fixBtn.addEventListener('click', function () {
            setMode('split');
            var near = function (v) { var b = 0; T.split_grid.forEach(function (x, i) { if (Math.abs(x - v) < Math.abs(T.split_grid[b] - v)) b = i; }); return b; };
            sf.value = near(T.optimizer.thresholds[G[0]]); sm.value = near(T.optimizer.thresholds[G[1]]);
            fixed = true; root.classList.add('is-fixed');
            render();
        });
        root.classList.add('is-live');
        setMode('one');
        render();
    }
})();
