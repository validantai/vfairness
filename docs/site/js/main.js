/**
 * vfairness Documentation - Main JavaScript
 * validant.ai - Systems for Trust & Agency
 */

(function() {
    'use strict';

    // ============================================================
    // DOM Ready
    // ============================================================

    // ── Helper: schedule work off the critical path ──
    var _deferQueue = [];
    function deferInit(fn) { _deferQueue.push(fn); }
    function flushDeferred() {
        var ric = window.requestIdleCallback || function(cb) { setTimeout(cb, 1); };
        function next() {
            if (!_deferQueue.length) return;
            ric(function() {
                var fn = _deferQueue.shift();
                if (fn) fn();
                next();
            });
        }
        next();
    }

    document.addEventListener('DOMContentLoaded', function() {
        // ── Critical path — must be interactive immediately ──
        initDefaultTheme();
        initSkipLink();
        initThemeToggle();
        initNavigation();
        initMobileMenu();
        initMobileSearch();
        initApiPackageView();
        initTouchWording();

        // ── Deferred — heavy / non-essential, won't block the main thread ──
        deferInit(initSearch);
        deferInit(initCodeCopy);
        deferInit(initTabs);
        deferInit(initTableOfContents);
        deferInit(initMermaid);
        deferInit(initSmoothScroll);
        deferInit(initHeadingAnchors);
        deferInit(initSidebarFilter);
        deferInit(initSidebarScrollspy);
        deferInit(initMobileSections);
        deferInit(initPageMeta);
        deferInit(initGuideOverview);
        deferInit(initLogoFade);
        flushDeferred();
    });

    // ============================================================
    // Navigation
    // ============================================================

    // ============================================================
    // Logo greyscale fade
    // ============================================================

    function initLogoFade() {
        const logoName = document.querySelector('.logo-section .lib-name');
        if (!logoName) return;

        logoName.addEventListener('mouseenter', () => {
            logoName.classList.add('logo-hovered');
        });

        logoName.addEventListener('mouseleave', () => {
            logoName.classList.remove('logo-hovered');
            // Restart the desaturation keyframes from 0 %
            logoName.style.animation = 'none';
            void logoName.offsetWidth;          // force reflow
            logoName.style.animation = '';
        });
    }

    function initNavigation() {
        var currentPath = window.location.pathname;

        // --- Active-state: highlight dropdown child items + parent groups ---
        document.querySelectorAll('.nav-dropdown__item').forEach(function(link) {
            var href = link.getAttribute('href');
            if (href && (currentPath.endsWith(href) || currentPath.includes(href.replace('index.html', '').replace(/\.\.\//g, '')))) {
                link.classList.add('active');
                var parent = link.closest('.nav-item');
                if (parent) parent.classList.add('nav-item--active');
            }
        });

        // Highlight direct nav links (Getting Started)
        document.querySelectorAll('.nav-link:not(.nav-link--trigger)').forEach(function(link) {
            var href = link.getAttribute('href');
            if (href && (currentPath.endsWith(href) || currentPath.includes(href.replace('index.html', '').replace(/\.\.\//g, '')))) {
                link.classList.add('active');
            }
        });

        // Also mark sidebar links active
        document.querySelectorAll('.sidebar-nav a').forEach(function(link) {
            var href = link.getAttribute('href');
            if (href && (currentPath.endsWith(href) || currentPath.includes(href.replace('index.html', '')))) {
                link.classList.add('active');
            }
        });

        // --- Sticky header shadow ---
        var header = document.querySelector('.site-header');
        if (header) {
            window.addEventListener('scroll', function() {
                header.style.boxShadow = window.scrollY > 10
                    ? '0 2px 8px rgba(0,0,0,0.1)'
                    : 'none';
            });
        }

        // --- Dropdown hover intent ---
        var DELAY_OPEN = 80;
        var DELAY_CLOSE = 220;
        var dropdownItems = document.querySelectorAll('.nav-item--has-dropdown');

        dropdownItems.forEach(function(item) {
            var openTimer = null;
            var closeTimer = null;
            var trigger = item.querySelector('.nav-link--trigger');

            function openDrop() {
                clearTimeout(closeTimer);
                dropdownItems.forEach(function(other) {
                    if (other !== item) closeDrop.call(null, other);
                });
                item.classList.add('nav-item--open');
                if (trigger) trigger.setAttribute('aria-expanded', 'true');
            }

            function closeDrop(el) {
                var target = el || item;
                clearTimeout(openTimer);
                target.classList.remove('nav-item--open');
                var t = target.querySelector('.nav-link--trigger');
                if (t) t.setAttribute('aria-expanded', 'false');
            }

            item.addEventListener('mouseenter', function() {
                clearTimeout(closeTimer);
                openTimer = setTimeout(openDrop, DELAY_OPEN);
            });

            item.addEventListener('mouseleave', function() {
                clearTimeout(openTimer);
                closeTimer = setTimeout(function() { closeDrop(item); }, DELAY_CLOSE);
            });

            if (trigger) {
                trigger.addEventListener('click', function(e) {
                    e.preventDefault();
                    item.classList.contains('nav-item--open') ? closeDrop(item) : openDrop();
                });

                trigger.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        item.classList.contains('nav-item--open') ? closeDrop(item) : openDrop();
                        if (item.classList.contains('nav-item--open')) {
                            var first = item.querySelector('.nav-dropdown__item');
                            if (first) setTimeout(function() { first.focus(); }, 50);
                        }
                    }
                    if (e.key === 'Escape') {
                        closeDrop(item);
                        trigger.focus();
                    }
                });
            }

            var dropdown = item.querySelector('.nav-dropdown');
            if (dropdown) {
                dropdown.addEventListener('keydown', function(e) {
                    var links = Array.from(dropdown.querySelectorAll('.nav-dropdown__item'));
                    var idx = links.indexOf(document.activeElement);
                    if (e.key === 'ArrowDown') {
                        e.preventDefault();
                        (links[idx + 1] || links[0]).focus();
                    }
                    if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        (links[idx - 1] || links[links.length - 1]).focus();
                    }
                    if (e.key === 'Escape') {
                        closeDrop(item);
                        if (trigger) trigger.focus();
                    }
                });
            }
        });

        // Close all on outside click
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.nav-item--has-dropdown')) {
                dropdownItems.forEach(function(item) {
                    item.classList.remove('nav-item--open');
                    var t = item.querySelector('.nav-link--trigger');
                    if (t) t.setAttribute('aria-expanded', 'false');
                });
            }
        });
    }

    // ============================================================
    // Default Gradient Theme
    // ============================================================

    function initDefaultTheme() {
        const body = document.body;
        if (!body) return;

        const legacyThemes = ['theme-logo', 'theme-fresh', 'theme-vivid'];
        legacyThemes.forEach(themeClass => body.classList.remove(themeClass));
        body.classList.add('theme-sky');

        try {
            localStorage.setItem('vf-gradient-theme', 'theme-sky');
        } catch (err) {
            // Ignore storage restrictions (private mode / disabled storage).
        }
    }

    // ============================================================
    // Skip link: the first focusable element on every page
    // ============================================================

    function initSkipLink() {
        if (document.querySelector('.skip-link')) return;
        var target = document.querySelector('main') || document.querySelector('.main-content');
        if (!target) return;
        if (!target.id) target.id = 'main-content';
        var a = document.createElement('a');
        a.className = 'skip-link';
        a.href = '#' + target.id;
        a.textContent = 'Skip to content';
        document.body.insertBefore(a, document.body.firstChild);
    }

    // ============================================================
    // Theme: system / light / dark
    // The <head> of every page sets html[data-theme] before first paint
    // from localStorage 'vf-theme'; this adds the control and keeps the
    // two attributes (resolved theme, chosen mode) in step.
    // ============================================================

    var THEME_KEY = 'vf-theme';
    var THEME_ORDER = ['system', 'light', 'dark'];
    var THEME_LABEL = { system: 'Theme: system', light: 'Theme: light', dark: 'Theme: dark' };

    function readThemeMode() {
        try { return localStorage.getItem(THEME_KEY) || 'system'; } catch (e) { return 'system'; }
    }

    function applyTheme(mode) {
        var mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
        var dark = mode === 'dark' || (mode === 'system' && mq && mq.matches);
        var root = document.documentElement;
        root.setAttribute('data-theme', dark ? 'dark' : 'light');
        root.setAttribute('data-theme-mode', mode);
        // The colour logo is ink-on-light; dark mode swaps in the white one.
        var logos = document.querySelectorAll('img.logo-full, img.brand-logo');
        for (var i = 0; i < logos.length; i++) {
            var img = logos[i];
            if (!img.dataset.lightSrc) img.dataset.lightSrc = img.getAttribute('src');
            var light = img.dataset.lightSrc;
            var darkSrc = light.replace(/VALIDANT_AI_logo_v1\.0_fancy_green\.png$|VALIDANT_AI_logo_web_v1_1\.svg$/, 'VALIDANT_AI_logo_web_white_v1_1.svg');
            img.setAttribute('src', dark ? darkSrc : light);
        }
        var toggles = document.querySelectorAll('.theme-toggle');
        for (var j = 0; j < toggles.length; j++) {
            toggles[j].setAttribute('data-mode', mode);
            toggles[j].setAttribute('aria-label', THEME_LABEL[mode] + '. Activate to change.');
            toggles[j].setAttribute('title', THEME_LABEL[mode]);
            var lbl = toggles[j].querySelector('.tt-label');
            if (lbl) lbl.textContent = THEME_LABEL[mode];
        }
    }

    function initThemeToggle() {
        var svg =
            '<svg class="tt-auto" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="M12 3.5v17" /><path d="M12 3.5a8.5 8.5 0 0 1 0 17z" fill="currentColor" stroke="none"/></svg>' +
            '<svg class="tt-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>' +
            '<svg class="tt-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
        function make(withLabel) {
            var b = document.createElement('button');
            b.type = 'button';
            b.className = 'theme-toggle';
            b.innerHTML = svg + (withLabel ? '<span class="tt-label"></span>' : '');
            b.addEventListener('click', function () {
                var cur = readThemeMode();
                var next = THEME_ORDER[(THEME_ORDER.indexOf(cur) + 1) % THEME_ORDER.length];
                try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
                applyTheme(next);
            });
            return b;
        }
        var cta = document.querySelector('.nav-cta');
        if (cta && !cta.querySelector('.theme-toggle')) {
            var gh = cta.querySelector('a.btn-secondary');
            cta.insertBefore(make(false), gh || null);
        }
        var mfoot = document.querySelector('.mobile-nav-footer');
        if (mfoot && !mfoot.querySelector('.theme-toggle')) mfoot.appendChild(make(true));
        applyTheme(readThemeMode());
        if (window.matchMedia) {
            var mq = window.matchMedia('(prefers-color-scheme: dark)');
            var onChange = function () { if (readThemeMode() === 'system') applyTheme('system'); };
            if (mq.addEventListener) mq.addEventListener('change', onChange);
            else if (mq.addListener) mq.addListener(onChange);
        }
    }

    // ============================================================
    // Touch devices: "Hover or focus for ..." becomes "Tap for ..."
    // ============================================================

    function initTouchWording() {
        if (!window.matchMedia || !window.matchMedia('(hover: none)').matches) return;
        var els = document.querySelectorAll('.hero-stat-open');
        for (var i = 0; i < els.length; i++) {
            var n = els[i].firstChild;
            if (n && n.nodeType === 3) n.nodeValue = n.nodeValue.replace(/^\s*Hover or focus for/, 'Tap for');
        }
        // On a phone an open stat panel covers most of the screen, so "tap
        // elsewhere" has almost nowhere to land. Give every panel a Close
        // button. Blurring the tile fires the page's own focusout handler,
        // which is what hides the panel; the class is cleared as a fallback.
        var panels = document.querySelectorAll('.capability-panel');
        for (var j = 0; j < panels.length; j++) {
            (function (panel) {
                if (panel.querySelector('.capability-panel__close')) return;
                var btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'capability-panel__close';
                btn.textContent = 'Close';
                btn.addEventListener('click', function (e) {
                    e.preventDefault();
                    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
                    panel.classList.remove('is-open');
                });
                panel.insertBefore(btn, panel.firstChild);
            })(panels[j]);
        }
    }

    // ============================================================
    // Reading order of the documentation, used for the previous / next links
    // at the foot of each page and for grouping search results. It follows the
    // header navigation: Getting Started, API Reference, the Learn menu, Our
    // Offering, then the Explore menu.
    // ============================================================

    var DOC_ORDER = [
        { u: 'getting-started/', t: 'Getting Started', g: 'Start' },
        { u: 'api-reference/', t: 'API Reference', g: 'Reference' },
        { u: 'test-objects/', t: 'Find a Test Object', g: 'Learn' },
        { u: 'business-guide/', t: 'Business Guide', g: 'Learn' },
        { u: 'sample-assessment/', t: 'Sample Assessment: Tabular', g: 'Learn' },
        { u: 'sample-assessment/cv-audit.html', t: 'Sample Assessment: Computer Vision', g: 'Learn' },
        { u: 'sample-assessment/llm-audit.html', t: 'Sample Assessment: LLM', g: 'Learn' },
        { u: 'sample-assessment/ranking-audit.html', t: 'Sample Assessment: Ranking', g: 'Learn' },
        { u: 'concepts/', t: 'Concepts', g: 'Learn' },
        { u: 'concepts/references.html', t: 'Academic References', g: 'Learn' },
        { u: 'quality-and-hardening/', t: 'Quality & Hardening', g: 'Learn' },
        { u: 'release-pipeline/', t: 'Releasing', g: 'Learn' },
        { u: 'concepts/changelog.html', t: 'Changelog', g: 'Learn' },
        { u: 'our-offering/', t: 'Our Offering', g: 'Offering' },
        { u: 'svg-gallery/', t: 'SVG Gallery', g: 'Explore' },
        { u: 'workflow-integration/', t: 'Workflow Integration', g: 'Explore' },
        { u: 'usage-map/', t: 'Library Usage Map', g: 'Explore' },
        { u: 'llm-testing/', t: 'LLM Fairness Testing', g: 'Explore' },
        { u: 'agent-testing/', t: 'Agent Fairness Testing', g: 'Explore' },
        { u: 'multi-agent-testing/', t: 'Multi-Agent Testing', g: 'Explore' },
        { u: 'explainability-and-xai/', t: 'Explainability & XAI', g: 'Explore' },
        { u: 'training-time-interventions/', t: 'Training-Time Interventions', g: 'Explore' }
    ];

    function siteRoot() {
        var css = document.querySelector('link[href*="css/styles.css"]');
        var href = css ? css.getAttribute('href') : 'css/styles.css';
        if (href.charAt(0) === '/') return '/';
        return href.replace(/css\/styles\.css.*$/, '');
    }

    // This page's path relative to the site root, normalised like DOC_ORDER
    // ("concepts/", "concepts/references.html"). The root is resolved from the
    // page's own stylesheet link, so it is right at any depth and on any host.
    function pageKey() {
        var rootPath = new URL(siteRoot() || './', location.href).pathname;
        var rel = location.pathname.indexOf(rootPath) === 0 ? location.pathname.slice(rootPath.length) : location.pathname.replace(/^\//, '');
        rel = rel.replace(/index\.html$/, '');
        // Cloudflare serves /x/y.html as /x/y: map back to the file name.
        if (rel && !rel.endsWith('/') && !/\.html$/.test(rel)) {
            var asFile = rel + '.html';
            for (var i = 0; i < DOC_ORDER.length; i++) { if (DOC_ORDER[i].u === asFile) return asFile; }
            rel += '/';
        }
        return rel;
    }

    function groupFor(path) {
        var p = path.replace(/#.*$/, '').replace(/index\.html$/, '');
        for (var i = 0; i < DOC_ORDER.length; i++) {
            if (p === DOC_ORDER[i].u || (DOC_ORDER[i].u.endsWith('/') && p.indexOf(DOC_ORDER[i].u) === 0 && DOC_ORDER[i].u !== 'sample-assessment/')) return DOC_ORDER[i].g;
        }
        if (p.indexOf('sample-assessment/') === 0) return 'Learn';
        return 'More';
    }

    // ============================================================
    // Heading links: a small link mark on every section heading. Clicking it
    // puts the address of that exact section in the URL and the clipboard.
    // ============================================================

    function initHeadingAnchors() {
        var main = document.querySelector('.main-content') || document.querySelector('main');
        if (!main) return;
        var used = {};
        main.querySelectorAll('h2, h3').forEach(function (h) {
            if (h.querySelector('.heading-anchor') || h.closest('.mermaid, .callout, .card, .module-card, .compare-col, sponsor-wall')) return;
            var id = h.id;
            if (!id) {
                // The first heading of a section takes the section's own id,
                // which is what the sidebar and other pages already link to.
                var sec = h.closest('section[id]');
                if (sec && !used[sec.id] && sec.querySelector(h.tagName.toLowerCase()) === h) id = sec.id;
            }
            if (!id) {
                var base = h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60);
                if (!base) return;
                id = base;
                for (var n = 2; document.getElementById(id) || used[id]; n++) id = base + '-' + n;
                h.id = id;
            }
            used[id] = true;
            var a = document.createElement('a');
            a.className = 'heading-anchor';
            a.href = '#' + id;
            a.setAttribute('aria-label', 'Copy link to this section');
            a.innerHTML = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M6.5 9.5l3-3M7 4.5l1-1a2.5 2.5 0 0 1 3.5 3.5l-1 1M9 11.5l-1 1A2.5 2.5 0 0 1 4.5 9l1-1" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
            a.addEventListener('click', function (e) {
                e.preventDefault();
                var url = location.href.replace(/#.*$/, '') + '#' + id;
                try { history.replaceState(null, '', '#' + id); } catch (err) {}
                var el = document.getElementById(id);
                if (el) el.scrollIntoView();
                var done = function () { a.classList.add('is-copied'); setTimeout(function () { a.classList.remove('is-copied'); }, 1400); };
                if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(url).then(done, function () {});
            });
            h.appendChild(a);
        });
    }

    // ============================================================
    // Sidebar filter: a box at the top of long sidebars (API Reference has
    // 183 links) that narrows the list to what matches as you type.
    // ============================================================

    function initSidebarFilter() {
        var sidebar = document.querySelector('.docs-layout .sidebar');
        if (!sidebar) return;
        var links = sidebar.querySelectorAll('.sidebar-nav a');
        if (links.length < 40) return;
        var box = document.createElement('div');
        box.className = 'sidebar-filter';
        box.innerHTML = '<input type="search" id="sidebar-filter-input" placeholder="Filter this list" aria-label="Filter the sidebar" autocomplete="off"><span class="sidebar-filter__count" aria-live="polite"></span>';
        sidebar.insertBefore(box, sidebar.firstChild);
        var input = box.querySelector('input'), count = box.querySelector('.sidebar-filter__count');
        var sections = sidebar.querySelectorAll('.sidebar-section');
        input.addEventListener('input', function () {
            var q = input.value.trim().toLowerCase();
            var shown = 0;
            sections.forEach(function (sec) {
                var title = sec.querySelector('.sidebar-title');
                var titleHit = q && title && title.textContent.toLowerCase().indexOf(q) >= 0;
                var any = false;
                sec.querySelectorAll('.sidebar-nav li').forEach(function (li) {
                    var hit = !q || titleHit || li.textContent.toLowerCase().indexOf(q) >= 0;
                    li.hidden = !hit;
                    if (hit) { any = true; shown++; }
                });
                sec.hidden = !!q && !any;
                sec.classList.toggle('is-filtering', !!q);
            });
            count.textContent = q ? shown + ' match' + (shown === 1 ? '' : 'es') : '';
        });
    }

    // ============================================================
    // Sidebar follows the reader: the link for the section on screen is
    // marked, and the sidebar scrolls to keep it in view.
    // ============================================================

    function initSidebarScrollspy() {
        var sidebar = document.querySelector('.docs-layout .sidebar');
        if (!sidebar) return;
        var pairs = [];
        sidebar.querySelectorAll('.sidebar-nav a[href^="#"]').forEach(function (a) {
            var id = decodeURIComponent(a.getAttribute('href').slice(1));
            var el = id && document.getElementById(id);
            if (el) pairs.push({ a: a, el: el });
        });
        if (pairs.length < 3) return;
        var current = null, ticking = false;
        function mark() {
            ticking = false;
            var line = window.scrollY + 140, best = null, bestTop = -Infinity;
            for (var i = 0; i < pairs.length; i++) {
                if (pairs[i].el.closest('[hidden]')) continue;   // in a closed API package
                var top = pairs[i].el.getBoundingClientRect().top + window.scrollY;
                if (top <= line && top > bestTop) { best = pairs[i]; bestTop = top; }
            }
            if (!best || best === current) return;
            if (current) current.a.classList.remove('is-current');
            best.a.classList.add('is-current');
            current = best;
            var sec = best.a.closest('.sidebar-section');
            if (sec && !sec.classList.contains('expanded') && !sec.classList.contains('is-filtering')) sec.classList.add('expanded');
            var r = best.a.getBoundingClientRect(), sr = sidebar.getBoundingClientRect();
            if (r.top < sr.top + 40 || r.bottom > sr.bottom - 40) sidebar.scrollTop += r.top - sr.top - sr.height / 3;
        }
        window.addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(mark); } }, { passive: true });
        mark();
    }

    // ============================================================
    // Sections on phones and tablets: the sidebar is hidden below 1024px, so
    // long pages get a "Sections" drop-down under their title.
    // ============================================================

    function initMobileSections() {
        if (!window.matchMedia || !window.matchMedia('(max-width: 1024px)').matches) return;
        var sidebar = document.querySelector('.docs-layout .sidebar');
        var main = document.querySelector('.docs-layout .main-content');
        if (!sidebar || !main || main.querySelector('.mobile-sections')) return;
        var links = sidebar.querySelectorAll('.sidebar-nav a[href]');
        if (links.length < 4) return;
        var d = document.createElement('details');
        d.className = 'mobile-sections';
        var html = '<summary>Sections <span>' + links.length + '</span></summary><div class="mobile-sections__body">';
        sidebar.querySelectorAll('.sidebar-section').forEach(function (sec) {
            var t = sec.querySelector('.sidebar-title');
            var ls = sec.querySelectorAll('.sidebar-nav a[href]');
            if (!ls.length) return;
            var tc = t ? t.cloneNode(true) : null;
            if (tc) tc.querySelectorAll('.sidebar-toggle-indicator, [aria-hidden="true"]').forEach(function (x) { x.remove(); });
            html += '<p class="mobile-sections__group">' + (tc ? tc.innerHTML : '') + '</p><ul>';
            ls.forEach(function (a) { html += '<li><a href="' + a.getAttribute('href') + '">' + a.textContent.trim().replace(/[<>&]/g, '') + '</a></li>'; });
            html += '</ul>';
        });
        d.innerHTML = html + '</div>';
        var h1 = main.querySelector('h1');
        var after = h1 && h1.nextElementSibling && h1.nextElementSibling.tagName === 'P' ? h1.nextElementSibling : h1;
        if (after && after.parentNode) after.parentNode.insertBefore(d, after.nextSibling);
        else main.insertBefore(d, main.firstChild);
        d.addEventListener('click', function (e) { if (e.target.closest('a')) d.open = false; });
    }

    // ============================================================
    // Page foot: when the page last changed, a way to report a problem, and
    // the previous / next page in the reading order. The date comes from
    // data/page-dates.json, generated from git history at deploy time.
    // ============================================================

    function initPageMeta() {
        var main = document.querySelector('.docs-layout .main-content .content-container') || document.querySelector('.docs-layout .main-content') || document.querySelector('main.page-wrapper');
        if (!main || main.querySelector('.page-foot')) return;
        if (!document.querySelector('.site-header')) return;
        var key = pageKey();
        var idx = -1;
        for (var i = 0; i < DOC_ORDER.length; i++) { if (DOC_ORDER[i].u === key) { idx = i; break; } }
        var root = siteRoot();
        var title = (document.querySelector('main h1, h1') || {}).textContent || document.title;
        var issue = 'https://github.com/validantai/vfairness/issues/new?title=' +
            encodeURIComponent('Docs: ' + title.trim()) + '&body=' + encodeURIComponent('Page: ' + location.href.replace(/#.*$/, '') + '\n\nWhat is wrong or unclear:\n');
        var foot = document.createElement('footer');
        foot.className = 'page-foot';
        var html = '';
        if (idx >= 0) {
            var prev = DOC_ORDER[idx - 1], next = DOC_ORDER[idx + 1];
            html += '<nav class="page-pager" aria-label="Previous and next page">' +
                (prev ? '<a class="page-pager__prev" href="' + root + prev.u + '"><span><i aria-hidden="true">\u2190</i>Previous</span><strong>' + prev.t + '</strong></a>' : '<span></span>') +
                (next ? '<a class="page-pager__next" href="' + root + next.u + '"><span>Next<i aria-hidden="true">\u2192</i></span><strong>' + next.t + '</strong></a>' : '<span></span>') +
                '</nav>';
        }
        html += '<div class="page-foot__meta"><span class="page-foot__date" hidden></span>' +
            '<a href="' + issue + '" target="_blank" rel="noopener">Report a problem with this page</a></div>';
        foot.innerHTML = html;
        main.appendChild(foot);
        fetch(root + 'data/page-dates.json', { cache: 'no-cache' })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (d) {
                if (!d || !d.pages) return;
                var iso = d.pages[key] || d.pages[key.replace(/\/$/, '/index.html')];
                if (!iso) return;
                var label = new Date(iso + 'T12:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
                var span = foot.querySelector('.page-foot__date');
                span.textContent = 'Updated ' + label;
                span.hidden = false;
                var eb = document.querySelector('[data-dated]');
                if (eb && !eb.querySelector('.eyebrow-date')) {
                    var s2 = document.createElement('span');
                    s2.className = 'eyebrow-date';
                    s2.textContent = ' · updated ' + label;
                    eb.appendChild(s2);
                }
            })
            .catch(function () {});
    }

    // ============================================================
    // Long guides (Business Guide): an "In this guide" list at the top, and
    // very long sections folded to their opening with "Show the full
    // section". The fold uses hidden="until-found", so find-in-page and links
    // into the folded part still open it.
    // ============================================================

    function initGuideOverview() {
        if (!/business-guide/.test(location.pathname)) return;
        var main = document.querySelector('.main-content .content-container') || document.querySelector('.main-content');
        if (!main) return;
        var secs = Array.prototype.filter.call(main.querySelectorAll(':scope > section[id]'), function (s) { return s.querySelector('h2'); });
        if (secs.length < 4) return;
        var words = function (el) { return (el.textContent || '').split(/\s+/).length; };
        var total = 0;
        var items = secs.map(function (s) {
            var w = words(s); total += w;
            var h = s.querySelector('h2');
            return '<li><a href="#' + s.id + '">' + h.textContent.trim().replace(/[<>&]/g, '') + '</a><span>' + Math.max(1, Math.round(w / 230)) + ' min</span></li>';
        }).join('');
        var box = document.createElement('nav');
        box.className = 'guide-overview';
        box.setAttribute('aria-label', 'In this guide');
        box.innerHTML = '<p class="guide-overview__title">In this guide <span>' + secs.length + ' sections · about ' + Math.round(total / 230) + ' min</span></p><ol>' + items + '</ol>';
        var first = secs[0];
        var h1 = main.querySelector('h1');
        var anchor = h1 && h1.nextElementSibling && h1.nextElementSibling.tagName === 'P' ? h1.nextElementSibling : null;
        if (anchor) anchor.parentNode.insertBefore(box, anchor.nextSibling); else first.parentNode.insertBefore(box, first);
        // Fold sections longer than ~2.5 screens, keeping the heading and opening.
        secs.forEach(function (s) {
            if (s.getBoundingClientRect().height < 2600) return;
            var kids = Array.prototype.slice.call(s.children);
            var keep = 0, acc = 0;
            for (var i = 0; i < kids.length; i++) { acc += kids[i].getBoundingClientRect().height; keep = i + 1; if (acc > 700 && i >= 2) break; }
            if (keep >= kids.length - 1) return;
            var rest = document.createElement('div');
            rest.className = 'guide-fold';
            rest.setAttribute('hidden', 'until-found');
            kids.slice(keep).forEach(function (k) { rest.appendChild(k); });
            s.appendChild(rest);
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'guide-fold__toggle';
            btn.textContent = 'Show the full section';
            btn.setAttribute('aria-expanded', 'false');
            s.insertBefore(btn, rest);
            var open = function () { rest.removeAttribute('hidden'); btn.remove(); };
            btn.addEventListener('click', open);
            rest.addEventListener('beforematch', open);
        });
        var openTarget = function () {
            var id = decodeURIComponent(location.hash.slice(1));
            var el = id && document.getElementById(id);
            var f = el && el.closest('.guide-fold');
            if (f) { f.removeAttribute('hidden'); var b = f.previousElementSibling; if (b && b.classList.contains('guide-fold__toggle')) b.remove(); el.scrollIntoView(); }
        };
        window.addEventListener('hashchange', openTarget);
        openTarget();
    }

    // ============================================================
    // API Reference, one package at a time. The page holds 14 packages
    // (Overview, the 11 numbered sections, Rendering, Infrastructure) in one
    // file, which several generators and tests read by path, so the file stays
    // whole and the READER sees one package at a time: a switcher under the
    // title, a URL per package (?pkg=...), and "All on one page". Hidden
    // packages use hidden="until-found", so find-in-page still finds text in
    // them and opens the right package; links to any anchor do the same.
    // ============================================================

    var API_PACKAGES = [
        { k: 'overview', n: '', t: 'Overview', start: null },
        { k: 'preprocessing', n: '1', t: 'Data & Preprocessing', start: '#preprocessing-overview' },
        { k: 'training', n: '2', t: 'Training-Time Interventions', start: '#in-processing-overview' },
        { k: 'prediction', n: '3', t: 'Prediction-Time Interventions', start: '#post-processing-overview' },
        { k: 'evaluation', n: '4', t: 'Evaluation & Measurement', start: '#evaluation-overview' },
        { k: 'operations', n: '5', t: 'Operations & Monitoring', start: '#operations-overview' },
        { k: 'reporting', n: '6', t: 'Reporting & Dashboards', start: '#reporting-overview' },
        { k: 'experimentation', n: '7', t: 'Experimentation', start: '#experimentation-overview' },
        { k: 'cicd', n: '8', t: 'CI/CD Integration', start: '#cicd-overview' },
        { k: 'llm', n: '9', t: 'LLM Fairness Testing', start: '#llm-overview' },
        { k: 'agents', n: '10', t: 'Agent Fairness Testing', start: '#agent-overview' },
        { k: 'multi-agent', n: '11', t: 'Multi-Agent Fairness Testing', start: '#multi-agent-overview' },
        { k: 'rendering', n: '', t: 'Rendering Adapters', start: '#rendering-overview' },
        { k: 'infrastructure', n: '', t: 'Infrastructure', start: '#net-overview' }
    ];

    function initApiPackageView() {
        if (!/api-reference/.test(location.pathname)) return;
        var box = document.querySelector('.main-content .content-container');
        var intro = document.getElementById('introduction');
        if (!box || !intro) return;
        // 1. Tag every top-level block with its package.
        var starts = {};
        API_PACKAGES.forEach(function (pk) {
            if (!pk.start) return;
            var el = box.querySelector(pk.start);
            if (!el || el.parentElement !== box) return;
            var prev = el.previousElementSibling;
            if (prev && prev.classList.contains('category-divider')) el = prev;
            starts[pk.k] = el;
        });
        if (Object.keys(starts).length < 10) return;   // structure changed: leave the page whole
        var current = 'overview';
        var blocks = [];
        Array.prototype.forEach.call(box.children, function (el) {
            for (var k in starts) { if (starts[k] === el) current = k; }
            el.setAttribute('data-pkg', current);
            blocks.push(el);
        });
        // The title stays visible in every package: everything else in the
        // introduction belongs to the Overview package.
        var keep = [];
        var h1 = intro.querySelector('h1');
        if (h1) {
            var e = h1.previousElementSibling; if (e) keep.push(e);
            keep.push(h1);
            var lead = h1.nextElementSibling; if (lead && lead.tagName === 'P') keep.push(lead);
        }
        var introKids = Array.prototype.filter.call(intro.children, function (c) { return keep.indexOf(c) < 0; });
        // 2. The switcher, right under the title.
        var nav = document.createElement('nav');
        nav.className = 'pkg-switch';
        nav.setAttribute('aria-label', 'API packages');
        nav.innerHTML = API_PACKAGES.map(function (pk) {
            return '<a href="?pkg=' + pk.k + '" data-k="' + pk.k + '">' + (pk.n ? '<span class="sec-num">' + pk.n + '</span>' : '') + pk.t + '</a>';
        }).join('') + '<a href="?pkg=all" data-k="all" class="pkg-switch__all">All on one page</a>';
        var anchor = keep[keep.length - 1] || h1;
        anchor.parentNode.insertBefore(nav, anchor.nextSibling);
        // 3. Next / previous package at the end of each package.
        var pager = document.createElement('nav');
        pager.className = 'page-pager pkg-pager';
        pager.setAttribute('aria-label', 'Previous and next package');
        var foot = box.querySelector('.page-foot');
        box.insertBefore(pager, foot || null);

        var visible = null;
        function show(k, opts) {
            opts = opts || {};
            if (k !== 'all' && !API_PACKAGES.some(function (p) { return p.k === k; })) k = 'overview';
            visible = k;
            blocks.forEach(function (el) {
                var on = k === 'all' || el.getAttribute('data-pkg') === k || el === intro;
                if (on) el.removeAttribute('hidden'); else el.setAttribute('hidden', 'until-found');
            });
            introKids.forEach(function (el) {
                if (k === 'all' || k === 'overview') el.removeAttribute('hidden'); else el.setAttribute('hidden', 'until-found');
            });
            nav.querySelectorAll('a').forEach(function (a) { a.classList.toggle('is-active', a.getAttribute('data-k') === k); a.setAttribute('aria-current', a.getAttribute('data-k') === k ? 'page' : 'false'); });
            var i = API_PACKAGES.map(function (p) { return p.k; }).indexOf(k);
            var prev = i > 0 ? API_PACKAGES[i - 1] : null, next = i >= 0 && i < API_PACKAGES.length - 1 ? API_PACKAGES[i + 1] : null;
            pager.hidden = k === 'all';
            pager.innerHTML = (prev ? '<a class="page-pager__prev" href="?pkg=' + prev.k + '"><span><i aria-hidden="true">\u2190</i>Previous package</span><strong>' + (prev.n ? prev.n + '. ' : '') + prev.t + '</strong></a>' : '<span></span>') +
                (next ? '<a class="page-pager__next" href="?pkg=' + next.k + '"><span>Next package<i aria-hidden="true">\u2192</i></span><strong>' + (next.n ? next.n + '. ' : '') + next.t + '</strong></a>' : '<span></span>');
            if (typeof mermaid !== 'undefined') {
                var todo = Array.prototype.filter.call(document.querySelectorAll('.mermaid:not([data-processed])'), function (n) { return !n.closest('[hidden]'); });
                if (todo.length) mermaid.run({ nodes: todo }).catch(function () {});
            }
            if (opts.push) {
                var url = location.pathname + (k === 'overview' ? '' : '?pkg=' + k) + (opts.hash || '');
                try { history.pushState({ pkg: k }, '', url); } catch (e) {}
            }
            if (opts.top) { var r = nav.getBoundingClientRect(); window.scrollTo(0, Math.max(0, window.scrollY + r.top - 120)); }
            window.dispatchEvent(new Event('scroll'));
        }
        function pkgOf(el) {
            for (var q = el; q && q !== box; q = q.parentElement) { if (q.parentElement === box) return q.getAttribute('data-pkg'); }
            if (intro.contains(el)) return 'overview';
            return null;
        }
        function revealHash(push) {
            var id = decodeURIComponent(location.hash.slice(1));
            var el = id && document.getElementById(id);
            if (!el) return false;
            var k = pkgOf(el);
            if (k && visible !== 'all' && k !== visible) show(k, { push: push, hash: location.hash });
            el.scrollIntoView();
            return true;
        }
        // Initial state: an anchor wins, then ?pkg=, then Overview.
        var params = new URLSearchParams(location.search);
        show(params.get('pkg') || 'overview');
        if (location.hash) revealHash(false);
        nav.addEventListener('click', function (e) {
            var a = e.target.closest('a[data-k]'); if (!a) return;
            e.preventDefault(); show(a.getAttribute('data-k'), { push: true, top: true });
        });
        pager.addEventListener('click', function (e) {
            var a = e.target.closest('a'); if (!a) return;
            e.preventDefault(); show(new URLSearchParams(a.getAttribute('href').slice(1)).get('pkg'), { push: true, top: true });
        });
        // In-page links (sidebar, overview cards, search results on this page).
        document.addEventListener('click', function (e) {
            var a = e.target.closest('a[href*="#"]'); if (!a || e.defaultPrevented) return;
            var href = a.getAttribute('href');
            if (href.charAt(0) !== '#' && !/api-reference\/(index\.html)?#/.test(href)) return;
            var id = decodeURIComponent(href.split('#')[1] || '');
            var el = id && document.getElementById(id);
            if (!el) return;
            var k = pkgOf(el);
            if (k && visible !== 'all' && k !== visible) {
                e.preventDefault();
                show(k, { push: true, hash: '#' + id });
                el.scrollIntoView();
            }
        }, true);
        window.addEventListener('hashchange', function () { revealHash(false); });
        window.addEventListener('popstate', function () {
            show(new URLSearchParams(location.search).get('pkg') || 'overview');
            if (location.hash) revealHash(false);
        });
        // Find-in-page reached text in a hidden package: open that package.
        blocks.concat(introKids).forEach(function (el) {
            el.addEventListener('beforematch', function () {
                var k = el === intro || introKids.indexOf(el) >= 0 ? 'overview' : el.getAttribute('data-pkg');
                show(k, { push: true });
            });
        });
    }

    // ============================================================
    // Search on phones: the header hides the search box below 768px, so
    // move it into the top of the menu panel instead of losing it.
    // ============================================================

    function initMobileSearch() {
        if (!window.matchMedia || !window.matchMedia('(max-width: 768px)').matches) return;
        var box = document.querySelector('.nav-cta .search-container');
        var panel = document.querySelector('.mobile-nav-panel');
        if (!box || !panel) return;
        var head = panel.querySelector('.mobile-nav-header');
        if (head && head.nextSibling) panel.insertBefore(box, head.nextSibling);
        else panel.insertBefore(box, panel.firstChild);
    }

    // ============================================================
    // Cookie Consent Manager (GDPR-aligned)
    // ============================================================


    // ============================================================
    // Code Copy Functionality
    // ============================================================

    function initCodeCopy() {
        const codeBlocks = document.querySelectorAll('.code-block');

        codeBlocks.forEach(block => {
            const copyBtn = block.querySelector('.code-copy');
            const code = block.querySelector('code');

            if (copyBtn && code) {
                copyBtn.addEventListener('click', async () => {
                    try {
                        await navigator.clipboard.writeText(code.textContent);
                        copyBtn.textContent = 'Copied!';
                        copyBtn.classList.add('copied');

                        setTimeout(() => {
                            copyBtn.textContent = 'Copy';
                            copyBtn.classList.remove('copied');
                        }, 2000);
                    } catch (err) {
                        console.error('Failed to copy:', err);
                        copyBtn.textContent = 'Failed';
                        setTimeout(() => {
                            copyBtn.textContent = 'Copy';
                        }, 2000);
                    }
                });
            }
        });

        // Also handle pre tags without the code-block wrapper
        const preTags = document.querySelectorAll('pre:not(.code-block pre)');
        preTags.forEach(pre => {
            // Create copy button
            const copyBtn = document.createElement('button');
            copyBtn.className = 'code-copy-inline';
            copyBtn.textContent = 'Copy';
            copyBtn.style.cssText = `
                position: absolute;
                top: 8px;
                right: 8px;
                padding: 4px 8px;
                font-size: 12px;
                background: #374151;
                border: none;
                color: #9ca3af;
                border-radius: 4px;
                cursor: pointer;
                opacity: 0;
                transition: opacity 0.2s;
            `;

            // Wrap pre in relative container
            const wrapper = document.createElement('div');
            wrapper.style.position = 'relative';
            pre.parentNode.insertBefore(wrapper, pre);
            wrapper.appendChild(pre);
            wrapper.appendChild(copyBtn);

            // Show on hover
            wrapper.addEventListener('mouseenter', () => {
                copyBtn.style.opacity = '1';
            });
            wrapper.addEventListener('mouseleave', () => {
                copyBtn.style.opacity = '0';
            });

            // Copy functionality
            copyBtn.addEventListener('click', async () => {
                const code = pre.querySelector('code') || pre;
                try {
                    await navigator.clipboard.writeText(code.textContent);
                    copyBtn.textContent = 'Copied!';
                    setTimeout(() => {
                        copyBtn.textContent = 'Copy';
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy:', err);
                }
            });
        });
    }

    // ============================================================
    // Tab Navigation
    // ============================================================

    function initTabs() {
        const tabContainers = document.querySelectorAll('.tabs');

        tabContainers.forEach(container => {
            const tabBtns = container.querySelectorAll('.tab-btn');
            const panels = container.parentElement.querySelectorAll('.tab-panel');

            tabBtns.forEach(btn => {
                btn.addEventListener('click', () => {
                    const targetId = btn.getAttribute('data-tab');

                    // Update buttons
                    tabBtns.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');

                    // Update panels
                    panels.forEach(panel => {
                        if (panel.id === targetId) {
                            panel.classList.add('active');
                        } else {
                            panel.classList.remove('active');
                        }
                    });
                });
            });
        });
    }

    // ============================================================
    // Search Functionality
    // ============================================================

    function initSearch() {
        const searchContainer = document.querySelector('.search-container');
        const searchInput = document.querySelector('.search-input');
        if (!searchInput || !searchContainer) return;

        // Synonym map for common search terms
        const synonyms = {
            'fair': ['fairness', 'equity', 'equitable', 'unbiased'],
            'fairness': ['fair', 'equity', 'equitable', 'unbiased'],
            'bias': ['biased', 'unfair', 'discrimination', 'discriminatory', 'prejudice'],
            'metric': ['metrics', 'measure', 'measurement', 'indicator'],
            'metrics': ['metric', 'measure', 'measurement', 'indicator'],
            'test': ['testing', 'tests', 'check', 'validate', 'validation'],
            'testing': ['test', 'tests', 'check', 'validate', 'validation'],
            'ml': ['machine learning', 'ai', 'model', 'algorithm'],
            'ai': ['artificial intelligence', 'machine learning', 'ml', 'model'],
            'model': ['models', 'ml', 'ai', 'algorithm', 'classifier'],
            'install': ['installation', 'setup', 'pip', 'configure'],
            'setup': ['install', 'installation', 'configure', 'config'],
            'example': ['examples', 'sample', 'demo', 'tutorial', 'code'],
            'examples': ['example', 'sample', 'demo', 'tutorial', 'code'],
            'how': ['tutorial', 'guide', 'howto', 'steps'],
            'why': ['reason', 'purpose', 'importance', 'benefit'],
            'error': ['errors', 'issue', 'problem', 'bug', 'fix'],
            'help': ['support', 'guide', 'documentation', 'docs'],
            'docs': ['documentation', 'help', 'guide', 'reference'],
            'doc': ['documentation', 'docs', 'help', 'guide'],
            'tpr': ['true positive rate', 'recall', 'sensitivity'],
            'fpr': ['false positive rate', 'fall-out'],
            'ppv': ['positive predictive value', 'precision'],
            'group': ['groups', 'demographic', 'subgroup', 'category'],
            'stat': ['statistical', 'statistics', 'stats'],
            'stats': ['statistical', 'statistics', 'stat'],
            'viz': ['visualization', 'visualize', 'plot', 'chart', 'graph'],
            'plot': ['visualization', 'chart', 'graph', 'figure', 'viz'],
            'chart': ['visualization', 'plot', 'graph', 'figure', 'viz'],
            // Additional synonyms for broader search coverage
            'train': ['training', 'learning', 'in-processing', 'optimization'],
            'training': ['train', 'learning', 'in-processing', 'fit'],
            'post': ['post-processing', 'output', 'after', 'adjustment'],
            'pre': ['pre-processing', 'data', 'before', 'input'],
            'constraint': ['constrained', 'limit', 'bound', 'restriction', 'requirement'],
            'causal': ['causation', 'causality', 'cause', 'mechanism', 'counterfactual'],
            'cause': ['causal', 'causation', 'reason', 'why'],
            'monitor': ['monitoring', 'track', 'observe', 'watch', 'dashboard'],
            'monitoring': ['monitor', 'tracking', 'observability', 'production'],
            'deploy': ['deployment', 'production', 'launch', 'release'],
            'production': ['deploy', 'deployment', 'live', 'real-world'],
            'drift': ['shift', 'change', 'degradation', 'evolution'],
            'threshold': ['cutoff', 'boundary', 'limit', 'decision point'],
            'library': ['libraries', 'package', 'tool', 'framework', 'toolkit'],
            'implement': ['implementation', 'deploy', 'build', 'develop', 'code'],
            'governance': ['oversight', 'accountability', 'control', 'management'],
            'compliance': ['regulatory', 'regulation', 'legal', 'law', 'requirement'],
            'regulation': ['regulatory', 'compliance', 'law', 'legal', 'rule'],
            'eu': ['european', 'europe', 'gdpr', 'ai act'],
            'law': ['legal', 'regulation', 'compliance', 'rule', 'legislation'],
            'scrum': ['agile', 'sprint', 'development', 'methodology'],
            'agile': ['scrum', 'sprint', 'iterative', 'development'],
            'pipeline': ['workflow', 'process', 'mlops', 'ci/cd'],
            'feedback': ['loop', 'cycle', 'iteration', 'amplification'],
        };

        // Global search index - covers all documentation pages
        // Each item has keywords and description for better matching
        const searchIndex = [
            // Getting Started
            { title: 'Getting Started', path: 'getting-started/', section: 'Getting Started', type: 'page', icon: 'rocket', keywords: ['start', 'begin', 'tutorial', 'introduction', 'intro', 'guide', 'learn', 'new', 'beginner'], description: 'Learn how to use vfairness for AI fairness analysis' },
            { title: 'Installation', path: 'getting-started/#installation', section: 'Getting Started', type: 'section', icon: 'download', keywords: ['install', 'pip', 'setup', 'requirements', 'dependencies', 'package', 'python', 'conda'], description: 'Install vfairness via pip or from source' },
            { title: 'Quick Start', path: 'getting-started/#quick-start', section: 'Getting Started', type: 'section', icon: 'play', keywords: ['quick', 'fast', 'example', 'demo', 'simple', 'basic', 'first', 'hello world'], description: 'Get started with your first fairness analysis in minutes' },
            { title: 'First Fairness Analysis', path: 'getting-started/#first-analysis', section: 'Getting Started', type: 'section', icon: 'chart', keywords: ['first', 'analyze', 'analysis', 'run', 'execute', 'compute', 'calculate'], description: 'Run your first complete fairness analysis' },
            { title: 'Adding Confidence Intervals', path: 'getting-started/#with-confidence', section: 'Getting Started', type: 'section', icon: 'stats', keywords: ['confidence', 'interval', 'ci', 'bootstrap', 'uncertainty', 'error bars', 'statistical'], description: 'Add statistical confidence to your fairness metrics' },
            { title: 'Bias Detection Tutorial', path: 'getting-started/#bias-detection', section: 'Getting Started', type: 'section', icon: 'search', keywords: ['bias', 'detect', 'detection', 'find', 'identify', 'discover', 'uncover'], description: 'Learn to detect and identify bias in ML models' },
            { title: 'Getting Explanations', path: 'getting-started/#explanations', section: 'Getting Started', type: 'section', icon: 'book', keywords: ['explain', 'explanation', 'interpret', 'understand', 'fairexplainer', 'human readable', 'plain english', 'natural language'], description: 'Generate human-readable explanations of fairness results' },
            { title: 'Visualization Tutorial', path: 'getting-started/#visualization', section: 'Getting Started', type: 'section', icon: 'chart', keywords: ['visual', 'plot', 'chart', 'graph', 'matplotlib', 'figure', 'image', 'diagram', 'display', 'show'], description: 'Create visual representations of fairness metrics' },
            { title: 'MLOps Integration', path: 'getting-started/#mlops', section: 'Getting Started', type: 'section', icon: 'integration', keywords: ['mlops', 'pipeline', 'ci/cd', 'automation', 'mlflow', 'production', 'deploy', 'workflow', 'continuous'], description: 'Integrate fairness checks into ML pipelines' },

            // API Reference
            { title: 'API Reference', path: 'api-reference/', section: 'API Reference', type: 'page', icon: 'code', keywords: ['api', 'reference', 'documentation', 'docs', 'functions', 'methods', 'classes', 'modules', 'library'], description: 'Complete API documentation for vfairness' },
            { title: 'FairnessAnalyzer', path: 'api-reference/#fairness-analyzer', section: 'API Reference', type: 'class', icon: 'box', keywords: ['analyzer', 'fairness', 'class', 'main', 'primary', 'core', 'central'], description: 'Main class for computing fairness metrics' },
            { title: 'BiasDetector', path: 'api-reference/#bias-detector', section: 'API Reference', type: 'class', icon: 'box', keywords: ['bias', 'detector', 'detect', 'class', 'find', 'identify', 'scan', 'check'], description: 'Automatically detect potential biases in your model' },
            { title: 'FairExplAIner', path: 'api-reference/#fair-explainer', section: 'API Reference', type: 'class', icon: 'box', keywords: ['explainer', 'explain', 'interpretation', 'human', 'readable', 'text', 'natural language', 'gpt', 'llm'], description: 'Generate natural language explanations of fairness results' },
            { title: 'Classification Metrics', path: 'api-reference/#classification-metrics', section: 'API Reference', type: 'section', icon: 'list', keywords: ['classification', 'binary', 'multiclass', 'predict', 'label', 'category', 'classifier', 'class'], description: 'Fairness metrics for classification models' },
            { title: 'Regression Metrics', path: 'api-reference/#regression-metrics', section: 'API Reference', type: 'section', icon: 'list', keywords: ['regression', 'continuous', 'numeric', 'predict', 'value', 'number', 'regressor'], description: 'Fairness metrics for regression models' },
            { title: 'Ranking Metrics', path: 'api-reference/#ranking-metrics', section: 'API Reference', type: 'section', icon: 'list', keywords: ['ranking', 'rank', 'order', 'position', 'recommendation', 'recommender', 'top-k', 'list'], description: 'Fairness metrics for ranking and recommendation systems' },
            { title: 'Demographic Parity', path: 'api-reference/#demographic-parity', section: 'API Reference', type: 'metric', icon: 'scale', keywords: ['demographic', 'parity', 'statistical', 'independence', 'selection', 'rate', 'dp', 'equal selection'], description: 'Equal positive prediction rates across groups' },
            { title: 'Equal Opportunity', path: 'api-reference/#equal-opportunity', section: 'API Reference', type: 'metric', icon: 'scale', keywords: ['equal', 'opportunity', 'tpr', 'true positive', 'recall', 'sensitivity', 'eo', 'fnr'], description: 'Equal true positive rates across groups' },
            { title: 'Equalized Odds', path: 'api-reference/#equalized-odds', section: 'API Reference', type: 'metric', icon: 'scale', keywords: ['equalized', 'odds', 'tpr', 'fpr', 'separation', 'eqodd', 'equal odds'], description: 'Equal TPR and FPR across groups' },
            { title: 'Disparate Impact', path: 'api-reference/#disparate-impact', section: 'API Reference', type: 'metric', icon: 'scale', keywords: ['disparate', 'impact', '80%', 'four-fifths', 'ratio', 'adverse', 'di', 'legal', 'eeoc'], description: '80% rule for adverse impact analysis' },
            { title: 'Predictive Parity', path: 'api-reference/#predictive-parity', section: 'API Reference', type: 'metric', icon: 'scale', keywords: ['predictive', 'parity', 'ppv', 'precision', 'sufficiency', 'positive predictive'], description: 'Equal precision across groups' },
            { title: 'Calibration', path: 'api-reference/#calibration', section: 'API Reference', type: 'metric', icon: 'scale', keywords: ['calibration', 'calibrated', 'probability', 'score', 'well-calibrated', 'calibration curve'], description: 'Probability calibration across groups' },
            { title: 'Confidence Intervals', path: 'api-reference/#confidence-intervals', section: 'API Reference', type: 'section', icon: 'stats', keywords: ['confidence', 'interval', 'ci', 'uncertainty', 'statistical', 'error', 'range', 'bound'], description: 'Statistical uncertainty quantification' },
            { title: 'Bootstrap CI', path: 'api-reference/#bootstrap-ci', section: 'API Reference', type: 'function', icon: 'function', keywords: ['bootstrap', 'resampling', 'nonparametric', 'bca', 'percentile', 'sample'], description: 'Bootstrap-based confidence intervals' },
            { title: 'Bayesian CI', path: 'api-reference/#bayesian-ci', section: 'API Reference', type: 'function', icon: 'function', keywords: ['bayesian', 'posterior', 'prior', 'small sample', 'credible', 'beta'], description: 'Bayesian credible intervals for small samples' },
            { title: 'Permutation Testing', path: 'api-reference/#permutation-testing', section: 'API Reference', type: 'function', icon: 'function', keywords: ['permutation', 'test', 'significance', 'p-value', 'hypothesis', 'null', 'shuffle'], description: 'Statistical significance testing' },
            { title: 'Effect Size', path: 'api-reference/#effect-size', section: 'API Reference', type: 'function', icon: 'function', keywords: ['effect', 'size', 'cohen', 'magnitude', 'practical', 'd', 'h', 'glass'], description: 'Practical significance measures' },
            { title: 'MLflow Integration', path: 'api-reference/#mlflow', section: 'API Reference', type: 'section', icon: 'integration', keywords: ['mlflow', 'tracking', 'experiment', 'log', 'artifact', 'run', 'metric logging'], description: 'Track fairness metrics in MLflow' },
            { title: 'pytest Assertions', path: 'api-reference/#pytest', section: 'API Reference', type: 'section', icon: 'check', keywords: ['pytest', 'test', 'assertion', 'testing', 'ci', 'unit', 'assert', 'check', 'fail'], description: 'Fairness assertions for automated testing' },
            { title: 'Visualization', path: 'api-reference/#visualization', section: 'API Reference', type: 'section', icon: 'chart', keywords: ['visualization', 'plot', 'chart', 'graph', 'matplotlib', 'figure', 'bar', 'heatmap', 'radar'], description: 'Built-in visualization functions' },
            { title: 'get_report', path: 'api-reference/#get-report', section: 'API Reference', type: 'function', icon: 'function', keywords: ['report', 'generate', 'summary', 'results', 'output', 'export', 'json', 'dict'], description: 'Generate comprehensive fairness reports' },
            { title: 'Protected Attributes', path: 'api-reference/#protected-attributes', section: 'API Reference', type: 'section', icon: 'shield', keywords: ['protected', 'sensitive', 'attribute', 'gender', 'race', 'age', 'sex', 'religion', 'disability'], description: 'Handling sensitive demographic attributes' },
            { title: 'Subgroup Analysis', path: 'api-reference/#subgroup', section: 'API Reference', type: 'section', icon: 'layers', keywords: ['subgroup', 'intersectional', 'intersection', 'combination', 'cross', 'multiple groups'], description: 'Analyze intersectional fairness' },

            // Business Guide
            { title: 'Business Guide', path: 'business-guide/', section: 'Business Guide', type: 'page', icon: 'briefcase', keywords: ['business', 'guide', 'manager', 'stakeholder', 'executive', 'leadership', 'non-technical', 'overview'], description: 'AI fairness guide for business stakeholders' },
            { title: 'Why Fairness Matters', path: 'business-guide/#why-fairness-matters', section: 'Business Guide', type: 'section', icon: 'alert', keywords: ['why', 'importance', 'matter', 'risk', 'benefit', 'value', 'roi', 'reputation', 'trust'], description: 'Business case for AI fairness' },
            { title: 'Regulatory Landscape', path: 'business-guide/#regulatory-landscape', section: 'Business Guide', type: 'section', icon: 'shield', keywords: ['regulatory', 'regulation', 'law', 'eu', 'gdpr', 'ai act', 'legal', 'compliance', 'government', 'policy'], description: 'AI fairness regulations worldwide' },
            { title: 'Understanding Fairness', path: 'business-guide/#understanding-fairness', section: 'Business Guide', type: 'section', icon: 'book', keywords: ['understand', 'concept', 'definition', 'what is', 'basics', 'fundamentals', 'learn'], description: 'Core fairness concepts explained simply' },
            { title: 'Demographic Parity Explained', path: 'business-guide/#demographic-parity', section: 'Business Guide', type: 'section', icon: 'scale', keywords: ['demographic', 'parity', 'business', 'explain', 'simple', 'plain'], description: 'Demographic parity in plain language' },
            { title: 'Equal Opportunity Explained', path: 'business-guide/#equal-opportunity', section: 'Business Guide', type: 'section', icon: 'scale', keywords: ['equal', 'opportunity', 'business', 'explain', 'simple', 'plain'], description: 'Equal opportunity in plain language' },
            { title: 'Choosing Metrics', path: 'business-guide/#choosing-metrics', section: 'Business Guide', type: 'section', icon: 'scale', keywords: ['choose', 'select', 'metric', 'which', 'decision', 'pick', 'right', 'appropriate', 'best'], description: 'How to select the right fairness metrics' },
            { title: 'Historical Patterns', path: 'business-guide/#historical-patterns', section: 'Business Guide', type: 'section', icon: 'clock', keywords: ['historical', 'history', 'pattern', 'past', 'legacy', 'previous', 'old data'], description: 'Understanding historical bias in data' },
            { title: 'Representation Bias', path: 'business-guide/#representation-bias', section: 'Business Guide', type: 'section', icon: 'pie', keywords: ['representation', 'underrepresented', 'sample', 'data', 'minority', 'imbalance', 'skew'], description: 'When groups are underrepresented' },
            { title: 'Proxy Variables', path: 'business-guide/#proxy-variables', section: 'Business Guide', type: 'section', icon: 'link', keywords: ['proxy', 'variable', 'indirect', 'correlation', 'zip code', 'surrogate', 'redlining'], description: 'Hidden discrimination through proxies' },
            { title: 'Audit Workflow', path: 'business-guide/#audit-workflow', section: 'Business Guide', type: 'section', icon: 'check', keywords: ['audit', 'workflow', 'process', 'step', 'procedure', 'checklist', 'review'], description: 'Step-by-step fairness audit process' },
            { title: 'Interpreting Results', path: 'business-guide/#interpreting-results', section: 'Business Guide', type: 'section', icon: 'chart', keywords: ['interpret', 'result', 'understand', 'read', 'meaning', 'analysis', 'what does'], description: 'How to read fairness analysis results' },
            { title: 'Thresholds & Standards', path: 'business-guide/#thresholds', section: 'Business Guide', type: 'section', icon: 'scale', keywords: ['threshold', 'standard', 'acceptable', 'limit', 'tolerance', 'cutoff', 'pass', 'fail'], description: 'Setting fairness thresholds' },
            { title: 'Remediation Strategies', path: 'business-guide/#remediation', section: 'Business Guide', type: 'section', icon: 'alert', keywords: ['remediation', 'fix', 'improve', 'mitigation', 'strategy', 'solve', 'correct', 'address'], description: 'How to address fairness issues' },
            { title: 'Understanding Fairness Reports', path: 'business-guide/#understanding-reports', section: 'Business Guide', type: 'section', icon: 'file', keywords: ['report', 'understand', 'read', 'interpret', 'document', 'output'], description: 'Reading and using fairness reports' },
            { title: 'Compliance Considerations', path: 'business-guide/#compliance', section: 'Business Guide', type: 'section', icon: 'shield', keywords: ['compliance', 'legal', 'regulation', 'requirement', 'law', 'gdpr', 'eeoc'], description: 'Meeting regulatory requirements' },
            { title: 'Stakeholder Communication', path: 'business-guide/#stakeholder-communication', section: 'Business Guide', type: 'section', icon: 'users', keywords: ['stakeholder', 'communication', 'present', 'explain', 'report', 'meeting', 'executive', 'board'], description: 'Communicating fairness to stakeholders' },

            // Core Concepts
            { title: 'Core Concepts', path: 'concepts/', section: 'Concepts', type: 'page', icon: 'book', keywords: ['concept', 'theory', 'fundamental', 'principle', 'basics', 'foundation', 'learn', 'understand'], description: 'Theoretical foundations of AI fairness' },
            { title: 'Fairness Definitions', path: 'concepts/#fairness-definitions', section: 'Concepts', type: 'section', icon: 'book', keywords: ['definition', 'define', 'what is', 'meaning', 'types', 'kinds'], description: 'Different mathematical definitions of fairness' },
            { title: 'Group Fairness', path: 'concepts/#fairness-definitions', section: 'Concepts', type: 'concept', icon: 'users', keywords: ['group', 'aggregate', 'statistical', 'population', 'demographic', 'collective'], description: 'Fairness across demographic groups' },
            { title: 'Individual Fairness', path: 'concepts/#fairness-definitions', section: 'Concepts', type: 'concept', icon: 'user', keywords: ['individual', 'similar', 'similarity', 'person', 'comparable', 'like'], description: 'Similar individuals treated similarly' },
            { title: 'The Impossibility Theorem', path: 'concepts/#impossibility', section: 'Concepts', type: 'section', icon: 'alert', keywords: ['impossibility', 'theorem', 'tradeoff', 'incompatible', 'cannot', 'choicera', 'tension', 'conflict'], description: 'Why you cannot satisfy all fairness criteria' },
            { title: 'Statistical Validation', path: 'concepts/#statistical-validation', section: 'Concepts', type: 'section', icon: 'stats', keywords: ['statistical', 'validation', 'significance', 'test', 'p-value', 'hypothesis', 'rigorous'], description: 'Statistical rigor in fairness analysis' },
            { title: 'Sample Size Guidelines', path: 'concepts/#sample-size', section: 'Concepts', type: 'section', icon: 'hash', keywords: ['sample', 'size', 'minimum', 'power', 'enough', 'data', 'small', 'large', 'n'], description: 'How much data you need for reliable analysis' },
            { title: 'Intersectionality', path: 'concepts/#intersectionality', section: 'Concepts', type: 'section', icon: 'layers', keywords: ['intersectionality', 'intersection', 'multiple', 'combined', 'subgroup', 'cross', 'overlap'], description: 'Fairness across intersecting identities' },
            { title: 'Fairness Gerrymandering', path: 'concepts/#intersectionality', section: 'Concepts', type: 'concept', icon: 'alert', keywords: ['gerrymandering', 'gaming', 'hide', 'subgroup', 'audit', 'manipulation', 'cheat'], description: 'How fairness can be gamed' },
            { title: 'Sources of Bias', path: 'concepts/#bias-sources', section: 'Concepts', type: 'section', icon: 'alert', keywords: ['source', 'origin', 'cause', 'where', 'how', 'types', 'kinds', 'forms'], description: 'Where bias comes from in ML systems' },
            { title: 'Historical Bias', path: 'concepts/#bias-sources', section: 'Concepts', type: 'concept', icon: 'clock', keywords: ['historical', 'past', 'legacy', 'systemic', 'structural', 'societal'], description: 'Bias from historical discrimination' },
            { title: 'Representation Bias', path: 'concepts/#bias-sources', section: 'Concepts', type: 'concept', icon: 'pie', keywords: ['representation', 'sample', 'data', 'collection', 'underrepresented', 'coverage'], description: 'Bias from unrepresentative data' },
            { title: 'Proxy Discrimination', path: 'concepts/#bias-sources', section: 'Concepts', type: 'concept', icon: 'link', keywords: ['proxy', 'indirect', 'redlining', 'correlation', 'surrogate', 'hidden'], description: 'Indirect discrimination through proxies' },
            { title: 'Measurement Bias', path: 'concepts/#bias-sources', section: 'Concepts', type: 'concept', icon: 'alert', keywords: ['measurement', 'measure', 'label', 'target', 'ground truth', 'outcome'], description: 'Bias in how outcomes are measured' },
            { title: 'Aggregation Bias', path: 'concepts/#bias-sources', section: 'Concepts', type: 'concept', icon: 'layers', keywords: ['aggregation', 'combine', 'single', 'model', 'one-size', 'heterogeneity'], description: 'Bias from one-size-fits-all models' },

            // Training-Time Concepts
            { title: 'Training-Time Fairness', path: 'concepts/#training-concepts', section: 'Concepts', type: 'section', icon: 'cog', keywords: ['training', 'in-processing', 'model', 'learning', 'algorithm', 'optimization', 'during training'], description: 'Fairness interventions during model training' },
            { title: 'Constrained Optimization', path: 'concepts/#constraint-optimization', section: 'Concepts', type: 'section', icon: 'scale', keywords: ['constraint', 'optimization', 'lagrangian', 'objective', 'minimize', 'loss'], description: 'Fairness through constrained optimization' },
            { title: 'Exponentiated Gradient', path: 'concepts/#constraint-optimization', section: 'Concepts', type: 'concept', icon: 'chart', keywords: ['exponentiated', 'gradient', 'reductions', 'agarwal', 'iterative', 'weight'], description: 'Reductions approach to fair classification' },
            { title: 'Lagrangian Relaxation', path: 'concepts/#constraint-optimization', section: 'Concepts', type: 'concept', icon: 'function', keywords: ['lagrangian', 'relaxation', 'multiplier', 'penalty', 'lambda', 'dual'], description: 'Lagrangian method for fairness constraints' },
            { title: 'Pareto Frontier', path: 'concepts/#constraint-optimization', section: 'Concepts', type: 'concept', icon: 'chart', keywords: ['pareto', 'frontier', 'tradeoff', 'optimal', 'boundary', 'curve'], description: 'Optimal fairness-accuracy tradeoffs' },

            // Post-Processing Methods
            { title: 'Post-Processing Methods', path: 'concepts/#post-processing', section: 'Concepts', type: 'section', icon: 'function', keywords: ['post-processing', 'output', 'adjustment', 'calibration', 'after training', 'black-box'], description: 'Adjusting model outputs for fairness' },
            { title: 'Threshold Optimization', path: 'concepts/#threshold-optimization', section: 'Concepts', type: 'section', icon: 'scale', keywords: ['threshold', 'cutoff', 'decision', 'boundary', 'group-specific', 'optimize'], description: 'Group-specific decision thresholds' },
            { title: 'Calibrated Equalized Odds', path: 'concepts/#post-processing', section: 'Concepts', type: 'concept', icon: 'scale', keywords: ['calibrated', 'equalized', 'odds', 'calibration', 'hardt'], description: 'Post-hoc equalized odds adjustment' },
            { title: 'Reject Option Classification', path: 'concepts/#post-processing', section: 'Concepts', type: 'concept', icon: 'function', keywords: ['reject', 'option', 'critical', 'region', 'uncertainty', 'flip'], description: 'Modifying uncertain predictions for fairness' },

            // Evaluation & Measurement
            { title: 'Fairness Libraries', path: 'concepts/#fairness-libraries', section: 'Concepts', type: 'section', icon: 'box', keywords: ['library', 'aif360', 'fairlearn', 'aequitas', 'tool', 'package', 'framework'], description: 'Open-source fairness libraries comparison' },
            { title: 'IBM AIF360', path: 'concepts/#fairness-libraries', section: 'Concepts', type: 'concept', icon: 'box', keywords: ['aif360', 'ibm', 'comprehensive', 'toolkit', 'metrics', 'mitigation'], description: 'IBM AI Fairness 360 toolkit' },
            { title: 'Microsoft Fairlearn', path: 'concepts/#fairness-libraries', section: 'Concepts', type: 'concept', icon: 'box', keywords: ['fairlearn', 'microsoft', 'sklearn', 'scikit-learn', 'constraint'], description: 'Microsoft Fairlearn library' },
            { title: 'Aequitas', path: 'concepts/#fairness-libraries', section: 'Concepts', type: 'concept', icon: 'box', keywords: ['aequitas', 'audit', 'chicago', 'report', 'bias audit'], description: 'Aequitas bias audit toolkit' },
            { title: 'Causal Fairness', path: 'concepts/#causal-fairness', section: 'Concepts', type: 'section', icon: 'link', keywords: ['causal', 'causation', 'counterfactual', 'dag', 'mechanism', 'pearl'], description: 'Causal approaches to fairness' },
            { title: 'Counterfactual Fairness', path: 'concepts/#causal-fairness', section: 'Concepts', type: 'concept', icon: 'link', keywords: ['counterfactual', 'what if', 'alternative', 'kusner', 'individual'], description: 'Fairness through counterfactual reasoning' },
            { title: 'Path-Specific Fairness', path: 'concepts/#causal-fairness', section: 'Concepts', type: 'concept', icon: 'link', keywords: ['path', 'specific', 'causal path', 'direct', 'indirect', 'mediation'], description: 'Blocking discriminatory causal paths' },

            // Operations & Monitoring
            { title: 'Fairness Monitoring', path: 'concepts/#fairness-monitoring', section: 'Concepts', type: 'section', icon: 'eye', keywords: ['monitoring', 'production', 'drift', 'real-time', 'alert', 'dashboard'], description: 'Monitoring fairness in production systems' },
            { title: 'Drift Detection', path: 'concepts/#fairness-monitoring', section: 'Concepts', type: 'concept', icon: 'alert', keywords: ['drift', 'shift', 'distribution', 'change', 'degradation', 'evolve'], description: 'Detecting fairness drift over time' },
            { title: 'Feedback Loops', path: 'concepts/#fairness-monitoring', section: 'Concepts', type: 'concept', icon: 'refresh', keywords: ['feedback', 'loop', 'cycle', 'amplification', 'self-reinforcing', 'runaway'], description: 'How models can amplify their own bias' },
            { title: 'Implementation Strategies', path: 'concepts/#implementation', section: 'Concepts', type: 'section', icon: 'cog', keywords: ['implementation', 'deploy', 'organizational', 'governance', 'scrum', 'agile'], description: 'Implementing fairness in organizations' },
            { title: 'Fair AI Scrum', path: 'concepts/#implementation', section: 'Concepts', type: 'concept', icon: 'check', keywords: ['scrum', 'agile', 'sprint', 'user story', 'definition of done', 'retrospective'], description: 'Integrating fairness into agile development' },
            { title: 'Fairness Governance', path: 'concepts/#implementation', section: 'Concepts', type: 'concept', icon: 'shield', keywords: ['governance', 'accountability', 'role', 'responsibility', 'committee', 'oversight'], description: 'Organizational fairness governance structures' },
            { title: 'Regulatory Compliance', path: 'concepts/#implementation', section: 'Concepts', type: 'concept', icon: 'shield', keywords: ['regulatory', 'compliance', 'eu ai act', 'gdpr', 'eeoc', 'law', 'legal', 'requirement'], description: 'Meeting AI fairness regulations' },

            // References
            { title: 'Academic References', path: 'concepts/references.html', section: 'Resources', type: 'page', icon: 'book', keywords: ['reference', 'academic', 'paper', 'research', 'citation', 'publication', 'study', 'literature'], description: 'Academic papers and research on AI fairness' },
            { title: 'Foundational Works', path: 'concepts/references.html#foundational', section: 'Resources', type: 'section', icon: 'book', keywords: ['foundational', 'seminal', 'classic', 'original', 'key', 'important'], description: 'Seminal papers in AI fairness' },
            { title: 'Fairness Metrics Research', path: 'concepts/references.html#metrics', section: 'Resources', type: 'section', icon: 'scale', keywords: ['research', 'paper', 'metric', 'study', 'academic'], description: 'Research on fairness metrics' },
            { title: 'Bias Detection Research', path: 'concepts/references.html#bias', section: 'Resources', type: 'section', icon: 'search', keywords: ['research', 'paper', 'bias', 'detection', 'academic'], description: 'Research on bias detection methods' },
            { title: 'Ethical Frameworks', path: 'concepts/references.html#ethical', section: 'Resources', type: 'section', icon: 'compass', keywords: ['ethical', 'ethics', 'framework', 'principle', 'moral', 'philosophy'], description: 'Ethical frameworks for AI' },

            // Changelog
            { title: 'Changelog', path: 'concepts/changelog.html', section: 'Resources', type: 'page', icon: 'history', keywords: ['changelog', 'version', 'history', 'update', 'release', 'changes', 'new', 'whats new'], description: 'Version history and release notes' },
            { title: 'Version 0.1.0', path: 'concepts/changelog.html#v0.1.0', section: 'Changelog', type: 'version', icon: 'tag', keywords: ['0.1.0', 'beta', 'latest', 'new', 'current', 'recent'], description: 'Current beta release features' },
            { title: 'Version 0.0.9', path: 'concepts/changelog.html#v0.0.9', section: 'Changelog', type: 'version', icon: 'tag', keywords: ['0.0.9', 'previous', 'older'], description: 'Previous release features' },

            // Legal
            { title: 'Legal & License', path: 'legal/', section: 'Legal', type: 'page', icon: 'file', keywords: ['legal', 'license', 'apache-2.0', 'apache', 'terms', 'copyright', 'open source', 'oss'], description: 'License and legal information' },

            // Capabilities
            { title: 'Explainability & XAI', path: 'explainability-and-xai/', section: 'Capabilities', type: 'page', icon: 'book', keywords: ['explainability', 'xai', 'shap', 'lundberg', 'decomposition', 'counterfactual', 'dice', 'attribution', 'interpretability', 'feature importance', 'proxy', 'explain'], description: 'SHAP attributions, the Lundberg fairness decomposition, DiCE counterfactuals, and explanation-quality diagnostics' },
            { title: 'Training-Time Interventions', path: 'training-time-interventions/', section: 'Capabilities', type: 'page', icon: 'integration', keywords: ['in-processing', 'in_processing', 'training', 'fairness-aware', 'loss', 'constraint', 'regularizer', 'calibration', 'adversarial', 'reduction'], description: 'Fairness-aware model training: constrained losses, regularizers, and calibrators that intervene during learning' },
        ];

        // The generated index (scripts/build_search_index.py): every page, every
        // anchored heading and every documented API object. The curated list
        // above keeps its synonyms and ranks first; an entry whose anchor no
        // longer exists on its page is dropped rather than shown as a dead link.
        let generatedLoaded = false;
        function loadGeneratedIndex() {
            if (generatedLoaded) return Promise.resolve();
            generatedLoaded = true;
            return fetch(getBasePath() + 'data/search-index.json', { cache: 'no-cache' })
                .then(r => r.ok ? r.json() : null)
                .then(data => {
                    if (!data || !data.items) return;
                    const known = new Set(data.items.map(i => i.u));
                    for (let i = searchIndex.length - 1; i >= 0; i--) {
                        const p = searchIndex[i].path;
                        if (p.includes('#') && !known.has(p)) searchIndex.splice(i, 1);
                    }
                    const have = new Set(searchIndex.map(e => e.title.toLowerCase() + '|' + e.path));
                    const icon = { page: 'file', section: 'hash', api: 'function' };
                    const label = { page: 'page', section: 'section', api: 'API' };
                    data.items.forEach(i => {
                        const key = i.t.toLowerCase() + '|' + i.u;
                        if (have.has(key)) return;
                        have.add(key);
                        searchIndex.push({ title: i.t, path: i.u, section: i.s, type: label[i.k] || i.k,
                            icon: icon[i.k] || 'file', keywords: [], description: i.d, generated: true });
                    });
                    if (searchInput.value.trim().length >= 2) performSearch(searchInput.value);
                })
                .catch(() => {});
        }

        // Fuzzy matching function - calculates similarity between two strings
        function fuzzyMatch(str, query) {
            str = str.toLowerCase();
            query = query.toLowerCase();

            // Exact match
            if (str.includes(query)) return 1.0;

            // Check for single character typos (Levenshtein distance 1)
            if (query.length >= 3) {
                // Simple edit distance check for short queries
                let matches = 0;
                let j = 0;
                for (let i = 0; i < str.length && j < query.length; i++) {
                    if (str[i] === query[j]) {
                        matches++;
                        j++;
                    }
                }
                const similarity = matches / query.length;
                if (similarity >= 0.75) return similarity * 0.6; // Penalize fuzzy matches
            }

            return 0;
        }

        // Expand query with synonyms
        function expandQuery(query) {
            const words = query.toLowerCase().split(/\s+/);
            const expanded = new Set(words);

            words.forEach(word => {
                if (synonyms[word]) {
                    synonyms[word].forEach(syn => expanded.add(syn));
                }
            });

            return Array.from(expanded);
        }

        // Create results container
        let resultsContainer = document.createElement('div');
        resultsContainer.className = 'search-results';
        searchContainer.appendChild(resultsContainer);

        // Create overlay
        let overlay = document.createElement('div');
        overlay.className = 'search-overlay';
        document.body.appendChild(overlay);

        // Add keyboard shortcut indicator (detect Mac vs Windows)
        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        let shortcutIndicator = document.createElement('div');
        shortcutIndicator.className = 'search-shortcut';
        shortcutIndicator.innerHTML = isMac ? '<kbd>⌘</kbd><kbd>K</kbd>' : '<kbd>Ctrl</kbd><kbd>K</kbd>';
        searchContainer.appendChild(shortcutIndicator);

        // Update placeholder
        searchInput.placeholder = 'Search docs';

        let selectedIndex = -1;

        // Manage shortcut visibility
        function updateShortcutVisibility() {
            if (document.activeElement === searchInput || searchInput.value.length > 0) {
                shortcutIndicator.style.display = 'none';
            } else {
                shortcutIndicator.style.display = 'flex';
            }
        }

        // Get icon SVG based on type
        function getIconSVG(icon) {
            const icons = {
                'rocket': '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>',
                'download': '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
                'play': '<polygon points="5 3 19 12 5 21 5 3"/>',
                'chart': '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
                'code': '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
                'box': '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>',
                'list': '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
                'scale': '<path d="M16 16l3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M2 16l3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
                'stats': '<path d="M3 3v18h18"/><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/>',
                'function': '<path d="M9 17H5a2 2 0 0 0 0 4h4a2 2 0 0 0 0-4Z"/><path d="M11 3H7a2 2 0 0 0 0 4h4a2 2 0 0 0 0-4Z"/><path d="M19 10h4a2 2 0 0 0 0-4h-4a2 2 0 0 0 0 4Z"/><path d="M17 17h4a2 2 0 0 0 0-4h-4a2 2 0 0 0 0 4Z"/><path d="M9 6v12"/><path d="M17 10v4"/>',
                'integration': '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
                'check': '<polyline points="20 6 9 17 4 12"/>',
                'briefcase': '<rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
                'file': '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>',
                'shield': '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
                'users': '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
                'user': '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
                'book': '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
                'alert': '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
                'layers': '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
                'clock': '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
                'pie': '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
                'link': '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
                'search': '<circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>',
                'compass': '<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>',
                'history': '<path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/>',
                'tag': '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/>',
                'hash': '<line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/>'
            };
            return icons[icon] || icons['file'];
        }

        // Determine base path
        // The site root relative to this page, read from the page's own
        // stylesheet link. A hard-coded list of five folders used to decide
        // this, so every result clicked on any other page linked to a path
        // that did not exist.
        function getBasePath() {
            const css = document.querySelector('link[href*="css/styles.css"]');
            const href = css ? css.getAttribute('href') : 'css/styles.css';
            if (href.startsWith('/')) return '/';
            return href.replace(/css\/styles\.css.*$/, '');
        }

        // Highlight matching text
        function highlightMatch(text, query) {
            if (!query) return text;
            const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
            return text.replace(regex, '<mark>$1</mark>');
        }

        // Render search results
        function renderResults(results, query) {
            if (results.length === 0) {
                resultsContainer.innerHTML = `
                    <div class="search-no-results">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <circle cx="11" cy="11" r="8"/>
                            <path d="M21 21l-4.35-4.35"/>
                        </svg>
                        <p>No results found for "<strong>${query}</strong>"</p>
                        <p style="font-size: 12px; margin-top: 4px;">Try different keywords or check the spelling</p>
                    </div>
                `;
                return;
            }

            const basePath = getBasePath();
            let html = '';
            // Grouped by where the result lives (Start, Reference, Learn ...),
            // groups in the order of their best result.
            const order = [];
            const groups = {};
            results.forEach((item) => {
                const g = groupFor(item.path);
                if (!groups[g]) { groups[g] = []; order.push(g); }
                groups[g].push(item);
            });
            const grouped = [];
            order.forEach((g) => groups[g].forEach((item) => grouped.push(item)));
            results = grouped;
            let lastGroup = null;

            results.forEach((item, index) => {
                const g = groupFor(item.path);
                if (g !== lastGroup) {
                    html += `<div class="search-results-header">${g === 'Reference' ? 'API Reference' : g}</div>`;
                    lastGroup = g;
                }
                const description = item.description ? `<div class="search-result-description">${highlightMatch(item.description, query)}</div>` : '';
                html += `
                    <a href="${basePath}${item.path}" class="search-result-item${index === selectedIndex ? ' selected' : ''}" data-index="${index}">
                        <div class="search-result-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                ${getIconSVG(item.icon)}
                            </svg>
                        </div>
                        <div class="search-result-content">
                            <div class="search-result-title">${highlightMatch(item.title, query)}</div>
                            ${description}
                            <div class="search-result-path">${item.section} · ${item.type}</div>
                        </div>
                    </a>
                `;
            });

            resultsContainer.innerHTML = html;
        }

        // Search function
        function performSearch(query) {
            if (query.length < 2) {
                hideResults();
                return;
            }

            const normalizedQuery = query.toLowerCase().trim();
            const queryWords = normalizedQuery.split(/\s+/);
            const expandedQueryWords = expandQuery(normalizedQuery);

            // Score and filter results
            const scoredResults = searchIndex.map(item => {
                let score = 0;
                const titleLower = item.title.toLowerCase();
                const sectionLower = item.section.toLowerCase();
                const typeLower = item.type.toLowerCase();
                const descriptionLower = (item.description || '').toLowerCase();

                // Exact title match (highest priority)
                if (titleLower === normalizedQuery) {
                    score += 200;
                }
                // Title contains query
                else if (titleLower.includes(normalizedQuery)) {
                    score += 100;
                }
                // Title starts with query
                if (titleLower.startsWith(normalizedQuery)) {
                    score += 50;
                }

                // Description match
                if (descriptionLower.includes(normalizedQuery)) {
                    score += 40;
                }

                // Section match
                if (sectionLower.includes(normalizedQuery)) {
                    score += 20;
                }

                // Type match
                if (typeLower.includes(normalizedQuery)) {
                    score += 10;
                }

                // Keywords match (check each keyword)
                if (item.keywords) {
                    for (const keyword of item.keywords) {
                        // Exact keyword match
                        if (keyword === normalizedQuery) {
                            score += 60;
                        }
                        // Keyword contains query or vice versa
                        else if (keyword.includes(normalizedQuery) || normalizedQuery.includes(keyword)) {
                            score += 35;
                        }

                        // Check expanded query (synonyms)
                        for (const expandedWord of expandedQueryWords) {
                            if (keyword === expandedWord) {
                                score += 25; // Synonym exact match
                            } else if (keyword.includes(expandedWord)) {
                                score += 15; // Synonym partial match
                            }
                        }

                        // Partial word matching with original query words
                        for (const word of queryWords) {
                            if (word.length >= 2) {
                                if (keyword.includes(word)) score += 12;
                                if (word.includes(keyword) && keyword.length >= 3) score += 8;
                            }
                        }
                    }
                }

                // Word-by-word matching for multi-word queries
                for (const word of queryWords) {
                    if (word.length >= 2) {
                        // Title word matching
                        if (titleLower.includes(word)) score += 25;
                        // Word boundary bonus (whole word match)
                        const wordBoundaryRegex = new RegExp(`\\b${word}\\b`);
                        if (wordBoundaryRegex.test(titleLower)) score += 15;

                        // Description word matching
                        if (descriptionLower.includes(word)) score += 10;
                        if (wordBoundaryRegex.test(descriptionLower)) score += 5;

                        // Section word matching
                        if (sectionLower.includes(word)) score += 5;
                    }
                }

                // Fuzzy matching for typo tolerance (only if no exact matches found)
                if (score === 0 && normalizedQuery.length >= 3) {
                    const titleFuzzy = fuzzyMatch(titleLower, normalizedQuery);
                    if (titleFuzzy > 0) score += titleFuzzy * 50;

                    for (const keyword of (item.keywords || [])) {
                        const keywordFuzzy = fuzzyMatch(keyword, normalizedQuery);
                        if (keywordFuzzy > 0) score += keywordFuzzy * 20;
                    }
                }

                if (item.generated && score > 0) score -= 5;
                return { ...item, score };
            })
            .filter(item => item.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, 15); // Limit to 15 results

            selectedIndex = scoredResults.length > 0 ? 0 : -1;
            renderResults(scoredResults, query);
            showResults();
        }

        function showResults() {
            resultsContainer.classList.add('active');
            overlay.classList.add('active');
        }

        function hideResults() {
            resultsContainer.classList.remove('active');
            overlay.classList.remove('active');
            selectedIndex = -1;
        }

        // Event listeners
        searchInput.addEventListener('input', debounce((e) => {
            loadGeneratedIndex();
            performSearch(e.target.value);
            updateShortcutVisibility();
        }, 150));

        searchInput.addEventListener('focus', () => {
            loadGeneratedIndex();
            updateShortcutVisibility();
            if (searchInput.value.length >= 2) {
                performSearch(searchInput.value);
            }
        });

        searchInput.addEventListener('blur', () => {
            updateShortcutVisibility();
        });

        // Keyboard navigation
        searchInput.addEventListener('keydown', (e) => {
            const items = resultsContainer.querySelectorAll('.search-result-item');

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
                updateSelection(items);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                selectedIndex = Math.max(selectedIndex - 1, 0);
                updateSelection(items);
            } else if (e.key === 'Enter' && selectedIndex >= 0) {
                e.preventDefault();
                const selectedItem = items[selectedIndex];
                if (selectedItem) {
                    window.location.href = selectedItem.getAttribute('href');
                }
            } else if (e.key === 'Escape') {
                hideResults();
                searchInput.blur();
            }
        });

        function updateSelection(items) {
            items.forEach((item, index) => {
                item.classList.toggle('selected', index === selectedIndex);
            });
            // Scroll selected item into view
            if (items[selectedIndex]) {
                items[selectedIndex].scrollIntoView({ block: 'nearest' });
            }
        }

        // Close on overlay click
        overlay.addEventListener('click', () => {
            hideResults();
        });

        // Close on click outside
        document.addEventListener('click', (e) => {
            if (!searchContainer.contains(e.target)) {
                hideResults();
            }
        });

        // Keyboard shortcuts: Ctrl/Cmd + K, or "/" anywhere outside a field
        document.addEventListener('keydown', (e) => {
            const t = e.target;
            const typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
            if (e.key === '/' && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
                e.preventDefault();
                searchInput.focus();
                searchInput.select();
                return;
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                searchInput.focus();
                searchInput.select();
            }
        });
    }

    // ============================================================
    // Table of Contents
    // ============================================================

    function initTableOfContents() {
        const toc = document.querySelector('.toc-list');
        if (!toc) return;

        const headings = document.querySelectorAll('.main-content h2, .main-content h3');
        const tocLinks = toc.querySelectorAll('a');

        // Highlight current section on scroll
        function highlightCurrentSection() {
            let currentSection = '';

            headings.forEach(heading => {
                const rect = heading.getBoundingClientRect();
                if (rect.top <= 100) {
                    currentSection = heading.id;
                }
            });

            tocLinks.forEach(link => {
                if (link.getAttribute('href') === '#' + currentSection) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        }

        window.addEventListener('scroll', throttle(highlightCurrentSection, 100));
        highlightCurrentSection();
    }

    // ============================================================
    // Mermaid Diagrams
    // ============================================================

    function initMermaid() {
        if (typeof mermaid === 'undefined') return;

        // Mermaid v11: startOnLoad only fires if the lib loads before
        // DOMContentLoaded. With defer, DOMContentLoaded may already have
        // fired, so we disable startOnLoad and call run() explicitly.
        mermaid.initialize({
            startOnLoad: false,
            theme: 'neutral',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            },
            // Brand ink on paper (validant.ai), so diagrams read as part of
            // the page rather than as a third visual system.
            themeVariables: {
                fontFamily: "'Plus Jakarta Sans', -apple-system, 'Segoe UI', sans-serif",
                primaryColor: '#f4f3f1',
                primaryTextColor: '#1b2735',
                primaryBorderColor: '#293b51',
                lineColor: '#66737e',
                secondaryColor: '#e6f1f3',
                tertiaryColor: '#fcfbfa'
            }
        });

        // Render all .mermaid elements that haven't been processed yet
        // Diagrams inside a hidden API package are rendered when the package is
        // opened (initApiPackageView): a diagram laid out while hidden has no
        // size, so Mermaid would draw it at zero width.
        var nodes = Array.prototype.filter.call(document.querySelectorAll('.mermaid:not([data-processed])'), function (n) { return !n.closest('[hidden]'); });
        if (nodes.length) {
            mermaid.run({ nodes: nodes }).catch(function(err) {
                console.warn('Mermaid render error:', err);
            });
        }
    }

    // ============================================================
    // Anchor Scrolling — instant jump + subtle highlight flash
    // ============================================================

    function initSmoothScroll() {
        // CSS scroll-behavior: auto + scroll-padding-top handles the position.
        // We add a brief highlight flash so the user sees where they landed.

        function flashTarget(hash) {
            if (!hash || hash === '#') return;
            var id = hash.slice(1);
            try { id = decodeURIComponent(id); } catch (e) { /* keep raw */ }
            var el = document.getElementById(id);
            if (!el) return;
            el.classList.remove('anchor-highlight');
            void el.offsetWidth; // force reflow to restart animation
            el.classList.add('anchor-highlight');
            el.addEventListener('animationend', function handler() {
                el.classList.remove('anchor-highlight');
                el.removeEventListener('animationend', handler);
            });
        }

        // Flash on sidebar / in-page anchor clicks (browser handles the scroll natively).
        document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
            anchor.addEventListener('click', function() {
                var hash = this.getAttribute('href');
                // Small delay so the browser finishes scrolling first.
                setTimeout(function() { flashTarget(hash); }, 60);
            });
        });

        // Flash on page load with hash.
        if (window.location.hash) {
            window.addEventListener('load', function() {
                setTimeout(function() { flashTarget(window.location.hash); }, 100);
            });
        }

        // Flash on back/forward navigation.
        window.addEventListener('hashchange', function() {
            setTimeout(function() { flashTarget(window.location.hash); }, 60);
        });
    }

    // ============================================================
    // Mobile Menu
    // ============================================================

    function initMobileMenu() {
        var toggle = document.querySelector('.mobile-nav-toggle');
        var panel = document.querySelector('.mobile-nav-panel');
        var overlay = document.querySelector('.mobile-nav-overlay');
        var closeBtn = document.querySelector('.mobile-nav-close');

        if (!toggle || !panel) return;

        // Skip toggle/close binding if the inline bootstrap already wired them
        if (!toggle.hasAttribute('data-nav-init')) {
            toggle.setAttribute('data-nav-init', '1');

            function openMenu() {
                toggle.setAttribute('aria-expanded', 'true');
                panel.classList.add('is-open');
                panel.setAttribute('aria-hidden', 'false');
                if (overlay) overlay.classList.add('is-open');
                document.body.style.overflow = 'hidden';
                var first = panel.querySelector('button, a, input');
                if (first) setTimeout(function() { first.focus(); }, 100);
            }

            function closeMenu() {
                toggle.setAttribute('aria-expanded', 'false');
                panel.classList.remove('is-open');
                panel.setAttribute('aria-hidden', 'true');
                if (overlay) overlay.classList.remove('is-open');
                document.body.style.overflow = '';
                toggle.focus();
            }

            toggle.addEventListener('click', function() {
                toggle.getAttribute('aria-expanded') === 'true' ? closeMenu() : openMenu();
            });

            if (closeBtn) closeBtn.addEventListener('click', closeMenu);
            if (overlay) overlay.addEventListener('click', closeMenu);

            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape' && panel.classList.contains('is-open')) closeMenu();
            });
        }

        // Accordion groups
        panel.querySelectorAll('.mobile-nav-group__trigger').forEach(function(trigger) {
            trigger.addEventListener('click', function() {
                var expanded = trigger.getAttribute('aria-expanded') === 'true';
                trigger.setAttribute('aria-expanded', expanded ? 'false' : 'true');
            });
        });

        // Active links in mobile nav
        var currentPath = window.location.pathname;
        panel.querySelectorAll('a').forEach(function(link) {
            var href = link.getAttribute('href');
            if (href && (currentPath.endsWith(href) || currentPath.includes(href.replace('index.html', '').replace(/\.\.\//g, '')))) {
                link.classList.add('active');
                var group = link.closest('.mobile-nav-group');
                if (group) {
                    var gt = group.querySelector('.mobile-nav-group__trigger');
                    if (gt) gt.setAttribute('aria-expanded', 'true');
                }
            }
        });
    }

    // ============================================================
    // Utility Functions
    // ============================================================

    function throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // ============================================================
    // Syntax Highlighting (Simple)
    // ============================================================

    // Keywords recognised by the Python highlighter.
    const PY_KEYWORDS = new Set([
        'import', 'from', 'def', 'class', 'return', 'if', 'else', 'elif', 'for',
        'while', 'try', 'except', 'with', 'as', 'in', 'not', 'and', 'or', 'True',
        'False', 'None', 'async', 'await', 'yield', 'raise', 'pass', 'break',
        'continue', 'lambda'
    ]);

    function escapeHtml(s) {
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // Single-pass Python highlighter. The previous version chained several
    // String.replace() calls, so a later pass (strings) re-scanned the markup an
    // earlier pass (keywords) had already inserted: the string regex matched the
    // double quotes in class="hljs-keyword" and mangled it into visible text like
    // hljs-keyword. This tokenizer scans the raw source ONCE and only ever wraps
    // real tokens, so inserted <span> markup is never re-matched. Gap text and
    // token text are HTML-escaped, so code that contains <, > or & is safe too.
    function highlightPython(code) {
        // One alternation, priority order: comment | string | number | word.
        const token = /(#[^\n]*)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\b\d+\.?\d*\b)|([A-Za-z_]\w*)/g;
        let out = '';
        let last = 0;
        let prevWord = null;  // the previous identifier/keyword, for def/class names
        let m;
        while ((m = token.exec(code)) !== null) {
            out += escapeHtml(code.slice(last, m.index));
            const [full, comment, str, num, word] = m;
            if (comment !== undefined) {
                out += '<span class="hljs-comment">' + escapeHtml(comment) + '</span>';
                prevWord = null;
            } else if (str !== undefined) {
                out += '<span class="hljs-string">' + escapeHtml(str) + '</span>';
                prevWord = null;
            } else if (num !== undefined) {
                out += '<span class="hljs-number">' + num + '</span>';
                prevWord = null;
            } else {
                if (PY_KEYWORDS.has(word)) {
                    out += '<span class="hljs-keyword">' + word + '</span>';
                } else if (prevWord === 'def' || prevWord === 'class') {
                    // The name in a `def name` / `class name` declaration.
                    out += '<span class="hljs-function">' + word + '</span>';
                } else {
                    out += word;
                }
                prevWord = word;
            }
            last = token.lastIndex;
        }
        out += escapeHtml(code.slice(last));
        return out;
    }

    function highlightCode() {
        const codeBlocks = document.querySelectorAll('pre code');
        codeBlocks.forEach(block => {
            if (block.classList.contains('language-python') ||
                block.parentElement.classList.contains('language-python')) {
                block.innerHTML = highlightPython(block.textContent);
            }
        });
    }

    // Run syntax highlighting
    document.addEventListener('DOMContentLoaded', highlightCode);

    // ============================================================
    // Collapsible Sections
    // ============================================================

    function initCollapsibles() {
        const collapsibles = document.querySelectorAll('.collapsible');

        collapsibles.forEach(item => {
            const header = item.querySelector('.collapsible-header');
            const content = item.querySelector('.collapsible-content');

            if (header && content) {
                header.addEventListener('click', () => {
                    item.classList.toggle('open');
                    content.style.maxHeight = item.classList.contains('open')
                        ? content.scrollHeight + 'px'
                        : '0';
                });
            }
        });
    }

    document.addEventListener('DOMContentLoaded', initCollapsibles);

    // ============================================================
    // Changelog Collapse (details/summary toggle)
    // ============================================================

    function initChangelogCollapse() {
        // Every version is collapsible (2026-10-01). Only the four oldest were
        // written as <details>; the newest ones were plain sections, so
        // "Expand all" / "Collapse all" visibly did nothing at the top of the
        // page. Wrap each plain version in the same <details> structure: the
        // newest stays open, older ones start closed.
        var plain = document.querySelectorAll('.version-section');
        Array.prototype.forEach.call(plain, function (sec, i) {
            if (sec.querySelector(':scope > .version-details')) return;
            var header = sec.querySelector(':scope > .version-header');
            if (!header) return;
            var det = document.createElement('details');
            det.className = 'version-details';
            if (i === 0) det.open = true;
            var sum = document.createElement('summary');
            var toggle = document.createElement('span');
            toggle.className = 'version-toggle';
            toggle.innerHTML = '<span>' + (det.open ? 'Hide details' : 'Show details') + '</span><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>';
            header.appendChild(toggle);
            var body = document.createElement('div');
            body.className = 'version-body';
            var n = header.nextSibling;
            while (n) { var next = n.nextSibling; body.appendChild(n); n = next; }
            sec.insertBefore(det, header);
            sum.appendChild(header);
            det.appendChild(sum);
            det.appendChild(body);
            // A link to an entry inside a closed version opens it.
        });
        var openFromHash = function () {
            var id = decodeURIComponent(location.hash.slice(1));
            var el = id && document.getElementById(id);
            var d = el && (el.closest('.version-details') || el.querySelector(':scope > .version-details'));
            if (d && !d.open) { d.open = true; el.scrollIntoView(); }
        };
        window.addEventListener('hashchange', openFromHash);

        var allDetails = document.querySelectorAll('.version-details');
        if (!allDetails.length) return;
        openFromHash();

        // Update toggle text on open/close
        allDetails.forEach(function(details) {
            details.addEventListener('toggle', function() {
                var toggleText = details.querySelector('.version-toggle span');
                if (toggleText) {
                    toggleText.textContent = details.open ? 'Hide details' : 'Show details';
                }
            });
        });

        // Expand all / Collapse all buttons
        var expandBtn = document.getElementById('expand-all-versions');
        var collapseBtn = document.getElementById('collapse-all-versions');

        if (expandBtn) {
            expandBtn.addEventListener('click', function() {
                allDetails.forEach(function(d) { d.open = true; });
            });
        }
        if (collapseBtn) {
            collapseBtn.addEventListener('click', function() {
                allDetails.forEach(function(d) { d.open = false; });
            });
        }
    }

    document.addEventListener('DOMContentLoaded', initChangelogCollapse);

    // ============================================================
    // Dark Mode Toggle (Optional)
    // ============================================================

    function initDarkMode() {
        const toggle = document.querySelector('.dark-mode-toggle');
        if (!toggle) return;

        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const savedTheme = localStorage.getItem('theme');

        if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
            document.documentElement.classList.add('dark');
        }

        toggle.addEventListener('click', () => {
            document.documentElement.classList.toggle('dark');
            const isDark = document.documentElement.classList.contains('dark');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
        });
    }

    document.addEventListener('DOMContentLoaded', initDarkMode);

    // ============================================================
    // Feedback Widget
    // ============================================================

    function initFeedback() {
        const feedbackBtns = document.querySelectorAll('.feedback-btn');

        feedbackBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const value = btn.dataset.value;
                const section = btn.closest('section')?.id || 'general';

                // Track feedback (would integrate with analytics)
                console.log('Feedback:', { section, value });

                // Visual feedback
                btn.classList.add('selected');
                btn.parentElement.querySelectorAll('.feedback-btn').forEach(b => {
                    if (b !== btn) b.classList.add('disabled');
                });

                // Thank user
                const thanks = document.createElement('span');
                thanks.textContent = ' Thanks for your feedback!';
                thanks.className = 'feedback-thanks';
                btn.parentElement.appendChild(thanks);
            });
        });
    }

    document.addEventListener('DOMContentLoaded', initFeedback);

    // ============================================================
    // Version Warning Banner
    // ============================================================

    function checkVersion() {
        const currentVersion = document.querySelector('meta[name="version"]')?.content;
        const latestVersion = '0.1.0'; // Would fetch from API in production

        if (currentVersion && currentVersion !== latestVersion) {
            const banner = document.createElement('div');
            banner.className = 'version-warning';
            banner.innerHTML = `
                <span>You're viewing documentation for version ${currentVersion}.
                <a href="/docs/latest/">Switch to latest (${latestVersion})</a></span>
                <button class="close-btn" aria-label="Dismiss">&times;</button>
            `;
            document.body.prepend(banner);

            banner.querySelector('.close-btn').addEventListener('click', () => {
                banner.remove();
            });
        }
    }

    document.addEventListener('DOMContentLoaded', checkVersion);

    // ============================================================
    // Context-Aware Sidebar
    // ============================================================

    function initContextAwareSidebar() {
        const sidebar = document.querySelector('.sidebar');
        if (!sidebar) return;

        // Determine current page context from URL
        const currentPath = window.location.pathname;
        const pageContext = detectPageContext(currentPath);

        // Apply context-specific visibility
        applyContextVisibility(sidebar, pageContext);

        // Make sidebar sections collapsible for cleaner navigation
        initSidebarCollapse(sidebar);
    }

    function detectPageContext(path) {
        if (path.includes('getting-started')) return 'getting-started';
        if (path.includes('business-guide')) return 'business-guide';
        if (path.includes('api-reference')) return 'api-reference';
        if (path.includes('concepts')) return 'concepts';
        return 'home';
    }

    function applyContextVisibility(sidebar, context) {
        const sections = sidebar.querySelectorAll('.sidebar-section');

        sections.forEach(section => {
            const title = section.querySelector('.sidebar-title');
            if (!title) return;

            const titleText = title.textContent.toLowerCase();

            // Determine relevance based on context
            let isRelevant = true;
            let isPrimary = false;

            // Pipeline sections (1-5) are always somewhat relevant
            const isPipelineSection = /^[1-5]\./.test(title.textContent.trim());

            switch (context) {
                case 'getting-started':
                    // Show all pipeline sections, highlight tutorials
                    isPrimary = titleText.includes('getting started') ||
                               titleText.includes('introduction');
                    break;

                case 'business-guide':
                    // Show all sections, highlight business topics
                    isPrimary = titleText.includes('business') ||
                               titleText.includes('regulatory') ||
                               titleText.includes('understanding');
                    break;

                case 'api-reference':
                    // Show all sections, emphasize API-related
                    isPrimary = titleText.includes('api') ||
                               titleText.includes('class') ||
                               isPipelineSection;
                    break;

                case 'concepts':
                    // Show all sections, emphasize theory
                    isPrimary = titleText.includes('concepts') ||
                               titleText.includes('fairness') ||
                               titleText.includes('statistical');
                    break;
            }

            // Apply visual styling
            if (isPrimary) {
                section.classList.add('sidebar-section-primary');
            }

            // Expand relevant sections by default
            const navList = section.querySelector('.sidebar-nav');
            if (navList && isPrimary) {
                navList.style.display = 'block';
                section.classList.add('expanded');
            }
        });
    }

    function initSidebarCollapse(sidebar) {
        const sectionTitles = sidebar.querySelectorAll('.sidebar-title');

        sectionTitles.forEach(title => {
            const section = title.parentElement;
            const navList = section.querySelector('.sidebar-nav');

            if (!navList) return;

            // Primary module sections (Overview links) always stay expanded.
            if (title.classList.contains('sidebar-title-primary')) {
                navList.style.display = 'block';
                section.classList.add('expanded');
                return;
            }

            // Sync inline style with computed state so the first click works
            const isInitExpanded = section.classList.contains('expanded');
            navList.style.display = isInitExpanded ? 'block' : 'none';

            // Make title clickable to toggle section
            title.style.cursor = 'pointer';
            title.setAttribute('role', 'button');
            title.setAttribute('aria-expanded', isInitExpanded);

            // Add toggle indicator
            const indicator = document.createElement('span');
            indicator.className = 'sidebar-toggle-indicator';
            indicator.innerHTML = isInitExpanded ? '−' : '+';
            indicator.style.cssText = 'float: right; font-weight: normal; color: var(--color-gray-400); font-size: 12px;';
            title.appendChild(indicator);

            title.addEventListener('click', (e) => {
                // Don't toggle if clicking a link
                if (e.target.tagName === 'A') return;

                const isExpanded = section.classList.contains('expanded');

                navList.style.display = isExpanded ? 'none' : 'block';
                indicator.innerHTML = isExpanded ? '+' : '−';
                title.setAttribute('aria-expanded', !isExpanded);
                section.classList.toggle('expanded', !isExpanded);
            });
        });
    }

    /**
     * Ensure the sidebar section containing `linkEl` is expanded.
     * Called by the scroll-based active-link tracker so the active
     * indicator is never hidden inside a collapsed section.
     */
    function ensureSidebarSectionExpanded(linkEl) {
        if (!linkEl) return;
        const section = linkEl.closest('.sidebar-section');
        if (!section) return;

        /* Already visible — nothing to do */
        if (section.classList.contains('expanded')) return;

        const navList = section.querySelector('.sidebar-nav');
        const title   = section.querySelector('.sidebar-title');
        if (!navList) return;

        navList.style.display = 'block';
        section.classList.add('expanded');
        if (title) {
            title.setAttribute('aria-expanded', 'true');
            const indicator = title.querySelector('.sidebar-toggle-indicator');
            if (indicator) indicator.innerHTML = '−';
        }
    }

    /* Expose helper globally so inline page scripts can call it */
    window.ensureSidebarSectionExpanded = ensureSidebarSectionExpanded;

    document.addEventListener('DOMContentLoaded', initContextAwareSidebar);

    // ============================================================
    // Mermaid Lightbox with Zoom / Pan  (unified – single system)
    // ============================================================

    function initMermaidLightbox() {
        /* Prevent double-init (shared js may load on every page) */
        if (document.getElementById('mermaid-lightbox')) return;

        /* ---- Create lightbox DOM ---- */
        const lightboxHTML = `
            <div class="mermaid-lightbox" id="mermaid-lightbox">
                <div class="mermaid-lightbox-toolbar">
                    <span class="mermaid-lightbox-title">Diagram Viewer</span>
                    <div class="mermaid-lightbox-controls">
                        <button class="mermaid-lightbox-btn" data-action="zoom-out" title="Zoom out" aria-label="Zoom out">&minus;</button>
                        <span class="mermaid-lightbox-zoom-display">100%</span>
                        <button class="mermaid-lightbox-btn" data-action="zoom-in" title="Zoom in" aria-label="Zoom in">&plus;</button>
                        <button class="mermaid-lightbox-btn" data-action="reset" title="Reset view" aria-label="Reset view">&#8634;</button>
                        <button class="mermaid-lightbox-close" title="Close (Esc)" aria-label="Close">&times;</button>
                    </div>
                </div>
                <div class="mermaid-lightbox-canvas"></div>
                <div class="mermaid-lightbox-hint">Scroll to zoom &middot; Drag to pan &middot; Esc to close</div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', lightboxHTML);

        const lightbox   = document.getElementById('mermaid-lightbox');
        const canvas     = lightbox.querySelector('.mermaid-lightbox-canvas');
        const zoomLabel  = lightbox.querySelector('.mermaid-lightbox-zoom-display');
        const titleEl    = lightbox.querySelector('.mermaid-lightbox-title');
        const closeBtn   = lightbox.querySelector('.mermaid-lightbox-close');
        const controls   = lightbox.querySelectorAll('.mermaid-lightbox-btn');

        let scale = 1, panX = 0, panY = 0;
        let isDragging = false, dragStartX = 0, dragStartY = 0, dragPanX = 0, dragPanY = 0;
        let contentEl = null;   // the SVG clone in the canvas
        let initialScale = 1;
        const MIN_SCALE = 0.15, MAX_SCALE = 5, ZOOM_STEP = 0.15;

        /* Natural (un-scaled) width/height of the SVG clone – set by openLightbox */
        let naturalW = 0, naturalH = 0;

        function applyTransform() {
            if (!contentEl) return;
            /* Centre of the canvas in pixels */
            var cw = canvas.clientWidth  || window.innerWidth;
            var ch = canvas.clientHeight || window.innerHeight;
            /* Place the SVG so its centre aligns with canvas centre, then
               offset by pan, then scale from the SVG's own centre.       */
            var tx = cw / 2 - (naturalW / 2) * scale + panX;
            var ty = ch / 2 - (naturalH / 2) * scale + panY;
            contentEl.style.transform =
                'translate(' + tx + 'px, ' + ty + 'px) scale(' + scale + ')';
            zoomLabel.textContent = Math.round(scale * 100) + '%';
        }

        function resetView() {
            scale = initialScale; panX = 0; panY = 0;
            applyTransform();
        }

        /* ---- Open lightbox ---- */
        function openLightbox(mermaidEl) {
            /* Find the rendered SVG inside the container */
            const svg = mermaidEl.querySelector('svg') || (mermaidEl.tagName === 'svg' ? mermaidEl : null);
            if (!svg) return;

            canvas.innerHTML = '';
            const clone = svg.cloneNode(true);

            /* Strip all inline styles from the clone so page-level CSS
               scaling (e.g. mermaid-capped scale(0.88)) does not carry
               over.  Render at natural, unconstrained size so
               foreignObject text is never truncated. */
            clone.removeAttribute('style');
            clone.style.cssText =
                'position:absolute; top:0; left:0;' +
                'max-width:none !important; max-height:none !important;' +
                'overflow:visible; transform-origin:0 0; margin-bottom:0;';
            clone.setAttribute('overflow', 'visible');

            canvas.appendChild(clone);
            contentEl = clone;

            /* Title from nearest heading */
            let title = 'Diagram Viewer';
            const section = mermaidEl.closest('section, .card, article, .mermaid-container, .mermaid-zoomable');
            if (section) {
                const heading = section.querySelector('h1, h2, h3, h4, h5, h6');
                if (heading) title = heading.textContent.trim();
            }
            titleEl.textContent = title;

            lightbox.classList.add('open');
            document.body.style.overflow = 'hidden';

            /* Double-rAF: first frame lets the DOM lay out, second
               ensures getBoundingClientRect returns the real size.  */
            requestAnimationFrame(() => { requestAnimationFrame(() => {
                const rect  = clone.getBoundingClientRect();
                const origW = rect.width  || clone.scrollWidth  || 800;
                const origH = rect.height || clone.scrollHeight || 600;
                naturalW = origW;
                naturalH = origH;
                const vw = window.innerWidth  * 0.88;
                const vh = window.innerHeight * 0.82;
                const fitScale = Math.min(vw / origW, vh / origH, 1.0);
                initialScale = fitScale;
                scale = fitScale;
                panX = 0; panY = 0;
                applyTransform();
            }); });
        }

        /* ---- Close lightbox ---- */
        function closeLightbox() {
            lightbox.classList.remove('open');
            document.body.style.overflow = '';
            canvas.innerHTML = '';
            contentEl = null;
        }

        /* ---- Close button ---- */
        closeBtn.addEventListener('click', closeLightbox);

        /* ---- Click backdrop to close ---- */
        lightbox.addEventListener('click', (e) => {
            if (e.target === lightbox || e.target === canvas) closeLightbox();
        });

        /* ---- Escape key ---- */
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && lightbox.classList.contains('open')) closeLightbox();
        });

        /* ---- Toolbar zoom buttons ---- */
        controls.forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;
                if (action === 'zoom-in')  scale = Math.min(MAX_SCALE, scale * (1 + ZOOM_STEP));
                else if (action === 'zoom-out') scale = Math.max(MIN_SCALE, scale * (1 - ZOOM_STEP));
                else if (action === 'reset') { resetView(); return; }
                applyTransform();
            });
        });

        /* ---- Scroll-wheel zoom (towards cursor) ---- */
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            /* Cursor position relative to canvas centre */
            var mx = e.clientX - rect.left - rect.width / 2;
            var my = e.clientY - rect.top  - rect.height / 2;
            const oldScale = scale;
            if (e.deltaY < 0) scale = Math.min(MAX_SCALE, scale * (1 + ZOOM_STEP));
            else              scale = Math.max(MIN_SCALE, scale * (1 - ZOOM_STEP));
            const factor = scale / oldScale;
            panX = mx - factor * (mx - panX);
            panY = my - factor * (my - panY);
            applyTransform();
        }, { passive: false });

        /* ---- Drag to pan ---- */
        canvas.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            isDragging = true;
            dragStartX = e.clientX; dragStartY = e.clientY;
            dragPanX = panX; dragPanY = panY;
            canvas.classList.add('dragging');
        });
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            panX = dragPanX + (e.clientX - dragStartX);
            panY = dragPanY + (e.clientY - dragStartY);
            applyTransform();
        });
        document.addEventListener('mouseup', () => {
            isDragging = false;
            canvas.classList.remove('dragging');
        });

        /* ---- Touch pinch-zoom & pan ---- */
        let lastTouchDist = 0, lastTouchMid = null;
        canvas.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                isDragging = true;
                dragStartX = e.touches[0].clientX;
                dragStartY = e.touches[0].clientY;
                dragPanX = panX; dragPanY = panY;
            } else if (e.touches.length === 2) {
                isDragging = false;
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                lastTouchDist = Math.sqrt(dx * dx + dy * dy);
                lastTouchMid = {
                    x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
                    y: (e.touches[0].clientY + e.touches[1].clientY) / 2
                };
            }
        }, { passive: true });

        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && isDragging) {
                panX = dragPanX + (e.touches[0].clientX - dragStartX);
                panY = dragPanY + (e.touches[0].clientY - dragStartY);
                applyTransform();
            } else if (e.touches.length === 2 && lastTouchDist > 0) {
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const factor = dist / lastTouchDist;
                const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));
                const rect = canvas.getBoundingClientRect();
                const mx = lastTouchMid.x - rect.left;
                const my = lastTouchMid.y - rect.top;
                const sf = newScale / scale;
                panX = mx - sf * (mx - panX);
                panY = my - sf * (my - panY);
                scale = newScale;
                applyTransform();
                lastTouchDist = dist;
                lastTouchMid = {
                    x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
                    y: (e.touches[0].clientY + e.touches[1].clientY) / 2
                };
            }
        }, { passive: false });

        canvas.addEventListener('touchend', () => {
            isDragging = false;
            lastTouchDist = 0;
            lastTouchMid = null;
        });

        /* ---- Attach click-to-zoom handlers ----
           Covers every possible Mermaid wrapper pattern:
             .mermaid-zoomable .mermaid   (api-reference)
             .mermaid-container .mermaid  (concepts, business-guide)
             bare .mermaid                (landing page, loose diagrams)
             pre.mermaid                  (inline code-style)
        */
        function attachMermaidClickHandlers() {
            document.querySelectorAll(
                '.mermaid-zoomable .mermaid, .mermaid-container .mermaid, .mermaid, pre.mermaid'
            ).forEach(el => {
                if (el.dataset.lbAttached) return;
                el.dataset.lbAttached = 'true';
                el.style.cursor = 'zoom-in';
                el.addEventListener('click', (e) => {
                    e.stopPropagation();
                    openLightbox(el);
                });
            });
        }

        /* Run immediately, then re-run after Mermaid renders */
        attachMermaidClickHandlers();

        const observer = new MutationObserver(() => {
            setTimeout(attachMermaidClickHandlers, 100);
        });
        observer.observe(document.body, { childList: true, subtree: true });

        setTimeout(attachMermaidClickHandlers, 1000);
        setTimeout(attachMermaidClickHandlers, 3000);
    }

    document.addEventListener('DOMContentLoaded', initMermaidLightbox);

    // The Pre-Publication Review Modal that lived here was retired on
    // 2026-10-01; the header chip that briefly replaced it was removed the
    // same day at Daniel's request.

    // ============================================================
    // Export for external use
    // ============================================================

    window.vfairnessDocs = {
        initMermaid,
        highlightCode,
        initContextAwareSidebar,
        initMermaidLightbox
    };

})();

/* ── Reading Progress Bar (scroll depth) — same as docs.validant.ai ── */
(function initReadingProgress() {
    if (document.querySelector('.reading-progress')) return;
    var bar = document.createElement('div');
    bar.className = 'reading-progress';
    bar.innerHTML = '<div class="reading-progress-bar"></div>';
    document.body.appendChild(bar);
    var fill = bar.querySelector('.reading-progress-bar');

    function update() {
        var scrollTop = window.scrollY;
        var docHeight = document.documentElement.scrollHeight - window.innerHeight;
        var pct = docHeight > 0 ? Math.min(100, (scrollTop / docHeight) * 100) : 0;
        fill.style.width = pct + '%';
    }
    var ticking = false;
    window.addEventListener('scroll', function () {
        if (!ticking) { ticking = true; requestAnimationFrame(function () { update(); ticking = false; }); }
    }, { passive: true });
    window.addEventListener('resize', update, { passive: true });
    update();
})();
