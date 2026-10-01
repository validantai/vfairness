/**
 * <sponsor-wall> - the sponsor recognition surface, one implementation.
 *
 * WHY A CUSTOM ELEMENT. This wall has to appear on four surfaces built four
 * different ways: two hand-authored static pages here, a React app, and a
 * markdown-generated docs site. A React component could not serve the static
 * pages, and a build-time HTML partial could not serve React. A custom element
 * is the one thing all four can render with the same tag and the same markup, so
 * the wall cannot drift into four versions that look almost alike.
 *
 * Styles live in a shadow root so a host page's CSS cannot pull it out of shape.
 * CSS custom properties DO cross that boundary, which is the point: each host's
 * brand tokens flow in, and every token below has a fallback for hosts that do
 * not define it. Same component, correct on each site.
 *
 * The roster is data, not markup: docs/site/data/sponsors.json. Adding a sponsor
 * is one line in one file and every wall updates.
 *
 * Usage:
 *   <script type="module" src="/js/sponsor-wall.js"></script>
 *   <sponsor-wall src="/data/sponsors.json"></sponsor-wall>
 *   <sponsor-wall src="..." variant="compact" heading="Backed by"></sponsor-wall>
 *
 * Attributes: src, variant ("full" | "compact"), heading, subheading.
 */

const TEMPLATE_STYLES = `
:host {
  display: block;
  container-type: inline-size;

  /* Brand tokens, inherited from the host page when it defines them. Every one
     has a fallback, so the component is correct on a host that defines none.
     Custom properties cross the shadow boundary, so a host's dark theme flows
     in too. Fallbacks are the validant.ai Blanco values. */
  --sw-ink:      var(--color-gray-900, #1b2735);
  --sw-text:     var(--color-gray-700, #3a4b5e);
  --sw-muted:    var(--color-gray-500, #5a6972);
  --sw-sheet:    var(--surface-raised, #ffffff);
  --sw-sunk:     var(--surface-sunk, #f4f3f1);
  --sw-rule:     var(--rule, rgba(41, 59, 81, .14));
  --sw-rule-2:   var(--rule-strong, rgba(41, 59, 81, .28));
  --sw-accent:   var(--color-primary, #006686);
  --sw-accent-h: var(--color-primary-light, #00526b);
  --sw-on-accent: #ffffff;
  --sw-word:     linear-gradient(120deg, #43c9fe 0%, #41ba1b 100%);
  --sw-mono:     var(--font-mono, 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace);
}
:host-context(html[data-theme="dark"]) { --sw-on-accent: #0d1a24; }
/* Hosts whose dark mode is a class on <html> (the platform app) and that do
   not define the docs tokens get dark fallbacks here, so the sheet follows. */
:host-context(html.dark) {
  --sw-ink: var(--color-gray-900, #eef2f6); --sw-text: var(--color-gray-700, #c3ccd5);
  --sw-muted: var(--color-gray-500, #98a5b1); --sw-sheet: var(--surface-raised, #152030);
  --sw-sunk: var(--surface-sunk, #17222e); --sw-rule: var(--rule, rgba(211, 220, 229, .13));
  --sw-rule-2: var(--rule-strong, rgba(211, 220, 229, .26)); --sw-accent: var(--color-primary, #5cc3e0);
  --sw-accent-h: var(--color-primary-light, #8dd8ee); --sw-on-accent: #0d1a24;
}

* { box-sizing: border-box; }

/* ── The sheet. Blanco (2026-10-01): a paper sheet with hairline rules and
   square corners, like every other surface on the site. It replaced a dark
   rounded plaque that was the one rounded, dark object on otherwise square,
   light pages. An empty wall still reads as inventory, not absence: that is
   carried by the single invitation per tier below, not by the colour. ── */
.wall {
  position: relative;
  padding: clamp(1.75rem, 4cqi, 3rem);
  background: var(--sw-sheet);
  border: 1px solid var(--sw-rule);
  color: var(--sw-text);
  font-family: var(--font-sans, 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif);
  /* Shadow DOM does NOT stop INHERITED properties. The offering page centres its
     recognition block, and that text-align crossed the boundary and centred the
     tier copy, so the wall looked different here than anywhere else. Reset the
     inheritable ones at the root and let each part opt back in. Caught by
     rendering it in the real page rather than in a bare harness. */
  text-align: left;
  line-height: 1.5;
  letter-spacing: normal;
  font-style: normal;
  text-transform: none;
}
/* The brand gradient, once, as a rail along the top edge. */
.wall::before {
  content: ''; position: absolute; inset: -1px -1px auto -1px; height: 2px;
  background: var(--sw-word);
}

/* ── Masthead: editorial, left-aligned ──────────────────────────────── */
.head {
  display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
  gap: .75rem 3rem; align-items: end; margin-bottom: clamp(1.75rem, 3.5cqi, 2.5rem);
}
@container (max-width: 640px) { .head { grid-template-columns: minmax(0, 1fr); } }
.eyebrow {
  font-family: var(--sw-mono); font-size: .6875rem; font-weight: 500;
  letter-spacing: .14em; text-transform: uppercase; color: var(--sw-muted);
  margin-bottom: .875rem;
}
h2.title {
  margin: 0; font-size: clamp(1.75rem, 1rem + 3cqi, 2.5rem);
  font-weight: 800; letter-spacing: -.035em; line-height: 1.05; color: var(--sw-ink);
}
.title em {
  font-style: italic; font-weight: 300; padding-right: .06em;
  background: var(--sw-word);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.sub { margin: 0; max-width: 52ch; font-size: .9375rem; line-height: 1.65; color: var(--sw-muted); }

/* The reason to want it: reach. Stated as a quiet list, not as pills. */
.reach {
  grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: .25rem 0;
  margin: .5rem 0 0; padding-top: .875rem; border-top: 1px solid var(--sw-rule);
  font-family: var(--sw-mono); font-size: .6875rem; letter-spacing: .1em; text-transform: uppercase; color: var(--sw-muted);
}
.reach span + span::before { content: '·'; margin: 0 .75rem; color: var(--sw-rule-2); }

/* ── Tier ───────────────────────────────────────────────────────────── */
.tier + .tier { margin-top: clamp(1.5rem, 3cqi, 2.25rem); }
.tier-head { display: flex; align-items: baseline; flex-wrap: wrap; gap: .75rem; margin-bottom: .625rem; }
.tier-name { font-size: 1rem; font-weight: 700; letter-spacing: -.01em; color: var(--sw-ink); }
.tier-from {
  font-family: var(--sw-mono); font-size: .625rem; font-weight: 500; letter-spacing: .1em;
  text-transform: uppercase; color: var(--sw-text);
  padding: .125rem .5rem; border: 1px solid var(--sw-rule-2);
  font-variant-numeric: tabular-nums;
}
.tier-perk { margin: 0 0 .875rem; font-size: .8125rem; line-height: 1.6; color: var(--sw-muted); }
.rule { height: 1px; background: var(--sw-rule); margin-bottom: .875rem; }

.grid { display: grid; gap: .625rem; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); }
.tier[data-id="corporate"] .grid { grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }

/* ── A filled placement ─────────────────────────────────────────────── */
.card {
  position: relative; display: flex; align-items: center; justify-content: center;
  gap: .625rem; text-align: center; text-decoration: none;
  padding: 1rem .875rem; min-height: 78px;
  background: var(--sw-sheet);
  border: 1px solid var(--sw-rule);
  color: var(--sw-ink); font-weight: 600; font-size: .875rem;
  transition: border-color .2s ease;
}
.tier[data-id="corporate"] .card { min-height: 100px; }
a.card:hover { border-color: var(--sw-ink); }
.card img { max-width: 100%; max-height: 46px; object-fit: contain; }
.card .initial {
  flex: none; width: 30px; height: 30px; border-radius: 50%;
  display: grid; place-items: center;
  border: 1px solid var(--sw-rule-2); color: var(--sw-ink); font-size: .8125rem; font-weight: 800;
}
.since { position: absolute; bottom: .375rem; right: .625rem; font-family: var(--sw-mono); font-size: .5625rem; letter-spacing: .06em; color: var(--sw-muted); }

/* ── The invitation. ONE per tier, never a field of empty boxes. ─────── */
.invite {
  position: relative;
  display: flex; flex-direction: column; align-items: flex-start; justify-content: center;
  gap: .375rem; padding: 1rem 1.125rem; min-height: 78px;
  background: var(--sw-sunk);
  border: 1px solid var(--sw-rule-2);
  border-left: 2px solid var(--sw-accent);
  text-align: left;
}
.tier[data-id="corporate"] .invite { min-height: 100px; }
/* With no members yet, one invitation next to one count leaves half the row
   empty. Let the invitation take that space so the tier reads as composed
   rather than as a row that ran out of content. */
.invite.lead { grid-column: span 2; }
@container (max-width: 460px) { .invite.lead { grid-column: auto; } }
.invite b { font-size: .9375rem; font-weight: 700; letter-spacing: -.01em; color: var(--sw-ink); }
.invite small { font-family: var(--sw-mono); font-size: .625rem; letter-spacing: .1em; text-transform: uppercase; color: var(--sw-muted); }

/* Scarcity as a sentence, not as twenty grey rectangles. */
.remaining {
  display: flex; align-items: center; justify-content: center;
  padding: 1rem; min-height: 78px;
  border: 1px dashed var(--sw-rule-2);
  font-family: var(--sw-mono); font-size: .6875rem; font-weight: 500; letter-spacing: .1em;
  text-transform: uppercase; color: var(--sw-muted);
}
.tier[data-id="corporate"] .remaining { min-height: 100px; }

/* ── Footer ─────────────────────────────────────────────────────────── */
.foot {
  margin-top: clamp(1.5rem, 3cqi, 2.25rem); padding-top: 1.25rem;
  border-top: 1px solid var(--sw-rule);
  display: flex; flex-wrap: wrap; align-items: center; gap: .875rem 1.25rem;
}
.cta {
  display: inline-flex; align-items: center; gap: .5rem;
  padding: .75rem 1.375rem;
  background: var(--sw-accent); color: var(--sw-on-accent); text-decoration: none;
  font-weight: 600; font-size: .875rem; border: 1px solid var(--sw-accent);
  transition: background .18s ease, border-color .18s ease;
}
.cta:hover { background: var(--sw-accent-h); border-color: var(--sw-accent-h); }
.cta:focus-visible { outline: 2px solid var(--sw-accent); outline-offset: 3px; }
.note { font-size: .8125rem; color: var(--sw-muted); }

/* ── Compact variant, for docs landings ─────────────────────────────── */
:host([variant="compact"]) .tier-perk,
:host([variant="compact"]) .sub,
:host([variant="compact"]) .reach { display: none; }
:host([variant="compact"]) .wall { padding: clamp(1.25rem, 3cqi, 2rem); }
:host([variant="compact"]) .tier + .tier { margin-top: 1.125rem; }
:host([variant="compact"]) .card,
:host([variant="compact"]) .invite,
:host([variant="compact"]) .remaining { min-height: 62px; padding: .75rem; }
:host([variant="compact"]) .tier[data-id="corporate"] .card,
:host([variant="compact"]) .tier[data-id="corporate"] .invite,
:host([variant="compact"]) .tier[data-id="corporate"] .remaining { min-height: 76px; }

@container (max-width: 460px) {
  .grid, .tier[data-id="corporate"] .grid { grid-template-columns: repeat(auto-fill, minmax(132px, 1fr)); }
}
`;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

class SponsorWall extends HTMLElement {
  static get observedAttributes() { return ["src", "variant", "heading", "subheading"]; }

  connectedCallback() {
    if (!this.shadowRoot) this.attachShadow({ mode: "open" });
    this.#load();
  }
  attributeChangedCallback() { if (this.shadowRoot) this.#load(); }

  async #load() {
    const src = this.getAttribute("src") || "/data/sponsors.json";
    let data = null;
    try {
      const res = await fetch(src, { cache: "no-cache" });
      if (res.ok) data = await res.json();
    } catch { /* fall through to the empty roster below */ }
    // A failed fetch must still render the wall. An invitation with no names is
    // the honest state today anyway, and a blank gap on the page would be worse.
    this.#render(data || { tiers: [], cta: null });
  }

  #member(member) {
    const inner = member.logo
      ? `<img src="${esc(member.logo)}" alt="${esc(member.name)}" loading="lazy">`
      : `<span class="initial" aria-hidden="true">${esc((member.name || "?").trim()[0].toUpperCase())}</span><span>${esc(member.name)}</span>`;
    const since = member.since ? `<span class="since">since ${esc(member.since)}</span>` : "";
    return member.url
      ? `<a class="card" href="${esc(member.url)}" target="_blank" rel="noopener sponsored">${inner}${since}</a>`
      : `<div class="card">${inner}${since}</div>`;
  }

  #render(data) {
    const heading = this.getAttribute("heading") || "Sponsors get the credit";
    const subheading = this.getAttribute("subheading")
      || "Patrons are named here, wherever this wall appears, and in the project credits. Corporate Patrons show their logo. Prefer to stay anonymous? That works too.";

    const tiers = (data.tiers || []).map((tier) => {
      const members = tier.members || [];
      const slots = Math.max(tier.slots || 0, members.length);
      const open = Math.max(slots - members.length, 0);

      // ONE invitation, never a field of empty boxes. Twenty grey rectangles
      // advertise how empty the wall is; a single lit placement plus a count
      // reads as inventory with room on it. This is the whole design decision.
      const cells = members.map((m) => this.#member(m));
      if (open > 0) {
        cells.push(`<div class="invite${members.length === 0 ? " lead" : ""}">
            <b>${tier.id === "corporate" ? "Your logo here" : "Your name here"}</b>
            <small>${esc(tier.from || "")}</small>
          </div>`);
        if (open > 1) {
          cells.push(`<div class="remaining">${open - 1} more open</div>`);
        }
      }

      return `
        <section class="tier" data-id="${esc(tier.id)}">
          <div class="tier-head">
            <span class="tier-name">${esc(tier.label)}</span>
            ${tier.from ? `<span class="tier-from">${esc(tier.from)}</span>` : ""}
          </div>
          <div class="rule"></div>
          ${tier.perk ? `<p class="tier-perk">${esc(tier.perk)}</p>` : ""}
          <div class="grid">${cells.join("")}</div>
        </section>`;
    }).join("");

    // Reach is the actual reason to want a placement, so name the surfaces
    // rather than leaving the reader to guess how many people see this.
    const reach = (data.reach || []).map((r) => `<span>${esc(r)}</span>`).join("");

    const cta = data.cta
      ? `<div class="foot">
           <a class="cta" href="${esc(data.cta.href)}">${esc(data.cta.label)}
             <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
               <path d="M6 3l5 5-5 5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
             </svg>
           </a>
           ${data.cta.note ? `<span class="note">${esc(data.cta.note)}</span>` : ""}
         </div>`
      : "";

    this.shadowRoot.innerHTML = `<style>${TEMPLATE_STYLES}</style>
      <div class="wall">
        <div class="head">
          <div>
            <div class="eyebrow">Recognition</div>
            <h2 class="title">${esc(heading).replace(/credit/i, "<em>credit</em>")}</h2>
          </div>
          <p class="sub">${esc(subheading)}</p>
          ${reach ? `<div class="reach">${reach}</div>` : ""}
        </div>
        ${tiers}
        ${cta}
      </div>`;
  }
}

if (!customElements.get("sponsor-wall")) customElements.define("sponsor-wall", SponsorWall);
export default SponsorWall;
