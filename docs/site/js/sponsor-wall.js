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
 * is one line in one file and every wall updates. The same file carries the
 * academic & research partners, a separate collaboration band below the
 * sponsor tiers that is chosen by the team and never sold.
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
     in too. Fallbacks are the Swiss design values (validant.ai, the platform,
     the platform docs): ink, steel, muted, paper, card, tint, hairline. */
  --sw-ink:      var(--color-gray-900, #2e4057);
  --sw-text:     var(--color-gray-700, #4e6078);
  --sw-muted:    var(--color-gray-500, #5a6a78);
  --sw-sheet:    var(--surface-raised, #ffffff);
  --sw-sunk:     var(--surface-sunk, #f4f3f2);
  --sw-rule:     var(--rule, #d9dde2);
  --sw-rule-2:   var(--rule-strong, #7d8a98);
  --sw-accent:   var(--color-primary, #006686);
  --sw-strong:   var(--sw-ink);
  --sw-on-strong: var(--surface-raised, #ffffff);
  --sw-sans:     var(--font-sans, 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif);
}
/* Hosts whose dark mode is a class on <html> (the platform app) and that do
   not define the docs tokens get the Swiss dark values here, so the sheet
   follows: the dark card, never a navy box. */
:host-context(html.dark) {
  --sw-ink: var(--color-gray-900, #e6e9ee); --sw-text: var(--color-gray-700, #bcc6d2);
  --sw-muted: var(--color-gray-500, #9eabb9); --sw-sheet: var(--surface-raised, #1c1f22);
  --sw-sunk: var(--surface-sunk, #17191c); --sw-rule: var(--rule, #2b3644);
  --sw-rule-2: var(--rule-strong, #66768a); --sw-accent: var(--color-primary, #5fb6d3);
  --sw-on-strong: var(--surface-raised, #1c1f22);
}
:host-context(html[data-theme="dark"]) { --sw-on-strong: var(--surface-raised, #1c1f22); }

* { box-sizing: border-box; }

/* ── The sheet. Swiss (2026-10-12, issue #403 of the platform): one design on
   the platform, the vfairness site and the docs. A tile: the card surface,
   one 1px ink rule on top, no side frame, no gradient (the gradient belongs
   to the logo alone), square corners. ── */
.wall {
  position: relative;
  padding: clamp(1.5rem, 4cqi, 3rem);
  background: var(--sw-sheet);
  border-top: 1px solid var(--sw-ink);
  color: var(--sw-text);
  font-family: var(--sw-sans);
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
  font-variant-numeric: tabular-nums;
}

/* ── Masthead: editorial, left-aligned ──────────────────────────────── */
.head {
  display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
  gap: .75rem 3rem; align-items: end; margin-bottom: clamp(1.5rem, 3.5cqi, 2.5rem);
}
@container (max-width: 640px) { .head { grid-template-columns: minmax(0, 1fr); } }
/* The eyebrow is the Swiss annotation: sans bold, small capitals, muted, a
   short hairline after it. */
.eyebrow {
  display: inline-flex; align-items: center; gap: .75rem;
  font-size: .625rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; color: var(--sw-muted);
  margin-bottom: .875rem;
}
.eyebrow::after { content: ''; width: 24px; height: 1px; background: var(--sw-rule); }
h2.title {
  margin: 0; font-size: clamp(1.5625rem, 1rem + 2.5cqi, 2.4375rem);
  font-weight: 300; letter-spacing: -.02em; line-height: 1.1; color: var(--sw-ink);
}
.sub { margin: 0; max-width: 52ch; font-size: 1rem; line-height: 1.6; color: var(--sw-muted); }

/* The reason to want it: reach. A quiet list in the meta size, not pills. */
.reach {
  grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: .25rem 0;
  margin: .5rem 0 0; padding-top: .875rem; border-top: 1px solid var(--sw-rule);
  font-size: .8125rem; color: var(--sw-muted);
}
.reach span + span::before { content: '·'; margin: 0 .75rem; color: var(--sw-muted); }

/* ── Tier ───────────────────────────────────────────────────────────── */
.tier + .tier { margin-top: clamp(1.5rem, 3cqi, 2.25rem); }
.tier-head { display: flex; align-items: baseline; flex-wrap: wrap; gap: .25rem 1rem; margin-bottom: .625rem; }
.tier-name { font-size: 1rem; font-weight: 600; color: var(--sw-ink); }
/* The price is a muted label in the meta size, not a framed mono chip. */
.tier-from { font-size: .8125rem; color: var(--sw-muted); }
.tier-perk { margin: 0 0 .875rem; font-size: .8125rem; line-height: 1.6; color: var(--sw-muted); }
.rule { height: 1px; background: var(--sw-rule); margin-bottom: 1rem; }

/* Equal tiles: one size per tier, one hairline frame each, one gap. */
.grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); }
.tier[data-id="corporate"] .grid { grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }

/* ── A filled placement ─────────────────────────────────────────────── */
.card {
  position: relative; display: flex; align-items: center; justify-content: center;
  gap: .625rem; text-align: center; text-decoration: none;
  padding: 1rem; min-height: 80px;
  background: var(--sw-sheet);
  border: 1px solid var(--sw-rule);
  color: var(--sw-ink); font-weight: 600; font-size: 1rem;
  transition: border-color .2s ease;
}
.tier[data-id="corporate"] .card { min-height: 104px; }
a.card:hover { border-color: var(--sw-ink); }
a.card:focus-visible, a.p-invite:focus-visible { outline: 2px solid var(--sw-accent); outline-offset: 2px; }
.card img { max-width: 100%; max-height: 46px; object-fit: contain; }
/* A logo with a dark-theme variant: show one or the other, never both. */
.card .lg-dark { display: none; }
:host-context(html[data-theme="dark"]) .card.has-dark .lg-light,
:host-context(html.dark) .card.has-dark .lg-light { display: none; }
:host-context(html[data-theme="dark"]) .card.has-dark .lg-dark,
:host-context(html.dark) .card.has-dark .lg-dark { display: block; }
.card .role { position: absolute; bottom: .375rem; left: .625rem; font-size: .8125rem; font-weight: 400; color: var(--sw-muted); }
.card .initial {
  flex: none; width: 32px; height: 32px; border-radius: 50%;
  display: grid; place-items: center;
  border: 1px solid var(--sw-rule-2); color: var(--sw-ink); font-size: .8125rem; font-weight: 600;
}
.since { position: absolute; bottom: .375rem; right: .625rem; font-size: .8125rem; font-weight: 400; color: var(--sw-muted); }

/* ── The invitation. ONE per tier, never a field of empty boxes. It is a
   tile like the others (no accent bar, no tint): the open place is said in
   words, the price under it in the muted label. ── */
.invite {
  position: relative;
  display: flex; flex-direction: column; align-items: flex-start; justify-content: center;
  gap: .25rem; padding: 1rem; min-height: 80px;
  background: var(--sw-sheet);
  border: 1px solid var(--sw-rule);
  text-align: left;
}
.tier[data-id="corporate"] .invite { min-height: 104px; }
.invite b { font-size: 1rem; font-weight: 600; color: var(--sw-ink); }
.invite small { font-size: .8125rem; color: var(--sw-muted); }

/* Scarcity as a sentence, not as twenty grey rectangles: a tile of the same
   size whose label counts the rest. */
.remaining {
  display: flex; align-items: center; justify-content: center;
  padding: 1rem; min-height: 80px;
  border: 1px solid var(--sw-rule);
  font-size: .8125rem; color: var(--sw-muted);
}
.tier[data-id="corporate"] .remaining { min-height: 104px; }

/* ── Footer ─────────────────────────────────────────────────────────── */
.foot {
  margin-top: clamp(1.5rem, 3cqi, 2.25rem); padding-top: 1.25rem;
  border-top: 1px solid var(--sw-rule);
  display: flex; flex-wrap: wrap; align-items: center; gap: .875rem 1.25rem;
}
/* The call to action is the Swiss primary: an ink fill, square, the page
   ground as its label in dark mode. */
.cta {
  display: inline-flex; align-items: center; gap: .5rem;
  padding: .75rem 1.25rem;
  background: var(--sw-strong); color: var(--sw-on-strong); text-decoration: none;
  font-weight: 600; font-size: 1rem; border: 1px solid var(--sw-strong);
  transition: opacity .18s ease;
}
.cta:hover { opacity: .85; }
.cta:focus-visible { outline: 2px solid var(--sw-accent); outline-offset: 3px; }
.note { font-size: .8125rem; color: var(--sw-muted); }

/* ── Academic & research partners. A collaboration band, NOT a tier: no
   price, no slot count, no "your logo here". It sits below the sponsor call
   to action on the tint, so it never reads as something you can buy.
   Logos are greyscale until hovered, so a row of university marks stays calm. */
.partners {
  margin: clamp(1.5rem, 3.5cqi, 2.5rem) calc(-1 * clamp(1.5rem, 4cqi, 3rem)) calc(-1 * clamp(1.5rem, 4cqi, 3rem));
  padding: clamp(1.5rem, 3.5cqi, 2.25rem) clamp(1.5rem, 4cqi, 3rem);
  background: var(--sw-sunk); border-top: 1px solid var(--sw-rule);
}
.partners-head { display: flex; align-items: baseline; flex-wrap: wrap; gap: .25rem 1rem; margin-bottom: .625rem; }
.kind { font-size: .8125rem; color: var(--sw-muted); }
.partners .grid { grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
.partners .card { background: var(--sw-sheet); min-height: 88px; }
.partners .card img { filter: grayscale(1); opacity: .78; transition: filter .25s ease, opacity .25s ease; }
.partners a.card:hover img { filter: none; opacity: 1; }
.p-invite {
  display: flex; flex-direction: column; align-items: flex-start; justify-content: center; gap: .25rem;
  padding: 1rem; min-height: 88px; text-align: left; text-decoration: none;
  background: var(--sw-sheet); border: 1px solid var(--sw-rule); color: inherit;
  transition: border-color .2s ease;
}
a.p-invite:hover { border-color: var(--sw-ink); }
.p-invite b { font-size: 1rem; font-weight: 600; color: var(--sw-ink); }
.p-invite small { font-size: .8125rem; color: var(--sw-muted); }
.p-invite .go { font-size: .8125rem; font-weight: 600; color: var(--sw-accent); }

/* ── Compact variant, for docs landings ─────────────────────────────── */
:host([variant="compact"]) .tier-perk,
:host([variant="compact"]) .sub,
:host([variant="compact"]) .reach { display: none; }
:host([variant="compact"]) .wall { padding: clamp(1.25rem, 3cqi, 2rem); }
:host([variant="compact"]) .partners { margin: 1.25rem calc(-1 * clamp(1.25rem, 3cqi, 2rem)) calc(-1 * clamp(1.25rem, 3cqi, 2rem)); padding: 1.25rem clamp(1.25rem, 3cqi, 2rem); }
:host([variant="compact"]) .partners .card, :host([variant="compact"]) .p-invite { min-height: 66px; padding: .75rem; }
:host([variant="compact"]) .tier + .tier { margin-top: 1.125rem; }
:host([variant="compact"]) .card,
:host([variant="compact"]) .invite,
:host([variant="compact"]) .remaining { min-height: 64px; padding: .75rem; }
:host([variant="compact"]) .tier[data-id="corporate"] .card,
:host([variant="compact"]) .tier[data-id="corporate"] .invite,
:host([variant="compact"]) .tier[data-id="corporate"] .remaining { min-height: 80px; }

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

  #member(member, rel = "noopener sponsored") {
    // logo_dark (optional): a light-on-dark variant, swapped in by the host's
    // dark theme, so a black wordmark does not vanish on the dark sheet.
    const inner = member.logo
      ? `<img class="lg-light" src="${esc(member.logo)}" alt="${esc(member.name)}" loading="lazy">` +
        (member.logo_dark ? `<img class="lg-dark" src="${esc(member.logo_dark)}" alt="" aria-hidden="true" loading="lazy">` : "")
      : `<span class="initial" aria-hidden="true">${esc((member.name || "?").trim()[0].toUpperCase())}</span><span>${esc(member.name)}</span>`;
    const since = member.since ? `<span class="since">since ${esc(member.since)}</span>` : "";
    const role = member.role ? `<span class="role">${esc(member.role)}</span>` : "";
    const cls = `card${member.logo && member.logo_dark ? " has-dark" : ""}`;
    return member.url
      ? `<a class="${cls}" href="${esc(member.url)}" target="_blank" rel="${rel}">${inner}${role}${since}</a>`
      : `<div class="${cls}">${inner}${role}${since}</div>`;
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
      // advertise how empty the wall is; a single placement plus a count reads
      // as inventory with room on it. This is the whole design decision. Every
      // cell is the same tile (Swiss: equal tiles), so the invitation no
      // longer spans two columns.
      const cells = members.map((m) => this.#member(m));
      if (open > 0) {
        cells.push(`<div class="invite">
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

    // Academic & research partners: a collaboration band, decided by the team.
    // Deliberately unlike a tier: no price chip, no slot count, and the one
    // open cell asks for a conversation, not a payment.
    const p = data.partners;
    const partners = p ? (() => {
      const members = p.members || [];
      const inv = p.invite || null;
      const cells = members.map((m) => this.#member(m, "noopener"));
      if (inv) {
        const body = `<b>${esc(inv.title || "Research collaboration?")}</b>${inv.note ? `<small>${esc(inv.note)}</small>` : ""}${inv.href ? `<span class="go">${esc(inv.label || "Talk to us")} &rarr;</span>` : ""}`;
        const cls = "p-invite";
        cells.push(inv.href
          ? `<a class="${cls}" href="${esc(inv.href)}" target="_blank" rel="noopener">${body}</a>`
          : `<div class="${cls}">${body}</div>`);
      }
      return `
        <section class="partners" data-id="${esc(p.id || "academic")}">
          <div class="partners-head">
            <span class="tier-name">${esc(p.label || "Academic & research partners")}</span>
            ${p.kind ? `<span class="kind">${esc(p.kind)}</span>` : ""}
          </div>
          <div class="rule"></div>
          ${p.perk ? `<p class="tier-perk">${esc(p.perk)}</p>` : ""}
          <div class="grid">${cells.join("")}</div>
        </section>`;
    })() : "";

    this.shadowRoot.innerHTML = `<style>${TEMPLATE_STYLES}</style>
      <div class="wall">
        <div class="head">
          <div>
            <div class="eyebrow">Recognition</div>
            <h2 class="title">${esc(heading)}</h2>
          </div>
          <p class="sub">${esc(subheading)}</p>
          ${reach ? `<div class="reach">${reach}</div>` : ""}
        </div>
        ${tiers}
        ${cta}
        ${partners}
      </div>`;
  }
}

if (!customElements.get("sponsor-wall")) customElements.define("sponsor-wall", SponsorWall);
export default SponsorWall;
