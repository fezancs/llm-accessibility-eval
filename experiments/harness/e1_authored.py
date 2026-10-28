"""Authored base pages B05-B12 for the E1 repair/audit corpus (tasks/e1_repair_corpus.yaml).

Written for this study with LLM assistance; the author reviews them before the freeze.
Every page is self-contained (inline CSS/JS, local images) and contains a target for every
operator M01-M12, each with a stable, descriptive id. Run once, then format with prettier:

    python -m harness.e1_authored && npx prettier --write data/corpus/base/B0[5-9].html data/corpus/base/B1[0-2].html
"""
from __future__ import annotations

from . import config

OUT = config.DATA / "corpus" / "base"

CSS = """
:root { --ink: #1f2933; --muted: #4a5560; --brand: #0b5394; --brand-dark: #073763; --panel: #f3f6f9; --line: #c9d2db; --danger: #b42318; }
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; font-size: 1rem; line-height: 1.6; color: var(--ink); background: #fff; }
a { color: var(--brand); }
a:hover { color: var(--brand-dark); }
:focus-visible { outline: 3px solid #1d4ed8; outline-offset: 2px; }
.skip-link { position: absolute; left: -999px; top: 0; background: #fff; color: var(--brand-dark); padding: 0.5rem 1rem; z-index: 10; }
.skip-link:focus { left: 1rem; }
.visually-hidden { position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0; }
.wrap { max-width: 72rem; margin: 0 auto; padding: 0 1.25rem; }
.site-header { background: var(--brand-dark); color: #fff; }
.header-inner { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem; padding: 1rem 0; }
.logo { font-size: 1.5rem; font-weight: 700; color: #fff; text-decoration: none; }
.main-nav ul { list-style: none; display: flex; flex-wrap: wrap; gap: 1.25rem; margin: 0; padding: 0; }
.main-nav a { color: #fff; font-weight: 600; text-decoration: none; padding: 0.25rem 0; display: inline-block; }
.main-nav a[aria-current="page"] { text-decoration: underline; text-underline-offset: 0.3em; }
.site-header :focus-visible { outline-color: #fff; }
main { padding: 2rem 0 3rem; }
h1 { font-size: 2rem; line-height: 1.25; margin: 0 0 1rem; }
h2 { font-size: 1.5rem; line-height: 1.3; margin: 2rem 0 0.75rem; }
h3 { font-size: 1.2rem; line-height: 1.35; margin: 1.25rem 0 0.5rem; }
img { max-width: 100%; height: auto; }
.btn { display: inline-flex; align-items: center; gap: 0.5rem; min-height: 2.75rem; padding: 0.55rem 1.1rem; border: 2px solid var(--brand); border-radius: 0.4rem; background: var(--brand); color: #fff; font: inherit; font-weight: 600; cursor: pointer; text-decoration: none; }
.btn-secondary { background: #fff; color: var(--brand); }
.icon-btn { display: inline-flex; align-items: center; justify-content: center; width: 2.75rem; height: 2.75rem; border: 2px solid #5f6b76; border-radius: 0.4rem; background: #fff; color: var(--ink); cursor: pointer; }
.icon-btn svg, .icon-link svg { width: 1.25rem; height: 1.25rem; }
.icon-link { display: inline-flex; width: 2.75rem; height: 2.75rem; align-items: center; justify-content: center; border-radius: 50%; background: var(--brand-dark); color: #fff; }
label, legend { display: block; font-weight: 600; margin-bottom: 0.25rem; }
fieldset { border: 1px solid var(--line); border-radius: 0.4rem; padding: 0.75rem 1rem 1rem; margin: 0 0 1.1rem; }
.choice { display: flex; align-items: center; gap: 0.5rem; margin: 0.35rem 0; }
.choice label { display: inline; font-weight: 400; margin: 0; }
input[type="checkbox"], input[type="radio"] { width: 1.5rem; height: 1.5rem; margin: 0; }
input[type="text"], input[type="email"], input[type="tel"], input[type="date"], input[type="time"], input[type="password"], input[type="search"], select, textarea { width: 100%; max-width: 28rem; min-height: 2.75rem; padding: 0.5rem 0.7rem; border: 2px solid #5f6b76; border-radius: 0.35rem; font: inherit; color: var(--ink); background: #fff; }
.field { margin-bottom: 1.1rem; }
.hint { color: var(--muted); margin: 0 0 0.35rem; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 0.5rem; padding: 1rem 1.25rem; margin: 1rem 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: 1.25rem; padding: 0; list-style: none; }
.card { border: 1px solid var(--line); border-radius: 0.5rem; padding: 1.1rem; background: #fff; }
.card-head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.card-head h3 { margin: 0; }
table { border-collapse: collapse; width: 100%; max-width: 48rem; }
caption { text-align: left; font-weight: 700; padding: 0.5rem 0; }
th, td { border: 1px solid var(--line); padding: 0.5rem 0.75rem; text-align: left; }
thead th { background: var(--panel); }
.error-text { color: var(--danger); font-weight: 600; margin: 0.25rem 0 0; }
.error-summary { border: 3px solid var(--danger); border-radius: 0.4rem; padding: 1rem 1.25rem; margin-bottom: 1.5rem; }
.site-footer { background: #e9eef3; border-top: 1px solid var(--line); padding: 2rem 0; margin-top: 2rem; }
.footer-grid { display: flex; flex-wrap: wrap; gap: 2rem; justify-content: space-between; }
.footer-links { list-style: none; padding: 0; margin: 0; }
.footer-links li { margin: 0.35rem 0; }
.site-footer a { color: var(--brand-dark); }
.site-footer .icon-link { color: #fff; }
.social { list-style: none; display: flex; gap: 0.75rem; padding: 0; margin: 0.75rem 0 0; }
"""

ICONS = {
    "search": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><circle cx="10" cy="10" r="6" fill="none" stroke="currentColor" stroke-width="2.5"/><path d="M15 15l6 6" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/></svg>',
    "arrow": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><path d="M5 12h12M12 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.5"/></svg>',
    "print": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><path d="M6 9V3h12v6M6 17H4v-6h16v6h-2M7 14h10v7H7z" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
    "share": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><circle cx="18" cy="5" r="3" fill="currentColor"/><circle cx="6" cy="12" r="3" fill="currentColor"/><circle cx="18" cy="19" r="3" fill="currentColor"/><path d="M8.6 13.5l6.8 4M15.4 6.5l-6.8 4" stroke="currentColor" stroke-width="2"/></svg>',
    "eye": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>',
    "edit": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16z" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
    "trash": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
    "clear": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2.5"/></svg>',
    "calendar": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M3 10h18M8 3v4M16 3v4" stroke="currentColor" stroke-width="2"/></svg>',
    "facebook": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><path d="M14 8h3V4h-3c-2.8 0-4 1.8-4 4v2H7v4h3v8h4v-8h3l1-4h-4V8.5c0-.3.2-.5.5-.5z" fill="currentColor"/></svg>',
    "x": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><path d="M4 4l16 16M20 4L4 20" stroke="currentColor" stroke-width="3"/></svg>',
    "instagram": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="4" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3.5" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
    "youtube": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><rect x="2" y="5" width="20" height="14" rx="4" fill="currentColor"/><path d="M10 9l5 3-5 3z" fill="#fff"/></svg>',
    "info": '<svg aria-hidden="true" focusable="false" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 11v6M12 7v1" stroke="currentColor" stroke-width="2.5"/></svg>',
}

NAV = [("index.html", "Home"), ("services.html", "Services"), ("appointments.html", "Appointments"),
       ("locations.html", "Locations"), ("contact.html", "Contact")]

# Shared toggle script: a <button> whose click shows/hides a panel. The handler is attached
# by id, so it still runs if operator M07 turns the button into a <div>.
TOGGLE_JS = """
function wireToggle(buttonId, panelId) {
  var button = document.getElementById(buttonId);
  var panel = document.getElementById(panelId);
  button.addEventListener("click", function () {
    panel.hidden = !panel.hidden;
    if (button.tagName === "BUTTON") { button.setAttribute("aria-expanded", String(!panel.hidden)); }
  });
}
"""


def shell(title: str, current: str, body: str, script: str = "", header_extra: str = "", footer_extra: str = "") -> str:
    nav = "\n".join(
        f'<li><a href="{href}"{" aria-current=\"page\"" if label == current else ""}>{label}</a></li>'
        for href, label in NAV)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<a class="skip-link" href="#main">Skip to main content</a>
<header class="site-header">
<div class="wrap header-inner">
<a class="logo" href="index.html">CityCare</a>
<nav class="main-nav" aria-label="Main">
<ul>
{nav}
</ul>
</nav>
{header_extra}
</div>
</header>
<main id="main">
<div class="wrap">
{body}
</div>
</main>
<footer class="site-footer">
<div class="wrap footer-grid">
<div>
<h2 class="visually-hidden">Contact CityCare</h2>
<p id="footer-address">CityCare, 1 Market Street, Old Town<br>Phone: <a href="tel:+441234567890">01234 567 890</a><br>Email: <a href="mailto:hello@citycare.example">hello@citycare.example</a></p>
<ul class="social">
<li><a id="social-facebook" class="icon-link" href="https://facebook.com/citycare" aria-label="CityCare on Facebook">{ICONS["facebook"]}</a></li>
<li><a id="social-x" class="icon-link" href="https://x.com/citycare" aria-label="CityCare on X">{ICONS["x"]}</a></li>
<li><a id="social-instagram" class="icon-link" href="https://instagram.com/citycare" aria-label="CityCare on Instagram">{ICONS["instagram"]}</a></li>
<li><a id="social-youtube" class="icon-link" href="https://youtube.com/citycare" aria-label="CityCare on YouTube">{ICONS["youtube"]}</a></li>
</ul>
</div>
<ul class="footer-links">
<li><a id="footer-privacy" href="privacy.html">Privacy notice</a></li>
<li><a id="footer-accessibility" href="accessibility.html">Accessibility statement</a></li>
<li><a id="footer-terms" href="terms.html">Terms of use</a></li>
<li><a id="footer-careers" href="careers.html">Work for CityCare</a></li>
</ul>
{footer_extra}
</div>
</footer>
<script>{TOGGLE_JS}{script}</script>
</body>
</html>
"""


PAGES: dict[str, dict] = {}

PAGES["B05"] = dict(
    title="CityCare: health care for everyone in our city", current="Home",
    header_extra=f"""<form class="site-search" role="search" action="search.html">
<label for="site-search-input" class="visually-hidden">Search CityCare</label>
<div style="display:flex;gap:0.5rem;">
<input type="search" id="site-search-input" name="q" style="max-width:16rem;">
<button type="submit" id="search-submit" class="icon-btn" aria-label="Search">{ICONS["search"]}</button>
</div>
</form>""",
    body=f"""<section aria-labelledby="hero-heading">
<h1 id="hero-heading">Care for everyone in our city</h1>
<img id="hero-photo" src="images/cc-hero-oldtown.png" width="640" height="360" alt="The Old Town clinic entrance, with a wide ramp beside the front steps">
<p id="hero-text">CityCare runs five public health clinics across the city. Our nurses, doctors and pharmacists see more than two thousand people every week, and every appointment is free at the point of use.</p>
<p><a class="btn" id="hero-book" href="appointments.html">Book an appointment</a>
<button type="button" class="btn btn-secondary" id="hours-toggle" aria-expanded="false" aria-controls="hours-panel">Show opening hours</button></p>
<div id="hours-panel" class="panel" hidden>
<p id="hours-text">Most clinics are open Monday to Friday from 8:00 to 18:00 and on Saturday morning from 9:00 to 13:00.</p>
</div>
</section>
<section aria-labelledby="services-heading">
<h2 id="services-heading">Our services</h2>
<ul class="cards">
<li class="card"><div class="card-head"><h3>Vaccinations</h3><a id="service-vaccinations-link" class="icon-link" href="vaccinations.html" aria-label="Vaccinations: find out more">{ICONS["arrow"]}</a></div>
<p id="vaccinations-text">Flu, COVID-19 and travel vaccinations for adults and children, with evening sessions at Riverside.</p></li>
<li class="card"><div class="card-head"><h3>Check-ups</h3><a id="service-checkups-link" class="icon-link" href="checkups.html" aria-label="Check-ups: find out more">{ICONS["arrow"]}</a></div>
<p id="checkups-text">Free health checks for adults aged 40 to 74, including blood pressure and cholesterol tests.</p></li>
<li class="card"><div class="card-head"><h3>Pharmacy</h3><a id="service-pharmacy-link" class="icon-link" href="pharmacy.html" aria-label="Pharmacy: find out more">{ICONS["arrow"]}</a></div>
<p id="pharmacy-text">Collect prescriptions and get advice on minor illnesses from our pharmacists, without an appointment.</p></li>
</ul>
</section>
<section aria-labelledby="news-heading">
<h2 id="news-heading">Latest news</h2>
<ul class="cards">
<li class="card"><img id="news-flu-photo" src="images/cc-news-flu.png" width="320" height="180" alt="A nurse preparing a flu vaccination at a clinic desk">
<h3><a href="news-flu.html">Flu vaccinations now available</a></h3><p>Published 2 October</p></li>
<li class="card"><img id="news-evening-photo" src="images/cc-news-evening.png" width="320" height="180" alt="Riverside clinic lit up in the evening">
<h3><a href="news-evening.html">Riverside clinic extends evening hours</a></h3><p>Published 18 September</p></li>
<li class="card"><img id="news-physio-photo" src="images/cc-news-physio.png" width="320" height="180" alt="A physiotherapist helping a patient with a knee exercise">
<h3><a href="news-physio.html">New physiotherapy service in Old Town</a></h3><p>Published 5 September</p></li>
</ul>
</section>""",
    footer_extra="""<form id="newsletter-form" action="subscribe.html">
<h2 style="font-size:1.2rem;margin-top:0;">Newsletter</h2>
<div class="field"><label for="newsletter-email">Email address</label>
<input type="email" id="newsletter-email" name="email" autocomplete="email"></div>
<button type="submit" class="btn" id="newsletter-subscribe">Subscribe</button>
</form>""",
    script='wireToggle("hours-toggle", "hours-panel");',
)

PAGES["B06"] = dict(
    title="Request an appointment - CityCare", current="Appointments",
    body=f"""<h1>Request an appointment</h1>
<p id="form-intro">Use this form to ask for an appointment at any CityCare clinic. We will contact you within two working days to confirm a time. Fields marked "required" must be filled in.</p>
<img id="reception-photo" src="images/cc-reception.png" width="640" height="360" alt="The reception desk at Riverside clinic, with a lowered counter section for wheelchair users">
<div id="error-summary" class="error-summary" tabindex="-1" hidden>
<h2 id="error-summary-heading">There is a problem</h2>
<ul id="error-list"></ul>
</div>
<form id="appointment-form" novalidate>
<h2 id="details-heading">Your details</h2>
<div class="field"><label for="full-name">Full name (required)</label>
<input type="text" id="full-name" name="name" autocomplete="name" aria-describedby="full-name-error">
<p id="full-name-error" class="error-text" hidden></p></div>
<div class="field"><label for="dob">Date of birth (required)</label>
<p class="hint" id="dob-hint">For example, 27 03 1985</p>
<input type="date" id="dob" name="dob" autocomplete="bday" aria-describedby="dob-hint dob-error">
<p id="dob-error" class="error-text" hidden></p></div>
<div class="field"><label for="phone">Phone number</label>
<input type="tel" id="phone" name="phone" autocomplete="tel"></div>
<div class="field"><label for="email">Email address (required)</label>
<input type="email" id="email" name="email" autocomplete="email" aria-describedby="email-error">
<p id="email-error" class="error-text" hidden></p></div>
<h2 id="visit-heading">Your visit</h2>
<div class="field"><label for="clinic">Preferred clinic (required)</label>
<select id="clinic" name="clinic" aria-describedby="clinic-error">
<option value="">Choose a clinic</option><option>Northside</option><option>Riverside</option><option>Old Town</option><option>Harbour</option>
</select>
<p id="clinic-error" class="error-text" hidden></p></div>
<fieldset>
<legend>Consultation type</legend>
<div class="choice"><input type="radio" id="type-person" name="type" value="person" checked><label for="type-person">In person</label></div>
<div class="choice"><input type="radio" id="type-video" name="type" value="video"><label for="type-video">Video call</label></div>
<div class="choice"><input type="radio" id="type-phone" name="type" value="phone"><label for="type-phone">Phone call</label></div>
</fieldset>
<div class="field"><label for="notes">Notes for the clinic</label>
<textarea id="notes" name="notes" rows="4"></textarea></div>
<div class="choice"><input type="checkbox" id="privacy" name="privacy" aria-describedby="privacy-error"><label for="privacy">I agree to the <a href="privacy.html">privacy notice</a> (required)</label></div>
<p id="privacy-error" class="error-text" hidden></p>
<p><button type="submit" class="btn" id="appointment-submit">Request appointment</button>
<button type="button" class="icon-btn" id="print-form" aria-label="Print this form">{ICONS["print"]}</button></p>
</form>
<p><button type="button" class="btn btn-secondary" id="next-steps-toggle" aria-expanded="false" aria-controls="next-steps-panel">What happens next?</button></p>
<div id="next-steps-panel" class="panel" hidden>
<p id="next-steps-text">A member of the clinic team will phone or email you to agree a time. If you need an interpreter or step-free access, tell us in the notes.</p>
</div>""",
    script="""wireToggle("next-steps-toggle", "next-steps-panel");
document.getElementById("print-form").addEventListener("click", function () { window.print(); });
document.getElementById("appointment-form").addEventListener("submit", function (event) {
  event.preventDefault();
  var checks = [["full-name", "Enter your full name"], ["dob", "Enter your date of birth"],
                ["email", "Enter your email address"], ["clinic", "Choose a clinic"], ["privacy", "Agree to the privacy notice"]];
  var list = document.getElementById("error-list");
  list.innerHTML = "";
  checks.forEach(function (c) {
    var field = document.getElementById(c[0]);
    var error = document.getElementById(c[0] + "-error");
    var empty = field.type === "checkbox" ? !field.checked : !field.value;
    field.setAttribute("aria-invalid", String(empty));
    error.hidden = !empty;
    error.textContent = empty ? "Error: " + c[1] + "." : "";
    if (empty) {
      var item = document.createElement("li");
      var link = document.createElement("a");
      link.href = "#" + c[0];
      link.textContent = c[1];
      item.appendChild(link);
      list.appendChild(item);
    }
  });
  var summary = document.getElementById("error-summary");
  summary.hidden = list.children.length === 0;
  if (!summary.hidden) { summary.focus(); }
});""",
)

PAGES["B07"] = dict(
    title="Clinic locations and opening hours - CityCare", current="Locations",
    body=f"""<h1>Clinic locations</h1>
<p id="locations-intro">CityCare has five clinics. All of them have step-free entrances, accessible toilets and hearing loops at reception. Parking is available at Northside and Harbour.</p>
<form id="postcode-form" action="locations.html">
<div class="field"><label for="postcode-input">Find your nearest clinic: enter your postcode</label>
<div style="display:flex;gap:0.5rem;"><input type="text" id="postcode-input" name="postcode" autocomplete="postal-code" style="max-width:12rem;">
<button type="submit" class="icon-btn" id="postcode-search" aria-label="Find nearest clinic">{ICONS["search"]}</button></div></div>
</form>
<h2 id="map-heading">Map</h2>
<img id="clinics-map" src="images/cc-map.png" width="640" height="360" alt="Map of the city showing the five CityCare clinics: Northside in the north, Hilltop in the west, Old Town in the centre, Riverside in the east and Harbour in the south">
<h2 id="hours-heading">Opening hours</h2>
<table id="hours-table">
<caption>Opening hours by clinic</caption>
<thead><tr><th scope="col">Clinic</th><th scope="col">Monday to Friday</th><th scope="col">Saturday</th><th scope="col">Sunday</th></tr></thead>
<tbody>
<tr><th scope="row">Northside</th><td>8:00 to 18:00</td><td>9:00 to 13:00</td><td>Closed</td></tr>
<tr><th scope="row">Riverside</th><td>8:00 to 20:00</td><td>9:00 to 13:00</td><td>Closed</td></tr>
<tr><th scope="row">Old Town</th><td>8:00 to 18:00</td><td>9:00 to 13:00</td><td>10:00 to 14:00</td></tr>
<tr><th scope="row">Harbour</th><td>8:00 to 18:00</td><td>Closed</td><td>Closed</td></tr>
<tr><th scope="row">Hilltop</th><td>9:00 to 17:00</td><td>Closed</td><td>Closed</td></tr>
</tbody>
</table>
<h2 id="addresses-heading">Addresses</h2>
<ul id="address-list">
<li><strong>Northside:</strong> 14 North Road. <a href="https://maps.example/northside">Get directions to Northside clinic</a></li>
<li><strong>Riverside:</strong> 2 Quay Street. <a href="https://maps.example/riverside">Get directions to Riverside clinic</a></li>
<li><strong>Old Town:</strong> 1 Market Street. <a href="https://maps.example/oldtown">Get directions to Old Town clinic</a></li>
<li><strong>Harbour:</strong> 40 Dock Lane. <a href="https://maps.example/harbour">Get directions to Harbour clinic</a></li>
<li><strong>Hilltop:</strong> 7 Summit Avenue. <a href="https://maps.example/hilltop">Get directions to Hilltop clinic</a></li>
</ul>
<p><button type="button" class="btn btn-secondary" id="access-toggle" aria-expanded="false" aria-controls="access-panel">Show accessibility details</button>
<button type="button" class="icon-btn" id="share-locations" aria-label="Share this page">{ICONS["share"]}</button></p>
<div id="access-panel" class="panel" hidden>
<p id="access-text">Each clinic has a lowered reception counter, an accessible toilet, a quiet waiting area and large-print signs. Assistance dogs are welcome.</p>
</div>""",
    script='wireToggle("access-toggle", "access-panel");',
)

PAGES["B08"] = dict(
    title="Sign in to your account - CityCare", current="",
    body=f"""<h1>Sign in</h1>
<p id="signin-intro">Sign in to see your appointments, test results and prescriptions. Your account is protected and only you can see your health information.</p>
<form id="signin-form" action="dashboard.html">
<div class="field"><label for="username">Username or email address</label>
<input type="text" id="username" name="username" autocomplete="username"></div>
<div class="field"><label for="password">Password</label>
<div style="display:flex;gap:0.5rem;"><input type="password" id="password" name="password" autocomplete="current-password" style="max-width:20rem;">
<button type="button" class="icon-btn" id="toggle-password" aria-label="Show password" aria-pressed="false">{ICONS["eye"]}</button></div></div>
<div class="choice"><input type="checkbox" id="remember" name="remember"><label for="remember">Remember me on this device</label></div>
<p><a id="forgot-link" href="reset.html">Forgot your password?</a></p>
<p><button type="submit" class="btn" id="signin-submit">Sign in</button></p>
</form>
<p><button type="button" class="btn btn-secondary" id="trouble-toggle" aria-expanded="false" aria-controls="trouble-panel">Having trouble signing in?</button></p>
<div id="trouble-panel" class="panel" hidden>
<p id="trouble-text">If you cannot sign in, phone the help line on 01234 567 890. It is open Monday to Friday from 8:00 to 18:00.</p>
</div>
<h2 id="new-heading">New to CityCare?</h2>
<p id="new-text">You can create an account with the letter we sent you after your first appointment. It has a ten-digit patient ID number.</p>
<p><a class="btn btn-secondary" href="register.html">Create an account</a></p>
<h2 id="app-heading">Get the app</h2>
<img id="app-screenshot" src="images/cc-app.png" width="640" height="360" alt="The CityCare app on a phone, showing a list of upcoming appointments">""",
    script="""wireToggle("trouble-toggle", "trouble-panel");
document.getElementById("toggle-password").addEventListener("click", function () {
  var field = document.getElementById("password");
  var show = field.type === "password";
  field.type = show ? "text" : "password";
  this.setAttribute("aria-pressed", String(show));
});""",
)

PAGES["B09"] = dict(
    title="Search results for flu - CityCare", current="Services",
    body=f"""<h1>Search results for "flu"</h1>
<form id="results-search" action="search.html">
<div class="field"><label for="results-query">Search CityCare</label>
<div style="display:flex;gap:0.5rem;"><input type="text" id="results-query" name="q" value="flu" style="max-width:20rem;">
<button type="button" class="icon-btn" id="clear-search" aria-label="Clear search">{ICONS["clear"]}</button>
<button type="submit" class="btn" id="results-submit">Search</button></div></div>
</form>
<p id="results-summary">Showing 10 of 42 results. Use the filters to narrow your search to one type of page.</p>
<p><button type="button" class="btn btn-secondary" id="filters-toggle" aria-expanded="true" aria-controls="filters-panel">Hide filters</button></p>
<div id="filters-panel" class="panel">
<h2 id="filters-heading">Filter results</h2>
<fieldset>
<legend>Type of page</legend>
<div class="choice"><input type="checkbox" id="filter-services" name="type" value="services"><label for="filter-services">Services</label></div>
<div class="choice"><input type="checkbox" id="filter-news" name="type" value="news"><label for="filter-news">News</label></div>
<div class="choice"><input type="checkbox" id="filter-advice" name="type" value="advice"><label for="filter-advice">Health advice</label></div>
</fieldset>
<div class="field"><label for="sort">Sort by</label>
<select id="sort" name="sort"><option>Most relevant</option><option>Newest first</option><option>Oldest first</option></select></div>
</div>
<h2 id="results-heading">Results</h2>
<ol id="results-list">
<li><h3><a href="r1.html">Flu vaccinations now available</a></h3><p>Book your free flu vaccination at any CityCare clinic.</p></li>
<li><h3><a href="r2.html">Who can get a free flu vaccination</a></h3><p>Adults aged 65 or over, pregnant people and carers are eligible.</p></li>
<li><h3><a href="r3.html">Flu symptoms in adults</a></h3><p>How to tell flu from a cold, and when to get help.</p></li>
<li><h3><a href="r4.html">Flu and children</a></h3><p>Nasal spray vaccinations for children aged 2 to 11.</p></li>
<li><h3><a href="r5.html">Treating flu at home</a></h3><p>Rest, fluids and advice from our pharmacists.</p></li>
<li><h3><a href="r6.html">Flu clinics at Riverside</a></h3><p>Evening and weekend flu clinics this autumn.</p></li>
<li><h3><a href="r7.html">Flu vaccination side effects</a></h3><p>What to expect after your vaccination.</p></li>
<li><h3><a href="r8.html">Flu in pregnancy</a></h3><p>Why the vaccination is recommended in every trimester.</p></li>
<li><h3><a href="r9.html">Bird flu: what you need to know</a></h3><p>Current advice for people who keep poultry.</p></li>
<li><h3><a href="r10.html">Flu season report</a></h3><p>How many people we vaccinated last winter.</p></li>
</ol>
<nav aria-label="Pagination">
<ul style="list-style:none;display:flex;gap:1rem;padding:0;">
<li><a href="?page=1" aria-current="page">1</a></li>
<li><a href="?page=2">2</a></li>
<li><a href="?page=3">3</a></li>
<li><a href="?page=2">Next page</a></li>
</ul>
</nav>
<aside aria-labelledby="campaign-heading" class="panel">
<h2 id="campaign-heading">This winter</h2>
<img id="campaign-photo" src="images/cc-flu-campaign.png" width="640" height="360" alt="A nurse giving a flu vaccination to an older man in a clinic room">
<p id="campaign-text">Our flu campaign runs until the end of February. Walk-in sessions are held every Saturday morning.</p>
</aside>""",
    script="""document.getElementById("filters-toggle").addEventListener("click", function () {
  var panel = document.getElementById("filters-panel");
  panel.hidden = !panel.hidden;
  if (this.tagName === "BUTTON") { this.setAttribute("aria-expanded", String(!panel.hidden)); }
  this.textContent = panel.hidden ? "Show filters" : "Hide filters";
});
document.getElementById("clear-search").addEventListener("click", function () {
  var field = document.getElementById("results-query");
  field.value = "";
  field.focus();
});""",
)

PAGES["B10"] = dict(
    title="Your dashboard - CityCare", current="Appointments",
    body=f"""<h1>Welcome back, Sam</h1>
<p id="dashboard-intro">Here is a summary of your care. Your next appointment is a check-up at Riverside clinic on Tuesday 12 March at 10:30.</p>
<ul class="cards" id="summary-cards">
<li class="card"><h2 style="margin-top:0;font-size:1.2rem;" id="appointments-card-heading">Appointments</h2><p>2 upcoming</p></li>
<li class="card"><h2 style="margin-top:0;font-size:1.2rem;">Prescriptions</h2><p>1 ready to collect</p></li>
<li class="card"><h2 style="margin-top:0;font-size:1.2rem;">Messages</h2><p>No new messages</p></li>
</ul>
<h2 id="upcoming-heading">Upcoming appointments</h2>
<table id="appointments-table">
<caption>Your upcoming appointments</caption>
<thead><tr><th scope="col">Date</th><th scope="col">Time</th><th scope="col">Clinic</th><th scope="col">Type</th><th scope="col">Actions</th></tr></thead>
<tbody>
<tr><td>Tuesday 12 March</td><td>10:30</td><td>Riverside</td><td>Check-up</td>
<td><button type="button" class="icon-btn" id="edit-appt-1" aria-label="Change appointment on 12 March">{ICONS["edit"]}</button>
<button type="button" class="icon-btn" id="delete-appt-1" aria-label="Cancel appointment on 12 March">{ICONS["trash"]}</button></td></tr>
<tr><td>Friday 5 April</td><td>14:00</td><td>Old Town</td><td>Vaccination</td>
<td><button type="button" class="icon-btn" id="edit-appt-2" aria-label="Change appointment on 5 April">{ICONS["edit"]}</button>
<button type="button" class="icon-btn" id="delete-appt-2" aria-label="Cancel appointment on 5 April">{ICONS["trash"]}</button></td></tr>
</tbody>
</table>
<p><button type="button" class="btn btn-secondary" id="past-toggle" aria-expanded="false" aria-controls="past-panel">Show past appointments</button></p>
<div id="past-panel" class="panel" hidden>
<p id="past-text">You had a blood test at Old Town clinic on 3 January and a vaccination at Harbour clinic on 14 November.</p>
</div>
<h2 id="notifications-heading">Notifications</h2>
<p class="choice"><button type="button" role="switch" aria-checked="true" class="btn btn-secondary" id="sms-switch">Text message reminders</button></p>
<p class="choice"><button type="button" role="switch" aria-checked="false" class="btn btn-secondary" id="email-switch">Email reminders</button></p>
<p id="settings-status" role="status"></p>
<h2 id="note-heading">Message your care team</h2>
<form id="note-form" action="dashboard.html">
<div class="field"><label for="note-input">Short message for your care team</label>
<input type="text" id="note-input" name="note"></div>
<button type="submit" class="btn" id="note-send">Send message</button>
</form>
<h2 id="team-heading">Your care team</h2>
<img id="team-photo" src="images/cc-team.png" width="640" height="360" alt="Dr Amira Khan and nurse Tom Reyes, your care team at Riverside clinic">""",
    script="""wireToggle("past-toggle", "past-panel");
["sms-switch", "email-switch"].forEach(function (id) {
  document.getElementById(id).addEventListener("click", function () {
    var on = this.getAttribute("aria-checked") !== "true";
    this.setAttribute("aria-checked", String(on));
    document.getElementById("settings-status").textContent = "Settings saved";
  });
});""",
)

PAGES["B11"] = dict(
    title="How to prepare for a blood test - CityCare", current="Services",
    header_extra="""<form class="site-search" role="search" action="search.html">
<label for="article-search" style="color:#fff;">Search health information</label>
<input type="text" id="article-search" name="q" style="max-width:14rem;">
</form>""",
    body=f"""<nav aria-label="Breadcrumb">
<ol style="list-style:none;display:flex;gap:0.5rem;padding:0;flex-wrap:wrap;">
<li><a href="index.html">Home</a> &gt;</li>
<li><a href="health.html">Health information</a> &gt;</li>
<li><a href="tests.html" aria-current="page">Tests</a></li>
</ol>
</nav>
<h1>How to prepare for a blood test</h1>
<p id="article-lead">Most blood tests need no preparation, but some ask you to stop eating for a few hours first. This page explains what to do before, during and after your test.</p>
<p><button type="button" class="icon-btn" id="print-page" aria-label="Print this page">{ICONS["print"]}</button>
<button type="button" class="icon-btn" id="share-article" aria-label="Share this page">{ICONS["share"]}</button></p>
<h2 id="before-heading">Before your test</h2>
<p id="before-text">Check your appointment letter. If it says "fasting", do not eat or drink anything except water for 8 to 12 hours before the test. Keep taking your usual medicines unless your doctor tells you otherwise.</p>
<h3 id="water-heading">Drinking water</h3>
<p>Drinking water before the test makes it easier for the nurse to find a vein.</p>
<h2 id="day-heading">On the day</h2>
<figure>
<img id="blood-test-photo" src="images/cc-blood-test.png" width="640" height="360" alt="A nurse placing a small plaster on a patient's arm after taking a blood sample">
<figcaption>A blood test usually takes less than five minutes.</figcaption>
</figure>
<p id="day-text">Wear a top with sleeves that roll up easily. Tell the nurse if you feel faint when giving blood, so you can lie down during the test.</p>
<h2 id="after-heading">After your test</h2>
<p id="after-text">You can go back to your normal activities straight away. Results usually take three to five working days and appear in your CityCare account.</p>
<p><button type="button" class="btn btn-secondary" id="references-toggle" aria-expanded="false" aria-controls="references-panel">Show sources</button></p>
<div id="references-panel" class="panel" hidden>
<p id="references-text">This page was written by CityCare nurses and reviewed by Dr Amira Khan in September.</p>
</div>
<section aria-labelledby="feedback-heading" class="panel">
<h2 id="feedback-heading" style="margin-top:0;">Was this page helpful?</h2>
<p><button type="button" class="btn btn-secondary" id="helpful-yes">Yes</button>
<button type="button" class="btn btn-secondary" id="helpful-no">No</button></p>
<p id="feedback-status" role="status"></p>
</section>""",
    script="""wireToggle("references-toggle", "references-panel");
document.getElementById("print-page").addEventListener("click", function () { window.print(); });
["helpful-yes", "helpful-no"].forEach(function (id) {
  document.getElementById(id).addEventListener("click", function () {
    document.getElementById("feedback-status").textContent = "Thank you for your feedback.";
  });
});""",
)

PAGES["B12"] = dict(
    title="Register for the community health fair - CityCare", current="",
    body=f"""<h1>Community health fair: register</h1>
<p id="event-intro">Join us at Old Town clinic for a free health fair with blood pressure checks, healthy cooking demonstrations and advice from our pharmacists. Places are limited, so please register.</p>
<img id="event-poster" src="images/cc-event.png" width="640" height="360" alt="Families visiting stalls at last year's health fair in the Old Town clinic garden">
<p><button type="button" class="icon-btn" id="share-event" aria-label="Share this event">{ICONS["share"]}</button>
<button type="button" class="icon-btn" id="calendar-event" aria-label="Add to calendar">{ICONS["calendar"]}</button></p>
<form id="event-form">
<h2 id="attendee-heading">About you</h2>
<div class="field"><label for="attendee-name">Your name</label>
<input type="text" id="attendee-name" name="name" autocomplete="name"></div>
<div class="field"><label for="attendee-email">Email address</label>
<input type="email" id="attendee-email" name="email" autocomplete="email"></div>
<h2 id="session-heading">Choose a session</h2>
<div class="field"><label for="event-date">Date</label>
<input type="date" id="event-date" name="date" min="2027-05-01" max="2027-05-02"></div>
<div class="field"><label for="event-time">Arrival time</label>
<input type="time" id="event-time" name="time"></div>
<fieldset>
<legend>Group size</legend>
<div class="choice"><input type="radio" id="group-1" name="group" value="1" checked><label for="group-1">Just me</label></div>
<div class="choice"><input type="radio" id="group-2" name="group" value="2"><label for="group-2">Two people</label></div>
<div class="choice"><input type="radio" id="group-3" name="group" value="3"><label for="group-3">Three or more</label></div>
</fieldset>
<p><button type="button" class="btn btn-secondary" id="access-options-toggle" aria-expanded="false" aria-controls="access-options">Show accessibility options</button></p>
<div id="access-options" class="panel" hidden>
<div class="choice"><input type="checkbox" id="need-bsl" name="needs" value="bsl"><label for="need-bsl">British Sign Language interpreter</label></div>
<div class="choice"><input type="checkbox" id="need-quiet" name="needs" value="quiet"><label for="need-quiet">Quiet room</label></div>
<div class="choice"><input type="checkbox" id="need-chair" name="needs" value="chair"><label for="need-chair">Wheelchair space</label></div>
</div>
<p><button type="submit" class="btn" id="event-submit">Register</button></p>
</form>
<div id="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-heading" class="panel" hidden style="position:fixed;top:20%;left:50%;transform:translateX(-50%);max-width:28rem;box-shadow:0 0 0 100vmax rgba(0,0,0,0.45);">
<h2 id="confirm-heading" style="margin-top:0;">You are registered</h2>
<p id="confirm-text">We have sent a confirmation email with your ticket.</p>
<button type="button" class="btn" id="confirm-close">Close</button>
</div>""",
    script="""wireToggle("access-options-toggle", "access-options");
var lastFocus = null;
document.getElementById("event-form").addEventListener("submit", function (event) {
  event.preventDefault();
  lastFocus = document.activeElement;
  document.getElementById("confirm-dialog").hidden = false;
  document.getElementById("confirm-close").focus();
});
function closeDialog() {
  document.getElementById("confirm-dialog").hidden = true;
  if (lastFocus) { lastFocus.focus(); }
}
document.getElementById("confirm-close").addEventListener("click", closeDialog);
document.getElementById("confirm-dialog").addEventListener("keydown", function (event) {
  if (event.key === "Escape") { closeDialog(); }
  if (event.key === "Tab") { event.preventDefault(); document.getElementById("confirm-close").focus(); }
});""",
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for pid, p in PAGES.items():
        html = shell(p["title"], p["current"], p["body"], p.get("script", ""),
                     p.get("header_extra", ""), p.get("footer_extra", ""))
        (OUT / f"{pid}.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
