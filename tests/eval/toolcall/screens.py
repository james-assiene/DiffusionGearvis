"""Synthetic screens of the tool-call eval set v1 (T-0080b criterion 1; human decision Q4: local HTML, offline).

Each screen is one self-contained HTML page (inline CSS, no external resource of any kind: no font, image,
script or stylesheet URL) rendered at 1280 x 800 CSS px, device scale 1. Elements a case can target carry
``data-t="<name>"``; ``build.py`` reads their bounding boxes from the DOM after layout and normalises them to
0-999. A screen may set ``scroll_y`` (page scroll), ``hscroll`` (``{data-t: scrollLeft}``) or ``vscroll``
(``{data-t: scrollTop}``) so that a scroll
case has content on both sides.

The split of the *screens* is fixed here: a screen belongs to exactly one split, so no dev case and test case
share a frame.
"""

from __future__ import annotations

import dataclasses
import html

W, H = 1280, 800


@dataclasses.dataclass(frozen=True)
class Screen:
    id: str
    split: str
    title: str
    body: str
    scroll_y: int = 0
    hscroll: tuple[tuple[str, int], ...] = ()
    vscroll: tuple[tuple[str, int], ...] = ()


CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif; font-size: 15px; color: #1f2328;
       background: #f6f7f9; width: 1280px; }
.top { height: 56px; background: #24292f; color: #fff; display: flex; align-items: center; padding: 0 20px; gap: 18px; }
.top .brand { font-weight: bold; font-size: 19px; }
.top a, .top span { color: #d0d7de; text-decoration: none; }
.side { position: absolute; left: 0; top: 56px; width: 220px; bottom: 0; background: #fff; border-right: 1px solid #d0d7de; padding: 14px; }
.side div { padding: 9px 10px; border-radius: 6px; margin-bottom: 4px; }
.side .on { background: #ddf4ff; font-weight: bold; }
.main { position: absolute; left: 240px; top: 76px; right: 20px; }
.card { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; }
.row { display: flex; gap: 12px; align-items: center; }
.btn { display: inline-block; padding: 8px 16px; border-radius: 6px; border: 1px solid #1f883d; background: #1f883d; color: #fff; font-weight: bold; }
.btn.sec { background: #f6f8fa; color: #24292f; border-color: #d0d7de; }
.btn.red { background: #cf222e; border-color: #cf222e; }
.btn.blue { background: #0969da; border-color: #0969da; }
input.f, .f { display: block; width: 100%; height: 38px; border: 1px solid #8c959f; border-radius: 6px; padding: 0 10px; font-size: 15px; background: #fff; font-family: inherit; }
label { display: block; font-weight: bold; margin: 12px 0 6px; }
.muted { color: #57606a; }
.list > div { padding: 12px 14px; border-bottom: 1px solid #eaeef2; background: #fff; }
.grid { display: grid; gap: 14px; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 10px; background: #eaeef2; font-size: 13px; }
.toggle { display: inline-block; width: 44px; height: 24px; border-radius: 12px; background: #8c959f; position: relative; vertical-align: middle; }
.toggle.on { background: #1f883d; }
.toggle::after { content: ""; position: absolute; top: 3px; left: 3px; width: 18px; height: 18px; border-radius: 9px; background: #fff; }
.toggle.on::after { left: 23px; }
.box { display: inline-block; width: 18px; height: 18px; border: 2px solid #57606a; border-radius: 3px; vertical-align: middle; background: #fff; }
.box.on { background: #0969da; border-color: #0969da; }
.strip { display: flex; gap: 14px; overflow-x: auto; overflow-y: hidden; white-space: nowrap; }
.strip::-webkit-scrollbar { display: none; }
.thumb { flex: 0 0 auto; width: 220px; height: 150px; border-radius: 8px; color: #fff; font-weight: bold; display: flex; align-items: flex-end; padding: 10px; }
.sel { background: #b6e3ff; }
table { border-collapse: collapse; background: #fff; }
td, th { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }
th { background: #f6f8fa; }
p { line-height: 1.6; margin: 0 0 14px; }
"""


def page(screen: Screen) -> str:
    """The complete HTML document of ``screen`` (no external resource)."""
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(screen.title)}</title><style>{CSS}</style></head>\n"
        f"<body>\n{screen.body}\n</body></html>\n"
    )


def _top(brand: str, *items: str) -> str:
    links = "".join(f"<span>{html.escape(i)}</span>" for i in items)
    return f'<div class="top"><span class="brand">{html.escape(brand)}</span>{links}</div>'


def _lorem(n: int, topic: str) -> str:
    sentences = [
        f"The {topic} has changed a great deal over the last decade, and the people who know it best say the change is not over.",
        "Local councils have published new figures every spring, and each year the numbers have surprised at least one group of experts.",
        "Residents interviewed for this piece described the shift in practical terms: shorter queues, longer evenings and fewer empty shops.",
        "Critics point out that the improvements are uneven, and that the outer districts have seen far less of the investment.",
        "A spokesperson said the next phase would focus on transport links, housing near the river and a new public library.",
        "Historians note that a similar wave of renewal in the 1960s ended abruptly when funding was withdrawn.",
    ]
    return "".join(f"<p>{' '.join(sentences[(i + j) % 6] for j in range(3))}</p>" for i in range(n))


# ----------------------------------------------------------------------------------------------- dev screens

D_LOGIN = Screen(
    "d01_login",
    "dev",
    "Sign in - Northwind Mail",
    _top("Northwind", "Help", "Status")
    + """
<div style="position:absolute;left:440px;top:120px;width:400px" class="card">
  <h2 style="margin:0 0 6px">Sign in to Northwind</h2>
  <div class="muted">Use your work email address.</div>
  <label for="em">Email address</label>
  <input class="f" id="em" data-t="email" placeholder="name@company.com">
  <label for="pw">Password</label>
  <input class="f" id="pw" type="password" data-t="password" placeholder="Password">
  <div class="row" style="margin:14px 0"><span class="box" data-t="remember"></span><span>Remember me</span>
    <span style="flex:1"></span><a href="#" data-t="forgot" style="color:#0969da">Forgot password?</a></div>
  <div><span class="btn blue" data-t="signin" style="width:100%;text-align:center">Sign in</span></div>
  <div style="margin-top:16px;text-align:center" class="muted">New here? <a href="#" data-t="create" style="color:#0969da">Create an account</a></div>
</div>""",
)

D_STORE = Screen(
    "d02_store",
    "dev",
    "Gadget Corner - Shop",
    _top("Gadget Corner", "Deals", "Orders")
    + """
<div style="position:absolute;left:380px;top:10px;width:520px"><input class="f" data-t="search" placeholder="Search products"></div>
<div style="position:absolute;right:24px;top:10px"><span class="btn sec" data-t="cart">Cart (2)</span></div>
<div class="side">
  <div class="on" data-t="cat_laptops">Laptops</div><div data-t="cat_phones">Phones</div>
  <div data-t="cat_audio">Audio</div><div data-t="cat_accessories">Accessories</div>
</div>
<div class="main grid" style="grid-template-columns: repeat(3, 1fr)">
  <div class="card"><div style="height:120px;background:#8250df;border-radius:6px"></div><h3>Aero 14 laptop</h3><div>&pound;899</div><span class="btn" data-t="add_aero">Add to cart</span></div>
  <div class="card"><div style="height:120px;background:#bf3989;border-radius:6px"></div><h3>Pulse X phone</h3><div>&pound;649</div><span class="btn" data-t="add_pulse">Add to cart</span></div>
  <div class="card"><div style="height:120px;background:#0969da;border-radius:6px"></div><h3>Quiet One headphones</h3><div>&pound;199</div><span class="btn" data-t="add_quiet">Add to cart</span></div>
  <div class="card"><div style="height:120px;background:#1a7f37;border-radius:6px"></div><h3>Dock Pro USB-C hub</h3><div>&pound;79</div><span class="btn" data-t="add_dock">Add to cart</span></div>
  <div class="card"><div style="height:120px;background:#9a6700;border-radius:6px"></div><h3>Tilt monitor arm</h3><div>&pound;59</div><span class="btn" data-t="add_tilt">Add to cart</span></div>
  <div class="card"><div style="height:120px;background:#57606a;border-radius:6px"></div><h3>Glide wireless mouse</h3><div>&pound;35</div><span class="btn" data-t="add_glide">Add to cart</span></div>
</div>""",
)

D_MAIL = Screen(
    "d03_mail",
    "dev",
    "Inbox - Northwind Mail",
    _top("Northwind Mail", "anna.berg@northwind.example")
    + """
<div style="position:absolute;left:480px;top:10px;width:560px"><input class="f" data-t="search" placeholder="Search mail"></div>
<div class="side">
  <div><span class="btn blue" data-t="compose">Compose</span></div>
  <div class="on" data-t="inbox">Inbox (5)</div><div data-t="sent">Sent</div><div data-t="drafts">Drafts</div><div data-t="spam">Spam</div>
</div>
<div class="main card list" data-t="messages" style="padding:0;height:700px;overflow-y:auto">
  <div data-t="msg_lena"><b>Lena Fischer</b> &mdash; Quarterly budget review <span class="muted">09:12</span></div>
  <div data-t="msg_omar"><b>Omar Haddad</b> &mdash; Lunch on Thursday? <span class="muted">08:47</span></div>
  <div data-t="msg_it"><b>IT Service Desk</b> &mdash; Your laptop is ready for collection <span class="muted">Yesterday</span></div>
  <div data-t="msg_priya"><b>Priya Nair</b> &mdash; Slides for the client pitch <span class="muted">Yesterday</span></div>
  <div data-t="msg_hr"><b>HR Team</b> &mdash; Holiday calendar 2027 <span class="muted">Mon</span></div>
  <div><b>Jonas Weber</b> &mdash; Re: invoice 4471 <span class="muted">Mon</span></div>
  <div><b>Facilities</b> &mdash; Fire alarm test on Friday <span class="muted">Sun</span></div>
  <div><b>Sara Lind</b> &mdash; Photos from the offsite <span class="muted">Sat</span></div>
  <div><b>Newsletter</b> &mdash; This week in product <span class="muted">Fri</span></div>
  <div><b>Lena Fischer</b> &mdash; Agenda for Monday <span class="muted">Fri</span></div>
  <div><b>Travel Desk</b> &mdash; Your booking is confirmed <span class="muted">Thu</span></div>
  <div><b>Omar Haddad</b> &mdash; Notes from the call <span class="muted">Thu</span></div>
  <div><b>Payroll</b> &mdash; Payslip available <span class="muted">Wed</span></div>
  <div><b>Priya Nair</b> &mdash; Draft contract <span class="muted">Wed</span></div>
  <div><b>Security</b> &mdash; New sign-in from Firefox <span class="muted">Tue</span></div>
  <div><b>Jonas Weber</b> &mdash; Invoice 4471 <span class="muted">Tue</span></div>
  <div><b>Calendar</b> &mdash; Reminder: dentist <span class="muted">Mon</span></div>
</div>""",
)

D_SETTINGS = Screen(
    "d04_settings",
    "dev",
    "Settings",
    _top("Settings", "Back to app")
    + """
<div class="side">
  <div class="on" data-t="tab_general">General</div><div data-t="tab_privacy">Privacy</div>
  <div data-t="tab_notifications">Notifications</div><div data-t="tab_account">Account</div>
</div>
<div class="main card" style="width:640px">
  <h2 style="margin-top:0">General</h2>
  <div class="row" style="margin:10px 0"><span style="width:260px">Dark mode</span><span class="toggle" data-t="dark"></span></div>
  <div class="row" style="margin:10px 0"><span style="width:260px">Install updates automatically</span><span class="toggle on" data-t="autoupdate"></span></div>
  <label>Display name</label><input class="f" data-t="display_name" placeholder="How others see you">
  <label>Language</label><div class="f" data-t="language" style="line-height:36px">English (UK) &#9662;</div>
  <div class="row" style="margin-top:22px"><span class="btn" data-t="save">Save changes</span><span class="btn sec" data-t="cancel">Cancel</span></div>
</div>""",
)

D_ARTICLE = Screen(
    "d05_article",
    "dev",
    "The new riverside - City Times",
    '<div class="top" style="position:fixed;left:0;right:0;top:0;z-index:2"><span class="brand">City Times</span><span>News</span><span>Sport</span><span>Culture</span></div>'
    + '<div data-t="article" style="margin:76px auto 0;width:760px;background:#fff;padding:24px 32px;min-height:2400px">'
    + "<h1>The new riverside: what changed and what did not</h1>"
    + _lorem(14, "riverside district")
    + "</div>",
    scroll_y=700,
)

D_CHAT = Screen(
    "d06_chat",
    "dev",
    "Chat - Teamline",
    _top("Teamline", "Online")
    + """
<div class="side">
  <div class="on" data-t="c_omar">Omar Haddad</div><div data-t="c_lena">Lena Fischer</div><div data-t="c_design">#design</div><div data-t="c_general">#general</div>
</div>
<div class="main card" style="height:640px">
  <div class="muted">Omar Haddad</div>
  <div style="margin:10px 0;padding:10px;background:#f6f8fa;border-radius:8px;width:420px">Are we still on for lunch on Thursday?</div>
  <div style="margin:10px 0 10px auto;padding:10px;background:#ddf4ff;border-radius:8px;width:420px">Yes! Where do you want to go?</div>
  <div style="margin:10px 0;padding:10px;background:#f6f8fa;border-radius:8px;width:420px">How about the noodle place on Mill Road?</div>
</div>
<div style="position:absolute;left:240px;right:20px;top:730px" class="row">
  <span class="btn sec" data-t="attach">Attach</span>
  <input class="f" data-t="message" placeholder="Message Omar Haddad" style="flex:1">
  <span class="btn blue" data-t="send">Send</span>
</div>""",
)

D_PHOTOS = Screen(
    "d07_photos",
    "dev",
    "Photos",
    _top("Photos", "Library", "Albums", "Shared")
    + """
<div style="position:absolute;left:24px;top:80px;right:24px">
  <h2>Summer trip &mdash; 24 photos</h2>
  <div class="strip" data-t="strip" style="width:1232px">
"""
    + "".join(
        f'<div class="thumb" style="background:hsl({(i * 37) % 360},55%,45%)">Photo {i + 1}</div>'
        for i in range(24)
    )
    + """
  </div>
  <div style="margin-top:24px;width:420px"><input class="f" data-t="search" placeholder="Search your photos"></div>
  <h2 style="margin-top:24px">Albums</h2>
  <div class="row"><span class="btn sec" data-t="album_family">Family</span><span class="btn sec" data-t="album_work">Work</span><span class="btn sec" data-t="album_pets">Pets</span><span class="btn sec" data-t="new_album">+ New album</span></div>
</div>""",
    hscroll=(("strip", 700),),
)

D_EDITOR = Screen(
    "d08_editor",
    "dev",
    "Draft - Notes",
    _top("Notes", "All notes", "Shared with me")
    + """
<div style="position:absolute;left:120px;top:76px;width:1040px">
  <div class="row card" style="padding:8px"><span class="btn sec" data-t="bold"><b>B</b></span><span class="btn sec" data-t="italic"><i>I</i></span>
    <span class="btn sec" data-t="undo">Undo</span><span class="btn sec" data-t="redo">Redo</span><span style="flex:1"></span><span class="btn" data-t="share">Share</span></div>
  <input class="f" data-t="title" placeholder="Untitled note" style="margin:14px 0;font-size:20px;height:46px">
  <div class="card" data-t="body" style="height:460px">
    <p>Packing list for the weekend:</p>
    <p>Rain jacket, walking boots, <span class="sel">the blue thermos and two maps</span>, charger.</p>
    <p>Remember to book the parking space before Friday.</p>
  </div>
</div>""",
)

# ---------------------------------------------------------------------------------------------- test screens

T_SIGNUP = Screen(
    "t01_signup",
    "test",
    "Create your account - Fieldnote",
    _top("Fieldnote", "Pricing", "Log in")
    + """
<div style="position:absolute;left:380px;top:90px;width:520px" class="card">
  <h2 style="margin:0">Create your account</h2>
  <div class="row"><div style="flex:1"><label>First name</label><input class="f" data-t="first"></div>
    <div style="flex:1"><label>Last name</label><input class="f" data-t="last"></div></div>
  <label>Email</label><input class="f" data-t="email">
  <label>Phone (optional)</label><input class="f" data-t="phone">
  <div class="row" style="margin:16px 0"><span class="box" data-t="terms"></span><span>I agree to the Terms of Service</span></div>
  <span class="btn" data-t="create" style="width:100%;text-align:center">Create account</span>
  <div style="margin-top:12px;text-align:center"><a href="#" data-t="login" style="color:#0969da">Already have an account? Log in</a></div>
</div>""",
)

T_RESULTS = Screen(
    "t02_results",
    "test",
    "heat pump grants - Search",
    """<div style="background:#fff;padding:18px 24px;border-bottom:1px solid #d0d7de" class="row">
  <span style="font-size:22px;font-weight:bold;color:#0969da">Findr</span>
  <input class="f" data-t="search" value="heat pump grants" style="width:600px">
  <span class="btn blue" data-t="go">Search</span></div>
<div style="padding:0 24px 0 160px;background:#fff" class="row">
  <span style="padding:10px" data-t="f_all"><b>All</b></span><span style="padding:10px" data-t="f_images">Images</span>
  <span style="padding:10px" data-t="f_news">News</span><span style="padding:10px" data-t="f_videos">Videos</span></div>
<div data-t="results" style="margin-left:160px;width:720px">
  <div style="margin:18px 0"><a href="#" data-t="r1" style="font-size:19px;color:#1a0dab">Boiler Upgrade Scheme: apply for a grant</a><div class="muted">gov.example &mdash; You could get up to &pound;7,500 towards a heat pump.</div></div>
  <div style="margin:18px 0"><a href="#" data-t="r2" style="font-size:19px;color:#1a0dab">Heat pump grants explained (2026 guide)</a><div class="muted">energysaving.example &mdash; Who qualifies, how long it takes, what installers charge.</div></div>
  <div style="margin:18px 0"><a href="#" data-t="r3" style="font-size:19px;color:#1a0dab">Are heat pumps worth it? Real owners answer</a><div class="muted">forum.example &mdash; 312 replies.</div></div>
  <div style="margin:18px 0"><a href="#" data-t="r4" style="font-size:19px;color:#1a0dab">Find an accredited installer near you</a><div class="muted">mcs.example &mdash; Search by postcode.</div></div>
  <div style="margin:18px 0"><a href="#" data-t="r5" style="font-size:19px;color:#1a0dab">Air source vs ground source heat pumps</a><div class="muted">which.example &mdash; Costs and savings compared.</div></div>
  <div style="margin:18px 0"><a href="#" style="font-size:19px;color:#1a0dab">Heat pump running costs calculator</a><div class="muted">costs.example &mdash; Enter your floor area and tariff.</div></div>
  <div style="margin:18px 0"><a href="#" style="font-size:19px;color:#1a0dab">Grant scams: what to watch out for</a><div class="muted">consumer.example &mdash; Five warning signs.</div></div>
  <div style="margin:18px 0"><a href="#" style="font-size:19px;color:#1a0dab">Do I need planning permission for a heat pump?</a><div class="muted">planning.example &mdash; Permitted development rules.</div></div>
  <div style="margin:18px 0"><a href="#" style="font-size:19px;color:#1a0dab">Heat pump noise: how loud is too loud?</a><div class="muted">acoustics.example &mdash; Decibel limits explained.</div></div>
  <div style="margin:18px 0"><a href="#" style="font-size:19px;color:#1a0dab">Insulation first? The fabric-first debate</a><div class="muted">building.example &mdash; Why order matters.</div></div>
  <div style="margin:18px 0"><a href="#" style="font-size:19px;color:#1a0dab">How long do heat pumps last?</a><div class="muted">energy.example &mdash; Lifespan and maintenance.</div></div>
  <div class="row" style="margin:24px 0"><span class="btn sec" data-t="p2">2</span><span class="btn sec" data-t="p3">3</span><span class="btn sec" data-t="next">Next &rsaquo;</span></div>
</div>""",
)

T_FLIGHTS = Screen(
    "t03_flights",
    "test",
    "Flights - Skyhop",
    _top("Skyhop", "Flights", "Hotels", "Car hire")
    + """
<div style="position:absolute;left:140px;top:120px;width:1000px" class="card">
  <div class="row"><span class="btn sec" data-t="return">Return</span><span class="btn sec" data-t="oneway">One way</span><span class="btn sec" data-t="multicity">Multi-city</span></div>
  <div class="row" style="margin-top:6px">
    <div style="flex:1"><label>From</label><input class="f" data-t="from" placeholder="City or airport"></div>
    <div style="flex:1"><label>To</label><input class="f" data-t="to" placeholder="City or airport"></div>
    <div style="width:180px"><label>Depart</label><input class="f" data-t="depart" placeholder="dd/mm/yyyy"></div>
    <div style="width:120px"><label>Passengers</label><input class="f" data-t="pax" value="1 adult"></div>
  </div>
  <div style="margin-top:22px;text-align:right"><span class="btn blue" data-t="search">Search flights</span></div>
</div>""",
)

T_CALENDAR = Screen(
    "t04_calendar",
    "test",
    "Calendar - March",
    _top("Calendar", "Day", "Week", "Month")
    + """
<div style="position:absolute;left:20px;top:70px" class="row"><span class="btn" data-t="new_event">New event</span><span class="btn sec" data-t="today">Today</span>
  <span class="btn sec" data-t="prev">&lsaquo;</span><span class="btn sec" data-t="next">&rsaquo;</span><b style="font-size:20px">March 2027</b></div>
<table style="position:absolute;left:20px;top:120px;width:880px">
<tr><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th><th>Sun</th></tr>
"""
    + "".join(
        "<tr>"
        + "".join(
            f'<td style="height:100px;vertical-align:top" data-t="day{w * 7 + d + 1}">{w * 7 + d + 1}'
            + (
                '<div class="tag sel" data-t="ev_standup">Stand-up</div>'
                if w * 7 + d + 1 == 9
                else ""
            )
            + ('<div class="tag" data-t="ev_dentist">Dentist</div>' if w * 7 + d + 1 == 17 else "")
            + "</td>"
            for d in range(7)
        )
        + "</tr>"
        for w in range(4)
    )
    + """
</table>
<div class="card" style="position:absolute;left:920px;top:120px;width:340px">
  <h3 style="margin-top:0">New event</h3>
  <label>Title</label><input class="f" data-t="ev_title" placeholder="Add title">
  <label>Location</label><input class="f" data-t="ev_location" placeholder="Add location">
  <div class="row" style="margin-top:16px"><span class="btn" data-t="ev_save">Save</span><span class="btn sec" data-t="ev_discard">Discard</span></div>
</div>""",
)

T_FILES = Screen(
    "t05_files",
    "test",
    "Files - Documents",
    _top("Files", "Home / Documents")
    + """
<div style="position:absolute;left:20px;top:70px;right:20px" class="row">
  <span class="btn sec" data-t="new_folder">New folder</span><span class="btn sec" data-t="upload">Upload</span>
  <span class="btn sec" data-t="rename">Rename</span><span class="btn red" data-t="delete">Delete</span><span style="flex:1"></span>
  <input class="f" data-t="search" placeholder="Search files" style="width:300px"></div>
<div class="card list" data-t="files" style="position:absolute;left:20px;top:130px;right:20px;padding:0">
  <div data-t="f_budget">&#128196; budget-2027.xlsx <span class="muted">2.1 MB</span></div>
  <div data-t="f_contract" class="sel">&#128196; contract-draft.docx <span class="muted">88 KB</span></div>
  <div data-t="f_photos">&#128193; Holiday photos <span class="muted">142 items</span></div>
  <div data-t="f_taxes">&#128193; Taxes <span class="muted">12 items</span></div>
  <div data-t="f_cv">&#128196; cv-anna-berg.pdf <span class="muted">240 KB</span></div>
  <div data-t="f_notes">&#128196; meeting-notes.txt <span class="muted">4 KB</span></div>
</div>""",
)

T_MUSIC = Screen(
    "t06_music",
    "test",
    "Music",
    _top("Tunebox", "Home", "Browse", "Library")
    + """
<div style="position:absolute;left:20px;top:76px;width:560px"><input class="f" data-t="search" placeholder="Search songs, artists, podcasts"></div>
<div class="card list" data-t="playlist" style="position:absolute;left:20px;top:130px;width:900px;padding:0">
  <div data-t="s1">1. Northern Lights &mdash; Aurora Hall <span class="muted">3:41</span></div>
  <div data-t="s2">2. Paper Boats &mdash; The Kites <span class="muted">4:02</span></div>
  <div data-t="s3" class="sel">3. Slow Tide &mdash; Mira Sol <span class="muted">5:15</span></div>
  <div data-t="s4">4. Late Train &mdash; Hollow Pines <span class="muted">3:28</span></div>
  <div data-t="s5">5. Glasshouse &mdash; Juniper <span class="muted">4:49</span></div>
</div>
<div style="position:absolute;left:0;right:0;top:700px;height:100px;background:#24292f;color:#fff" class="row">
  <span style="width:300px;padding-left:20px">Slow Tide &mdash; Mira Sol</span>
  <span class="btn sec" data-t="prev">&#9198; Prev</span><span class="btn blue" data-t="pause">&#9208; Pause</span><span class="btn sec" data-t="next">Next &#9197;</span>
  <span style="flex:1"></span><span data-t="shuffle" class="btn sec">Shuffle</span><span data-t="repeat" class="btn sec" style="margin-right:20px">Repeat</span>
</div>""",
)

T_TODO = Screen(
    "t07_todo",
    "test",
    "Todo",
    _top("Todo", "Personal")
    + """
<div style="position:absolute;left:340px;top:90px;width:600px" class="card">
  <div class="row"><input class="f" data-t="new_task" placeholder="What needs to be done?" style="flex:1"><span class="btn" data-t="add">Add</span></div>
  <div class="row" style="margin:12px 0"><span class="tag" data-t="all">All</span><span class="tag" data-t="active">Active</span><span class="tag" data-t="completed">Completed</span></div>
  <div class="list">
    <div class="row"><span class="box" data-t="cb_milk"></span>Buy oat milk</div>
    <div class="row"><span class="box on" data-t="cb_plants"></span><s>Water the plants</s></div>
    <div class="row"><span class="box" data-t="cb_dentist"></span>Call the dentist</div>
    <div class="row"><span class="box" data-t="cb_tax"></span>File the tax return</div>
    <div class="row"><span class="box" data-t="cb_bike"></span>Fix the bike light</div>
  </div>
  <div style="margin-top:12px"><span class="btn sec" data-t="clear">Clear completed</span></div>
</div>""",
)

T_MAPS = Screen(
    "t08_maps",
    "test",
    "Maps",
    """<div style="position:absolute;inset:0;background:repeating-linear-gradient(45deg,#e6efe0 0 40px,#dfe9d8 40px 80px)"></div>
<div style="position:absolute;left:380px;top:0;right:0;bottom:0;background:linear-gradient(90deg,transparent 48%,#fff 48% 52%,transparent 52%),linear-gradient(0deg,transparent 30%,#fff 30% 33%,transparent 33%)"></div>
<div class="card" style="position:absolute;left:16px;top:16px;width:350px;height:768px">
  <div class="row"><input class="f" data-t="search" placeholder="Search places"><span class="btn blue" data-t="go">Go</span></div>
  <div style="margin:12px 0"><span class="btn sec" data-t="directions">Directions</span></div>
  <div class="list">
    <div data-t="p_cafe"><b>Bluebird Caf&eacute;</b><div class="muted">Coffee &middot; 4.6 &middot; 300 m</div></div>
    <div data-t="p_library"><b>Central Library</b><div class="muted">Library &middot; 4.8 &middot; 650 m</div></div>
    <div data-t="p_pharmacy"><b>Rowan Pharmacy</b><div class="muted">Pharmacy &middot; 4.2 &middot; 900 m</div></div>
  </div>
</div>
<div style="position:absolute;right:20px;top:640px"><div class="btn sec" data-t="zoom_in" style="display:block;margin-bottom:6px">+</div><div class="btn sec" data-t="zoom_out" style="display:block">&minus;</div></div>
<div style="position:absolute;left:820px;top:300px;width:22px;height:22px;border-radius:11px;background:#cf222e;border:3px solid #fff" data-t="pin"></div>""",
)

T_SHEET = Screen(
    "t09_sheet",
    "test",
    "Budget 2027 - Sheets",
    _top("Sheets", "File", "Edit", "View", "Data")
    + """
<div style="position:absolute;left:20px;top:66px;right:20px" class="row"><b>fx</b><input class="f" data-t="formula" value="=SUM(B2:B7)"><span class="btn sec" data-t="sort">Sort A&ndash;Z</span><span class="btn sec" data-t="filter">Filter</span></div>
<div data-t="grid" style="position:absolute;left:20px;top:120px;width:1240px;overflow-x:auto" class="strip">
<table><tr><th></th>"""
    + "".join(f"<th style='min-width:120px'>{c}</th>" for c in "ABCDEFGHIJKL")
    + "</tr>"
    + "".join(
        f"<tr><th>{r}</th>"
        + "".join(
            (
                f'<td data-t="cell_{c}{r}"{" class=sel" if (c, r) == ("B", 4) else ""}>'
                + (
                    ["Item", "Rent", "Food", "Travel", "Energy", "Phone", "Total"][r - 1]
                    if c == "A"
                    else (
                        ["Amount", "1200", "420", "180", "95", "30", "=SUM"][r - 1]
                        if c == "B"
                        else ""
                    )
                )
                + "</td>"
            )
            for c in "ABCDEFGHIJKL"
        )
        + "</tr>"
        for r in range(1, 8)
    )
    + """</table></div>
<div style="position:absolute;left:20px;top:740px" class="row"><span class="btn sec" data-t="sheet1">Sheet1</span><span class="btn sec" data-t="sheet2">Sheet2</span><span class="btn sec" data-t="add_sheet">+</span></div>""",
)

T_VIDEO = Screen(
    "t10_video",
    "test",
    "How to sharpen a chisel - Vidstream",
    _top("Vidstream", "Home", "Subscriptions")
    + """
<div style="position:absolute;left:24px;top:76px;width:820px">
  <div style="height:440px;background:#000;border-radius:10px" data-t="player"></div>
  <h2>How to sharpen a chisel in five minutes</h2>
  <div class="row"><b>Workshop Wren</b><span class="muted">214K subscribers</span><span class="btn red" data-t="subscribe">Subscribe</span>
    <span style="flex:1"></span><span class="btn sec" data-t="like">&#128077; 12K</span><span class="btn sec" data-t="share">Share</span></div>
  <input class="f" data-t="comment" placeholder="Add a comment..." style="margin-top:16px">
</div>
<div class="list" data-t="related" style="position:absolute;left:870px;top:76px;width:390px;height:724px;overflow-y:auto">
"""
    + "".join(
        f'<div{f" data-t=v{i}" if i < 4 else ""}><b>{t}</b><div class="muted">{n} views</div></div>'
        for i, (t, n) in enumerate(
            [
                ("Flattening the back of a plane iron", "88K"),
                ("Japanese water stones for beginners", "1.2M"),
                ("Restoring a rusty hand saw", "430K"),
                ("Five joints every woodworker should know", "2.3M"),
                ("Building a simple workbench", "760K"),
                ("Strop or no strop?", "51K"),
                ("Cutting dovetails by hand", "1.1M"),
                ("Choosing your first chisels", "300K"),
                ("Oil or wax finish?", "95K"),
                ("Shop tour 2026", "40K"),
                ("Sharpening a card scraper", "120K"),
                ("Saw bench from one board", "64K"),
                ("Marking gauges compared", "33K"),
                ("A tool chest in a weekend", "410K"),
                ("Honing guides: worth it?", "77K"),
                ("Workshop dust collection on a budget", "220K"),
            ]
        )
    )
    + "</div>",
)

T_BANK = Screen(
    "t11_bank",
    "test",
    "Transfer money - Harbour Bank",
    _top("Harbour Bank", "Accounts", "Payments", "Cards")
    + """
<div class="side"><div data-t="accounts">Accounts</div><div class="on" data-t="transfer_tab">Transfer</div><div data-t="statements">Statements</div><div data-t="logout">Log out</div></div>
<div class="main card" style="width:620px">
  <h2 style="margin-top:0">Transfer money</h2>
  <label>From</label><div class="f" data-t="from" style="line-height:36px">Current account &middot; &pound;2,431.18 &#9662;</div>
  <label>Recipient</label><input class="f" data-t="recipient" placeholder="Name or sort code">
  <label>Amount (&pound;)</label><input class="f" data-t="amount" placeholder="0.00">
  <label>Reference</label><input class="f" data-t="reference" placeholder="Shown on their statement">
  <div class="row" style="margin-top:22px"><span class="btn" data-t="confirm">Confirm transfer</span><span class="btn sec" data-t="cancel">Cancel</span></div>
</div>""",
)

T_RECIPE = Screen(
    "t12_recipe",
    "test",
    "Lemon drizzle cake - Kitchen Notes",
    '<div class="top" style="position:fixed;left:0;right:0;top:0;z-index:2"><span class="brand">Kitchen Notes</span><span>Recipes</span><span>Baking</span><span class="btn sec" data-t="print" style="margin-left:auto">Print</span></div>'
    + '<div data-t="recipe" style="margin:76px auto 0;width:760px;background:#fff;padding:24px 32px;min-height:2600px">'
    + "<h1>Lemon drizzle cake</h1>"
    + _lorem(3, "classic lemon drizzle")
    + "<h2>Ingredients</h2><ul style='line-height:1.9'>"
    + "".join(
        f"<li>{i}</li>"
        for i in [
            "225 g unsalted butter, softened",
            "225 g caster sugar",
            "4 large eggs",
            "275 g self-raising flour",
            "2 tsp baking powder",
            "4 tbsp milk",
            "finely grated zest of 2 lemons",
            "juice of 1.5 lemons",
            "175 g granulated sugar for the drizzle",
        ]
    )
    + "</ul><h2>Method</h2><ol style='line-height:1.9'>"
    + "".join(
        f"<li>{m}</li>"
        for m in [
            "Heat the oven to 180C (160C fan) and line a 30 x 23 cm tin.",
            "Beat the butter, sugar, eggs, flour, baking powder, milk and zest until smooth.",
            "Spread into the tin and level the top.",
            "Bake for 35-40 minutes until golden and a skewer comes out clean.",
            "Mix the lemon juice and granulated sugar.",
            "Spoon the drizzle over the warm cake.",
            "Leave to cool in the tin, then cut into 24 squares.",
        ]
    )
    + "</ol>"
    + _lorem(6, "classic lemon drizzle")
    + "</div>",
    scroll_y=450,
)

T_CHECKOUT = Screen(
    "t13_checkout",
    "test",
    "Checkout - Gadget Corner",
    _top("Gadget Corner", "Secure checkout")
    + """
<div class="card" style="position:absolute;left:40px;top:76px;width:700px">
  <h2 style="margin-top:0">Shipping address</h2>
  <label>Full name</label><input class="f" data-t="name">
  <label>Address line 1</label><input class="f" data-t="address">
  <div class="row"><div style="flex:1"><label>City</label><input class="f" data-t="city"></div>
    <div style="width:200px"><label>Postcode</label><input class="f" data-t="postcode"></div></div>
  <label>Delivery notes</label><textarea class="f" data-t="notes" style="height:80px;padding:8px">Leave with the neighbour at number 12 if nobody is home.</textarea>
</div>
<div class="card" style="position:absolute;left:770px;top:76px;width:470px">
  <h3 style="margin-top:0">Order summary</h3>
  <div class="row"><span style="flex:1">Quiet One headphones</span><span>&pound;199.00</span></div>
  <div class="row"><span style="flex:1">Glide wireless mouse</span><span>&pound;35.00</span></div>
  <div class="row" style="margin-top:12px"><input class="f" data-t="coupon" placeholder="Discount code"><span class="btn sec" data-t="apply">Apply</span></div>
  <div class="row" style="margin-top:12px"><b style="flex:1">Total</b><b>&pound;234.00</b></div>
  <span class="btn" data-t="place_order" style="width:100%;text-align:center;margin-top:16px">Place order</span>
</div>""",
)

T_FEED = Screen(
    "t14_feed",
    "test",
    "Home - Circle",
    '<div class="top" style="position:fixed;left:0;right:0;top:0;z-index:2"><span class="brand">Circle</span><span>Home</span><span>Groups</span><span>Messages</span></div>'
    + """<div class="card" style="position:absolute;left:320px;top:72px;width:640px"><textarea class="f" data-t="composer" placeholder="What's on your mind?" style="height:70px;padding:8px"></textarea>
    <div style="text-align:right;margin-top:8px"><span class="btn blue" data-t="post">Post</span></div></div>
<div data-t="feed" style="position:absolute;left:320px;top:230px;width:640px;height:560px;overflow-y:auto">
"""
    + "".join(
        f'<div class="card" style="margin-top:14px"><b>{who}</b><p>{txt}</p><span class="btn sec"{f" data-t=like{i}" if i < 3 else ""}>Like</span> <span class="btn sec">Comment</span></div>'
        for i, (who, txt) in enumerate(
            [
                (
                    "Sara Lind",
                    "Finally finished the quilt I started in January. Never again (until next time).",
                ),
                ("Jonas Weber", "Anyone know a good bike mechanic near the station?"),
                (
                    "Priya Nair",
                    "Our team won the pub quiz! The tie-break was the height of Ben Nevis.",
                ),
                ("Omar Haddad", "Sourdough attempt number four. Getting there."),
                ("Lena Fischer", "Reminder: the book club meets on Thursday, chapter 9 to 14."),
                ("Mira Sol", "New song out on Friday. Thank you for listening."),
                ("Workshop Wren", "A sharp chisel is a safe chisel."),
                ("City Times", "The riverside project enters its second phase."),
            ]
        )
    )
    + "</div>",
    vscroll=(("feed", 300),),
)

T_DIALOG = Screen(
    "t15_dialog",
    "test",
    "Report.docx - Writer",
    _top("Writer", "File", "Edit", "Insert", "Format")
    + """
<div class="card" style="position:absolute;left:190px;top:90px;width:900px;height:680px;opacity:.55">
  <h1>Quarterly report</h1><p>Revenue grew in every region except the north, where two large contracts ended.</p>
  <p>Costs were flat. Headcount rose by four.</p></div>
<div style="position:absolute;inset:0;background:rgba(0,0,0,.35)"></div>
<div class="card" data-t="dialog" style="position:absolute;left:430px;top:280px;width:420px">
  <h3 style="margin-top:0">Save changes to &ldquo;Report.docx&rdquo;?</h3>
  <div class="muted">Your changes will be lost if you don't save them.</div>
  <div class="row" style="margin-top:20px;justify-content:flex-end"><span class="btn" data-t="save">Save</span><span class="btn sec" data-t="dont_save">Don't save</span><span class="btn sec" data-t="cancel">Cancel</span></div>
</div>""",
)

T_KANBAN = Screen(
    "t16_kanban",
    "test",
    "Board - Launch plan",
    _top("Boards", "Launch plan")
    + """
<div class="strip" data-t="board" style="position:absolute;left:20px;top:76px;width:1240px;height:700px;align-items:flex-start">
"""
    + "".join(
        f'<div class="card" style="flex:0 0 300px;white-space:normal;background:#eef1f4"><h3 style="margin-top:0">{col}</h3>'
        + "".join(f'<div class="card" style="margin-bottom:10px">{c}</div>' for c in cards)
        + (
            '<input class="f" data-t="add_card" placeholder="+ Add a card">'
            if col == "In progress"
            else ""
        )
        + "</div>"
        for col, cards in [
            ("Backlog", ["Localise onboarding", "Accessibility audit"]),
            ("To do", ["Write release notes", "Update pricing page"]),
            ("In progress", ["Fix login timeout", "Press kit"]),
            ("Review", ["New icon set"]),
            ("Done", ["Beta feedback survey", "Crash fixes"]),
            ("Archived", ["Old landing page"]),
        ]
    )
    + "</div>",
    hscroll=(("board", 320),),
)

SCREENS: tuple[Screen, ...] = (
    D_LOGIN,
    D_STORE,
    D_MAIL,
    D_SETTINGS,
    D_ARTICLE,
    D_CHAT,
    D_PHOTOS,
    D_EDITOR,
    T_SIGNUP,
    T_RESULTS,
    T_FLIGHTS,
    T_CALENDAR,
    T_FILES,
    T_MUSIC,
    T_TODO,
    T_MAPS,
    T_SHEET,
    T_VIDEO,
    T_BANK,
    T_RECIPE,
    T_CHECKOUT,
    T_FEED,
    T_DIALOG,
    T_KANBAN,
)
BY_ID = {s.id: s for s in SCREENS}
