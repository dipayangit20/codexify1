"""
TalentBridge - AI-Powered Talent Marketplace
Revenue Model: 10% platform commission per booking
Features: User Auth (JSON DB) + RAG Talent Search + Event Planner Chatbot
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import os
import re
import hashlib
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = "talentbridge_secret_2024"

ARTISTS_FILE = "artists.json"
USERS_FILE   = "users.json"

# ─────────────────────────────────────────────────────────────────
# USER DATABASE (JSON)
# ─────────────────────────────────────────────────────────────────
def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump([], f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def find_user_by_email(email):
    users = load_users()
    return next((u for u in users if u["email"].lower() == email.lower()), None)

def register_user(name, email, password, role):
    if find_user_by_email(email):
        return None, "An account with this email already exists."
    users  = load_users()
    new_id = max((u["id"] for u in users), default=0) + 1
    user   = {
        "id":         new_id,
        "name":       name,
        "email":      email,
        "password":   hash_password(password),
        "role":       role,
        "created_at": datetime.now().isoformat(),
        "avatar":     f"https://i.pravatar.cc/150?img={new_id + 50}"
    }
    users.append(user)
    save_users(users)
    return user, None

def login_user(email, password):
    user = find_user_by_email(email)
    if not user:
        return None, "No account found with this email."
    if user["password"] != hash_password(password):
        return None, "Incorrect password. Please try again."
    return user, None

# ─────────────────────────────────────────────────────────────────
# ARTISTS DATABASE
# ─────────────────────────────────────────────────────────────────
def load_artists():
    with open(ARTISTS_FILE, "r") as f:
        return json.load(f)

def save_artists(artists):
    with open(ARTISTS_FILE, "w") as f:
        json.dump(artists, f, indent=2)

# ─────────────────────────────────────────────────────────────────
# RAG / AI SYSTEM — lazy loaded
# ─────────────────────────────────────────────────────────────────
_model            = None
_index            = None
_artist_embeddings = None
_artists_for_rag  = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def build_artist_text(artist):
    return (
        f"{artist['name']} is a {artist['category']} in {artist['location']}. "
        f"Skills: {', '.join(artist['skills'])}. "
        f"Price: ${artist['price_min']} to ${artist['price_max']}. "
        f"Rating: {artist['rating']}. "
        f"Available for: {', '.join(artist['event_types'])}. "
        f"{artist['bio']}"
    )

def build_index():
    global _index, _artist_embeddings, _artists_for_rag
    try:
        import faiss
        artists = load_artists()
        model   = get_model()
        texts   = [build_artist_text(a) for a in artists]
        embs    = model.encode(texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embs)
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs)
        _index, _artist_embeddings, _artists_for_rag = idx, embs, artists
        print("[RAG] FAISS index built successfully")
    except Exception as e:
        print(f"[RAG] FAISS unavailable, cosine fallback active: {e}")

def cosine_search(query_vec, top_k=6):
    artists = load_artists()
    model   = get_model()
    texts   = [build_artist_text(a) for a in artists]
    embs    = model.encode(texts, convert_to_numpy=True).astype("float32")
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    en = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    scores = en @ qn
    top    = np.argsort(scores)[::-1][:top_k]
    return [(artists[i], float(scores[i])) for i in top]

def extract_filters(text):
    filters = {}
    m = re.search(r'\$(\d[\d,]*)[kK]?', text)
    if m:
        raw = m.group(1).replace(",", "")
        val = int(raw)
        if 'k' in m.group(0).lower():
            val *= 1000
        if val >= 100:
            filters["max_budget"] = val
    cities = ["new york", "chicago", "los angeles", "houston", "miami",
              "atlanta", "dallas", "seattle", "boston", "nashville",
              "las vegas", "detroit", "new orleans", "san francisco"]
    tl = text.lower()
    for c in cities:
        if c in tl:
            filters["city"] = c
            break
    cats = ["singer", "dancer", "musician", "painter", "photographer", "dj", "performer"]
    for cat in cats:
        if cat in tl:
            filters["category"] = cat.capitalize()
            break
    return filters

def apply_filters(results, filters):
    out = []
    for artist, score in results:
        ok = True
        if "max_budget" in filters and artist["price_min"] > filters["max_budget"]:
            ok = False
        if "city" in filters and filters["city"] not in artist["location"].lower():
            ok = False
        if "category" in filters and artist["category"].lower() != filters["category"].lower():
            ok = False
        if ok:
            out.append((artist, score))
    return out if out else results

def rag_search(query, top_k=3):
    try:
        model   = get_model()
        qvec    = model.encode([query], convert_to_numpy=True).astype("float32")[0]
        filters = extract_filters(query)
        if _index is not None:
            import faiss
            q = qvec.copy().reshape(1, -1)
            faiss.normalize_L2(q)
            scores, indices = _index.search(q, min(top_k * 3, len(_artists_for_rag)))
            results = [(_artists_for_rag[i], float(scores[0][j]))
                       for j, i in enumerate(indices[0]) if i >= 0]
        else:
            results = cosine_search(qvec, top_k=top_k * 3)
        filtered = apply_filters(results, filters)[:top_k]
        return (filtered if filtered else results[:top_k]), filters
    except Exception as e:
        print(f"[RAG] Search error: {e}")
        artists = load_artists()
        return [(a, 0.5) for a in artists[:top_k]], {}

# ─────────────────────────────────────────────────────────────────
# EVENT PLANNER ENGINE
# ─────────────────────────────────────────────────────────────────

BUDGET_TEMPLATES = {
    "wedding": {
        "💐 Venue & Decor":        0.30,
        "🍽️ Catering & Drinks":    0.25,
        "🎵 Entertainment":        0.20,
        "📸 Photography & Video":  0.10,
        "💐 Flowers & Styling":    0.07,
        "💌 Invitations & Misc":   0.05,
        "🛡️ Contingency Buffer":   0.03,
    },
    "corporate": {
        "🏛️ Venue & AV Setup":     0.35,
        "🍽️ Catering":             0.25,
        "🎵 Entertainment":        0.15,
        "📸 Photography/Video":    0.10,
        "🎨 Branding & Signage":   0.08,
        "🚚 Logistics & Misc":     0.07,
    },
    "birthday": {
        "🏛️ Venue":                0.25,
        "🍕 Food & Drinks":        0.30,
        "🎵 Entertainment":        0.20,
        "📸 Photography":          0.08,
        "🎂 Cake & Desserts":      0.10,
        "🎈 Decorations":          0.07,
    },
    "festival": {
        "🎤 Stage & Sound":        0.30,
        "🎵 Artists/Performers":   0.30,
        "🚨 Security & Staffing":  0.15,
        "📣 Marketing":            0.10,
        "🍔 Food Vendors":         0.10,
        "🚚 Logistics":            0.05,
    },
    "party": {
        "🏛️ Venue":                0.25,
        "🍕 Food & Drinks":        0.35,
        "🎵 Entertainment":        0.20,
        "📸 Photography":          0.08,
        "🎈 Decorations":          0.12,
    },
    "gala": {
        "🏛️ Venue & Ambiance":     0.30,
        "🍽️ Catering (Fine Dining)":0.28,
        "🎵 Entertainment":        0.18,
        "📸 Photography/Video":    0.10,
        "💐 Flowers & Styling":    0.08,
        "💌 Invitations & Misc":   0.06,
    },
    "concert": {
        "🎤 Artist Booking":       0.40,
        "🏛️ Venue & Stage":        0.25,
        "🔊 Sound & Lighting":     0.15,
        "📣 Marketing & Tickets":  0.10,
        "🚨 Security & Staff":     0.07,
        "🚚 Logistics":            0.03,
    },
    "default": {
        "🏛️ Venue":                0.30,
        "🍽️ Catering":             0.25,
        "🎵 Entertainment":        0.20,
        "📸 Photography":          0.10,
        "🎈 Decorations":          0.08,
        "🚚 Miscellaneous":        0.07,
    }
}

EVENT_TIMELINES = {
    "wedding": [
        ("12–18 Months Before", "Book venue & set budget. Begin wedding planning."),
        ("9–12 Months Before",  "Book entertainment (singer, DJ, band) & photographer."),
        ("6–9 Months Before",   "Finalize catering, florals, send save-the-dates."),
        ("3–6 Months Before",   "Book hair/makeup, confirm all vendors."),
        ("1–2 Months Before",   "Final fittings, rehearsal dinner, finalize seating."),
        ("1 Week Before",       "Confirm all bookings, prepare payments, final run-through."),
        ("Day Of 💍",           "Arrive early, coordinate vendors — enjoy your day!"),
    ],
    "corporate": [
        ("3–6 Months Before", "Define objectives, set budget, secure venue."),
        ("2–3 Months Before", "Book speakers/entertainment, plan catering, send invites."),
        ("6–8 Weeks Before",  "Confirm AV setup, branding materials, agenda."),
        ("2–4 Weeks Before",  "Send reminders, confirm headcount, brief all vendors."),
        ("1 Week Before",     "Run tech checks, prepare materials, confirm logistics."),
        ("Day Of 🏢",         "Arrive 2h early, setup registration, welcome attendees."),
    ],
    "birthday": [
        ("6–8 Weeks Before", "Choose theme, book venue, estimate guest count."),
        ("4–6 Weeks Before", "Book entertainment, send invitations, plan menu."),
        ("2–4 Weeks Before", "Order cake, confirm vendors, plan decorations."),
        ("1 Week Before",    "Get RSVPs, confirm final numbers, prepare playlist."),
        ("Day Of 🎂",        "Set up early, welcome guests — have an amazing time!"),
    ],
    "festival": [
        ("6–12 Months Before", "Secure permits, book venue, finalize lineup budget."),
        ("4–6 Months Before",  "Book headlining artists, launch ticket sales."),
        ("2–4 Months Before",  "Book supporting acts, confirm food vendors, marketing."),
        ("4–8 Weeks Before",   "Finalize stage plan, security briefing, volunteer training."),
        ("1–2 Weeks Before",   "Load-in schedule, final checks, media accreditation."),
        ("Day Of 🎪",          "Gates open — enjoy the show!"),
    ],
    "party": [
        ("4–6 Weeks Before", "Book venue and entertainment."),
        ("2–4 Weeks Before", "Send invites, finalize catering and decorations."),
        ("1 Week Before",    "Confirm RSVPs and vendor headcounts."),
        ("Day Of 🥂",        "Set up early, welcome guests — party on!"),
    ],
    "gala": [
        ("4–6 Months Before", "Book venue, set gala theme, form planning committee."),
        ("2–4 Months Before", "Book entertainment, finalize catering, send invitations."),
        ("4–8 Weeks Before",  "Confirm all vendors, plan program & speeches."),
        ("1–2 Weeks Before",  "Final RSVP count, seating plan, briefing vendors."),
        ("Day Of ✨",         "Arrive early for setup — make it a night to remember!"),
    ],
    "default": [
        ("6–8 Weeks Before", "Book venue and set overall budget."),
        ("4–6 Weeks Before", "Book entertainment and catering, send invites."),
        ("2–4 Weeks Before", "Confirm vendors, finalize guest list."),
        ("1 Week Before",    "Confirm all bookings and logistics."),
        ("Day Of ✨",        "Arrive early and enjoy!"),
    ]
}

ARTIST_CATEGORY_FOR_EVENT = {
    "wedding":   ["Singer", "Musician", "Photographer", "DJ", "Dancer", "Painter"],
    "corporate": ["Performer", "DJ", "Musician", "Photographer", "Singer"],
    "birthday":  ["DJ", "Performer", "Musician", "Photographer"],
    "festival":  ["Singer", "DJ", "Dancer", "Musician", "Performer"],
    "party":     ["DJ", "Musician", "Performer", "Photographer"],
    "gala":      ["Singer", "Musician", "Photographer", "Dancer", "Painter"],
    "concert":   ["Singer", "DJ", "Musician", "Dancer"],
    "default":   ["Singer", "DJ", "Musician", "Photographer", "Performer"],
}

PRO_TIPS = {
    "wedding":   [
        "📅 Book vendors at least 12 months ahead for peak wedding season.",
        "💰 Keep 5–10% of budget as emergency buffer — always.",
        "👤 A day-of coordinator saves stress and ensures everything runs on time.",
    ],
    "corporate": [
        "🎯 Define your KPIs before booking entertainment — align with event goals.",
        "📹 Record key sessions for post-event ROI content marketing.",
        "🎮 Interactive entertainment (magicians, live painters) boosts engagement 40%.",
    ],
    "birthday":  [
        "🎵 Book DJ or entertainer at least 4–6 weeks ahead.",
        "🎁 Surprise elements (flash mob, live singer reveal) create lasting memories.",
        "🍕 Catering quality = ~30% of overall guest experience.",
    ],
    "festival":  [
        "📋 Secure all permits before any public announcements.",
        "☔ Always have a rain contingency plan — outdoor events depend on it.",
        "🏥 On-site medical team is non-negotiable for large outdoor festivals.",
    ],
    "party":     [
        "🎶 Great music = great party. Invest in a good DJ or live musician.",
        "🥂 Over-cater on drinks by 15% — you'll thank yourself later.",
        "📸 Even a casual photographer captures priceless memories.",
    ],
    "gala":      [
        "✨ Invest in lighting — it transforms any venue for a fraction of cost.",
        "🎤 A professional emcee keeps the program flowing smoothly.",
        "💌 Physical invitations for a gala elevate the guest experience.",
    ],
    "default":   [
        "📋 Always have a backup plan for key vendors.",
        "✅ Confirm all bookings 1 week before the event.",
        "💰 Keep a contingency budget of at least 5–10%.",
    ]
}

def detect_event_type(text):
    t = text.lower()
    if any(w in t for w in ["wedding", "marriage", "bride", "groom", "nuptial"]):
        return "wedding"
    if any(w in t for w in ["corporate", "company", "office", "conference", "business", "launch", "product launch"]):
        return "corporate"
    if any(w in t for w in ["birthday", "bday", "birth day"]):
        return "birthday"
    if any(w in t for w in ["festival", "fest", "music festival"]):
        return "festival"
    if any(w in t for w in ["gala", "charity", "fundraiser", "award", "black tie"]):
        return "gala"
    if any(w in t for w in ["concert", "live show", "performance night"]):
        return "concert"
    if any(w in t for w in ["party", "celebration", "get together", "house party"]):
        return "party"
    return "default"

def extract_budget_from_text(text):
    patterns = [
        (r'\$(\d[\d,]*)[kK]', True),
        (r'(\d[\d,]*)[kK]\s*(?:dollar|usd|budget)?', True),
        (r'\$(\d[\d,]*)', False),
        (r'(\d[\d,]{2,})\s*(?:dollar|usd|budget)', False),
        (r'budget\s+(?:of|is|:)?\s*\$?(\d[\d,]*)', False),
    ]
    for pat, is_k in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(1).replace(",", "")
            val = int(raw)
            if is_k:
                val *= 1000
            if val >= 100:
                return val
    return None

def is_event_plan_request(text):
    t = text.lower()
    plan_kw  = ["plan", "planning", "organize", "help me", "budget", "breakdown",
                 "schedule", "timeline", "prepare", "arrange", "full plan", "complete",
                 "how much", "what do i need", "guide", "checklist"]
    event_kw = ["wedding", "party", "corporate", "birthday", "festival", "gala",
                 "concert", "event", "celebration", "ceremony"]
    has_plan  = any(k in t for k in plan_kw)
    has_event = any(e in t for e in event_kw)
    has_budget = extract_budget_from_text(text) is not None
    return (has_plan and has_event) or (has_event and has_budget)

def extract_event_name(text):
    patterns = [
        r'(?:for\s+(?:a|my|our|the)\s+)([\w\s]+?)(?:\s+with|\s+budget|\s+of|\s+event|\s+on|$)',
        r'(?:planning\s+(?:a|my|our)\s+)([\w\s]+?)(?:\s+with|\s+budget|\s+of|$)',
        r'(?:help\s+me\s+(?:plan|organize)\s+(?:a|my)?\s*)([\w\s]+?)(?:\s+with|\s+budget|$)',
        r'(?:organizing\s+(?:a|my|our)\s+)([\w\s]+?)(?:\s+with|\s+budget|$)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().rstrip("., ").title()
            if 2 < len(name) < 60:
                return name
    event_words = {
        "wedding": "Wedding", "birthday": "Birthday Party", "corporate": "Corporate Event",
        "festival": "Music Festival", "gala": "Gala Evening", "concert": "Concert Night",
        "party": "Party"
    }
    tl = text.lower()
    for key, label in event_words.items():
        if key in tl:
            return label
    return "Your Event"

def generate_event_plan(event_name, budget, query):
    event_type = detect_event_type(query + " " + event_name)
    template   = BUDGET_TEMPLATES.get(event_type, BUDGET_TEMPLATES["default"])
    timeline   = EVENT_TIMELINES.get(event_type, EVENT_TIMELINES["default"])
    tips       = PRO_TIPS.get(event_type, PRO_TIPS["default"])
    pref_cats  = ARTIST_CATEGORY_FOR_EVENT.get(event_type, ARTIST_CATEGORY_FOR_EVENT["default"])

    # Budget breakdown
    breakdown = {}
    for cat, pct in template.items():
        breakdown[cat] = round(budget * pct)

    # Entertainment budget
    ent_budget = next(
        (v for k, v in breakdown.items() if "entertainment" in k.lower() or "artist" in k.lower()),
        int(budget * 0.20)
    )

    # Artist matching
    artists    = load_artists()
    candidates = [a for a in artists
                  if a["price_min"] <= ent_budget and a["category"] in pref_cats]
    candidates.sort(key=lambda x: x["rating"], reverse=True)
    top_artists = candidates[:3]

    if len(top_artists) < 2:
        extras = [a for a in artists
                  if a["price_min"] <= ent_budget and a not in top_artists]
        extras.sort(key=lambda x: x["rating"], reverse=True)
        top_artists = (top_artists + extras)[:3]

    return {
        "event_name":           event_name,
        "event_type":           event_type,
        "total_budget":         budget,
        "breakdown":            breakdown,
        "timeline":             timeline,
        "tips":                 tips,
        "artists":              top_artists,
        "entertainment_budget": ent_budget
    }

# ─────────────────────────────────────────────────────────────────
# ROUTES — AUTH
# ─────────────────────────────────────────────────────────────────

@app.route("/auth", methods=["GET", "POST"])
def auth():
    error      = None
    active_tab = request.args.get("tab", "register")

    if request.method == "POST":
        action = request.form.get("action")

        if action == "register":
            name     = request.form.get("name", "").strip()
            email    = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            role     = request.form.get("role", "hirer")

            if not name or not email or not password:
                error = "All fields are required."; active_tab = "register"
            elif len(password) < 6:
                error = "Password must be at least 6 characters."; active_tab = "register"
            else:
                user, err = register_user(name, email, password, role)
                if err:
                    error = err; active_tab = "register"
                else:
                    session["user"] = {
                        "id": user["id"], "name": user["name"],
                        "email": user["email"], "role": user["role"],
                        "avatar": user["avatar"]
                    }
                    return redirect(url_for("build_profile") if role == "artist" else url_for("home"))

        elif action == "login":
            email    = request.form.get("email", "").strip()
            password = request.form.get("password", "")

            if not email or not password:
                error = "Email and password are required."; active_tab = "login"
            else:
                user, err = login_user(email, password)
                if err:
                    error = err; active_tab = "login"
                else:
                    session["user"] = {
                        "id": user["id"], "name": user["name"],
                        "email": user["email"], "role": user["role"],
                        "avatar": user.get("avatar", "https://i.pravatar.cc/150?img=50")
                    }
                    return redirect(url_for("home"))

    return render_template("auth.html", error=error, active_tab=active_tab)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))

# ─────────────────────────────────────────────────────────────────
# ROUTES — PAGES
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def landing():
    artists  = load_artists()
    featured = [a for a in artists if a["rating"] >= 4.8][:6]
    return render_template("landing.html", featured=featured, user=session.get("user"))

@app.route("/home")
def home():
    artists   = load_artists()
    category  = request.args.get("category", "")
    location  = request.args.get("location", "")
    search    = request.args.get("search", "")
    price_max = request.args.get("price_max", "")

    filtered = artists
    if category:
        filtered = [a for a in filtered if a["category"].lower() == category.lower()]
    if location:
        filtered = [a for a in filtered if location.lower() in a["location"].lower()]
    if search:
        q = search.lower()
        filtered = [a for a in filtered if
                    q in a["name"].lower() or q in a["category"].lower() or
                    any(q in s.lower() for s in a["skills"]) or
                    any(q in e.lower() for e in a["event_types"])]
    if price_max:
        try:
            pm = int(price_max)
            filtered = [a for a in filtered if a["price_min"] <= pm]
        except:
            pass

    categories = sorted(set(a["category"] for a in artists))
    cities     = sorted(set(a["city"]     for a in artists))
    return render_template("home.html", artists=filtered, categories=categories,
                           cities=cities, filters=request.args, user=session.get("user"))

@app.route("/build-profile", methods=["GET", "POST"])
def build_profile():
    if request.method == "POST":
        artists = load_artists()
        new_id  = max(a["id"] for a in artists) + 1 if artists else 1
        skills  = [s.strip() for s in request.form.get("skills", "").split(",") if s.strip()]
        parts   = request.form.get("price_range", "0-0").split("-")
        pmin    = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        pmax    = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

        new_artist = {
            "id": new_id, "name": request.form.get("name", ""),
            "email": request.form.get("email", ""), "phone": request.form.get("phone", ""),
            "address": request.form.get("address", ""), "city": request.form.get("city", ""),
            "location": f"{request.form.get('city','')}, {request.form.get('location','')}",
            "category": request.form.get("category", ""), "skills": skills,
            "price_min": pmin, "price_max": pmax,
            "availability": request.form.get("availability", ""),
            "portfolio_link": request.form.get("portfolio_link", ""),
            "bio": request.form.get("bio", ""),
            "rating": 0.0, "reviews": 0, "badge": "New Artist",
            "avatar": f"https://i.pravatar.cc/150?img={new_id + 40}",
            "event_types": []
        }
        artists.append(new_artist)
        save_artists(artists)
        return redirect(url_for("artist_profile", artist_id=new_id))
    return render_template("build_profile.html", user=session.get("user"))

@app.route("/artist/<int:artist_id>")
def artist_profile(artist_id):
    artists = load_artists()
    artist  = next((a for a in artists if a["id"] == artist_id), None)
    if not artist:
        return redirect(url_for("home"))
    return render_template("artist_profile.html", artist=artist, user=session.get("user"))

# ─────────────────────────────────────────────────────────────────
# ROUTE — EVENT PLANNER CHATBOT
# ─────────────────────────────────────────────────────────────────

@app.route("/plan-chat", methods=["POST"])
def plan_chat():
    try:
        data    = request.get_json()
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"type": "text", "response": "Please describe your event and budget!"})

        budget = extract_budget_from_text(message)

        # PATH 1: Full event plan
        if is_event_plan_request(message) and budget:
            event_name = extract_event_name(message)
            plan = generate_event_plan(event_name, budget, message)
            return jsonify({
                "type":                 "event_plan",
                "event_name":           plan["event_name"],
                "event_type":           plan["event_type"],
                "total_budget":         plan["total_budget"],
                "breakdown":            plan["breakdown"],
                "timeline":             plan["timeline"],
                "tips":                 plan["tips"],
                "entertainment_budget": plan["entertainment_budget"],
                "artists": [{
                    "id": a["id"], "name": a["name"], "category": a["category"],
                    "location": a["location"], "rating": a["rating"],
                    "price_min": a["price_min"], "badge": a.get("badge", ""),
                    "avatar": a["avatar"], "skills": a["skills"][:3],
                } for a in plan["artists"]]
            })

        # PATH 2: Artist search
        matches, filters = rag_search(message, top_k=3)
        response = _build_artist_response(message, matches, filters)
        return jsonify({
            "type":     "artists",
            "response": response,
            "artists":  _format_cards(matches)
        })

    except Exception as e:
        print(f"[Plan Chat Error] {e}")
        import traceback; traceback.print_exc()
        return jsonify({"type": "text", "response": "Something went wrong. Please try again."})

@app.route("/ai-chat", methods=["POST"])
def ai_chat():
    try:
        data  = request.get_json()
        query = data.get("message", "").strip()
        if not query:
            return jsonify({"response": "Please enter a message.", "artists": []})
        matches, filters = rag_search(query, top_k=3)
        return jsonify({
            "response": _build_artist_response(query, matches, filters),
            "artists":  _format_cards(matches)
        })
    except Exception as e:
        return jsonify({"response": "Error. Please try again.", "artists": []})

# ─────────────────────────────────────────────────────────────────
# ROUTE — BOOKING
# ─────────────────────────────────────────────────────────────────

@app.route("/book/<int:artist_id>", methods=["POST"])
def book_artist(artist_id):
    artists = load_artists()
    artist  = next((a for a in artists if a["id"] == artist_id), None)
    if not artist:
        return jsonify({"success": False, "message": "Artist not found"}), 404
    data         = request.get_json() or {}
    price        = data.get("price", artist["price_min"])
    commission   = round(price * 0.10, 2)
    payout       = round(price - commission, 2)
    booking      = {
        "booking_id":          f"TB-{artist_id}-{abs(hash(data.get('event_date','X'))) % 10000:04d}",
        "artist_name":         artist["name"],
        "event_date":          data.get("event_date", "TBD"),
        "event_type":          data.get("event_type", "Event"),
        "agreed_price":        price,
        "platform_commission": commission,
        "artist_payout":       payout,
        "status":              "confirmed"
    }
    return jsonify({"success": True, "message": f"Booking confirmed for {artist['name']}!", "booking": booking})

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _build_artist_response(query, matches, filters):
    if not matches:
        return "No artists found. Try broadening your search."
    lines = ["Here are the best matches I found:\n"]
    for i, (a, score) in enumerate(matches, 1):
        pct = min(int(score * 100), 99)
        lines.append(
            f"**{i}. {a['name']}** — {a['category']}\n"
            f"   📍 {a['location']} | ⭐ {a['rating']} | 💰 From ${a['price_min']}\n"
            f"   🎯 AI Match: {pct}%\n"
        )
    notes = []
    if "max_budget" in filters: notes.append(f"budget ≤ ${filters['max_budget']}")
    if "city"       in filters: notes.append(f"city: {filters['city'].title()}")
    if "category"   in filters: notes.append(f"type: {filters['category']}")
    if notes:
        lines.append(f"*Filtered by: {', '.join(notes)}*")
    return "\n".join(lines)

def _format_cards(matches):
    return [{
        "id": a["id"], "name": a["name"], "category": a["category"],
        "location": a["location"], "rating": a["rating"],
        "price_min": a["price_min"], "badge": a.get("badge",""),
        "avatar": a["avatar"], "match_score": min(int(s*100), 99)
    } for a, s in matches]

# ─────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[TalentBridge] Building RAG index...")
    build_index()
    print("[TalentBridge] Server ready → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)