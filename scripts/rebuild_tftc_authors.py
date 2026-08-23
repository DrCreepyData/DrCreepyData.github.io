"""
Rebuild tftc_authors.json from all_posts.json.
Run after fetching nosleep posts for TFTC authors.
"""
import json, numpy as np, os, sys
from collections import defaultdict

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

with open(os.path.join(DATA, "all_posts.json")) as f:
    all_posts = json.load(f)
with open(os.path.join(DATA, "episodes.json")) as f:
    episodes = json.load(f)
with open(os.path.join(DATA, "episode_source_stories.json")) as f:
    ep_source = json.load(f)

all_id_map = {p["id"]: p for p in all_posts}

EXCLUDED = {"Need Help","Offering Help","Writing Help","Story Shoutout",
            "Fan Story Discussion","Mod Announcement","Publishing Announcement",
            "Prompt (MOD APPROVED)","Need Help (ADVICE FLAIR)","Offering Help (ADVICE FLAIR)",
            "Looking for Feedback","Discussion","Fan Art","Story Notes","Venting",
            "Writing Prompt","JUST POSTED","Activities&Events","Non-Fiction"}
MIN_WORDS = 400

def passes(p):
    text = (p.get("selftext") or "").strip()
    if len(text.split()) < MIN_WORDS or text in ("[removed]","[deleted]"):
        return False
    if (p.get("flair") or "").strip() in EXCLUDED:
        return False
    return True

def subreddit_of(p):
    url = p.get("url","")
    if "/r/" in url:
        return url.split("/r/")[1].split("/")[0]
    return "unknown"

# Build episode -> author map
author_to_episodes = defaultdict(list)
ep_seen = defaultdict(set)
for ep in episodes:
    sids = ep.get("story_ids") or ([ep["story_id"]] if ep.get("story_id") else [])
    ep_info = {"title": ep["title"], "date": ep.get("date",""), "url": ep.get("url","")}
    for sid in sids:
        src = ep_source.get(sid, {})
        author = src.get("author","")
        if not author and sid in all_id_map:
            author = all_id_map[sid].get("author","")
        if not author:
            for pid in (src.get("post_ids") or []):
                if pid in all_id_map:
                    author = all_id_map[pid].get("author","")
                    break
        if author and author not in ("[deleted]",""):
            key = ep_info["url"]
            if key not in ep_seen[author]:
                ep_seen[author].add(key)
                author_to_episodes[author].append(ep_info)

# Collect TFTC authors and their cross-subreddit stories
tftc_authors_set = set()
for p in all_posts:
    if "/r/TalesFromTheCreeps/" in p.get("url",""):
        tftc_authors_set.add(p.get("author",""))
tftc_authors_set.discard("")
tftc_authors_set.discard("[deleted]")
tftc_authors_set.discard("AutoModerator")

SHOWN_SUBS = {"TalesFromTheCreeps", "nosleep", "shortscarystories", "TheCrypticCompendium"}

author_data = defaultdict(lambda: {"tftc": [], "nosleep": [], "shortscarystories": [], "TheCrypticCompendium": [], "episodes": []})

for p in all_posts:
    sub = subreddit_of(p)
    if sub not in SHOWN_SUBS:
        continue
    author = p.get("author","")
    if author not in tftc_authors_set:
        continue
    if not passes(p):
        continue
    entry = {
        "id": p["id"],
        "title": p["title"],
        "score": p["score"],
        "word_count": p.get("word_count") or len((p.get("selftext") or "").split()),
        "url": p["url"],
        "created_utc": p["created_utc"],
        "flair": p.get("flair",""),
    }
    key = "tftc" if sub == "TalesFromTheCreeps" else sub
    author_data[author][key].append(entry)

# Add episode matches
for author, eps in author_to_episodes.items():
    if author in author_data:
        author_data[author]["episodes"] = sorted(eps, key=lambda e: e.get("date",""))

# Sort stories within each author
for d in author_data.values():
    d["tftc"].sort(key=lambda s: -s["score"])
    d["nosleep"].sort(key=lambda s: -s["score"])

# Sort authors: CC episodes first, then most TFTC stories
authors_sorted = sorted(
    author_data.items(),
    key=lambda x: (-len(x[1]["episodes"]), -len(x[1]["tftc"]), -len(x[1]["nosleep"]))
)
result = {a: d for a, d in authors_sorted}

out = os.path.join(DATA, "tftc_authors.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)

total = len(result)
with_eps = sum(1 for d in result.values() if d["episodes"])
with_nosleep = sum(1 for d in result.values() if d["nosleep"])
print(f"tftc_authors.json: {total} authors, {with_eps} with CC eps, {with_nosleep} with nosleep posts")
print(f"File: {os.path.getsize(out)/1e6:.1f}MB")
