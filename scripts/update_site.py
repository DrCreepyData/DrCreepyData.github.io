"""
Daily update: fetch new r/TalesFromTheCreeps posts, regenerate stories.json
and embeddings.bin for DrCreepyData.github.io
"""
import json, os, time, datetime, requests, numpy as np

HEADERS = {"User-Agent": "CreepcastStoryFinder/1.0", "Accept": "application/json"}
URL     = "https://arctic-shift.photon-reddit.com/api/posts/search"
DATA    = os.path.join(os.path.dirname(__file__), "..", "data")
POSTS_F = os.path.join(DATA, "all_posts.json")
DIM     = 384
EXCLUDED = {"Need Help","Offering Help","Writing Help","Story Shoutout",
            "Fan Story Discussion","Mod Announcement","Publishing Announcement",
            "Prompt (MOD APPROVED)"}


def clean_post(p):
    selftext = (p.get("selftext") or "").strip()
    if len(selftext) < 500 or selftext in ("[removed]", "[deleted]"):
        return None
    if (p.get("link_flair_text") or "").strip() in EXCLUDED:
        return None
    permalink = p.get("permalink", "")
    return {
        "id":           p.get("id", ""),
        "title":        p.get("title", ""),
        "selftext":     selftext[:8000],
        "word_count":   len(selftext.split()),   # actual full-text word count
        "score":        p.get("score", 0),
        "url":          f"https://www.reddit.com{permalink}",
        "flair":        (p.get("link_flair_text") or ""),
        "created_utc":  p.get("created_utc", 0),
        "num_comments": p.get("num_comments", 0),
        "author":       p.get("author", ""),
    }


def fetch_newer(after_ts, existing_ids):
    new_posts, before, page = [], None, 0
    while True:
        params = {"subreddit": "TalesFromTheCreeps", "limit": 100, "sort": "desc"}
        if before:
            params["before"] = before
        try:
            resp = requests.get(URL, params=params, headers=HEADERS, timeout=20)
            posts = resp.json().get("data") or []
        except Exception as e:
            print(f"  Error page {page}: {e}"); break
        if not posts:
            break
        done = False
        for p in posts:
            if p.get("created_utc", 0) <= after_ts:
                done = True; break
            if p.get("id") in existing_ids:
                continue
            c = clean_post(p)
            if c:
                new_posts.append(c)
                existing_ids.add(p["id"])
        if done:
            break
        before = posts[-1]["created_utc"]
        page += 1
        time.sleep(0.3)
    print(f"  Fetched {len(new_posts)} new posts")
    return new_posts


def embed_texts(texts, model):
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True).astype(np.float32)


def main():
    os.makedirs(DATA, exist_ok=True)

    # Load or init post cache
    if os.path.exists(POSTS_F):
        with open(POSTS_F) as f:
            all_posts = json.load(f)
        print(f"Loaded {len(all_posts)} existing posts")
    else:
        all_posts = []
        print("Starting fresh post cache")

    existing_ids = {p["id"] for p in all_posts}
    newest_ts    = max((p["created_utc"] for p in all_posts), default=0)

    new_posts = fetch_newer(newest_ts, existing_ids)
    all_posts = all_posts + new_posts
    all_posts.sort(key=lambda p: p["created_utc"])

    with open(POSTS_F, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False)
    print(f"Saved {len(all_posts)} total posts")

    # Build stories.json: all eligible posts, metadata only (no text — loaded on demand)
    selected = [p for p in all_posts if len(p["selftext"]) >= 500]
    print(f"Selected {len(selected)} stories for site")

    def make_story(p):
        return {
            "id":           p["id"],
            "title":        p["title"],
            "author":       p["author"],
            "score":        p["score"],
            "created_utc":  p["created_utc"],
            "url":          p["url"],
            "flair":        p["flair"],
            "num_comments": p["num_comments"],
            "word_count":   p.get("word_count") or len(p["selftext"].split()),
        }

    stories = [make_story(p) for p in selected]
    stories_path = os.path.join(DATA, "stories.json")
    with open(stories_path, "w", encoding="utf-8") as f:
        json.dump(stories, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Saved stories.json: {os.path.getsize(stories_path)/1e6:.1f}MB")

    # Load existing embeddings if available
    emb_path = os.path.join(DATA, "embeddings.bin")
    id_path  = os.path.join(DATA, "embedded_ids.json")

    if os.path.exists(emb_path) and os.path.exists(id_path):
        with open(id_path) as f:
            embedded_ids = json.load(f)
        existing_embs = np.fromfile(emb_path, dtype=np.float32).reshape(len(embedded_ids), DIM)
        id_to_emb = {sid: existing_embs[i] for i, sid in enumerate(embedded_ids)}
        print(f"Loaded {len(embedded_ids)} existing embeddings")
    else:
        id_to_emb, embedded_ids = {}, []
        print("No existing embeddings — embedding all stories")

    # Embed only new stories (use full selftext from all_posts cache for embedding)
    post_by_id = {p["id"]: p for p in all_posts}
    to_embed = [s for s in stories if s["id"] not in id_to_emb]
    if to_embed:
        print(f"Embedding {len(to_embed)} new stories...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        texts = [s["title"] + " " + " ".join((post_by_id[s["id"]]["selftext"]).split()[:512]) for s in to_embed]
        new_embs = embed_texts(texts, model)
        for i, s in enumerate(to_embed):
            id_to_emb[s["id"]] = new_embs[i]
    else:
        print("No new stories to embed")

    # Write embeddings in stories order
    final_embs = np.stack([id_to_emb[s["id"]] for s in stories])
    final_embs.tofile(emb_path)
    with open(id_path, "w") as f:
        json.dump([s["id"] for s in stories], f)
    print(f"Saved embeddings.bin: {final_embs.nbytes/1e6:.1f}MB")

    # Update last_updated
    with open(os.path.join(DATA, "last_updated.json"), "w") as f:
        json.dump({"updated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                   "total_posts": len(all_posts),
                   "stories_on_site": len(stories)}, f)

    print("Done!")


if __name__ == "__main__":
    main()
