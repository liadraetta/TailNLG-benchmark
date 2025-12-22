# extracting_triples.py
import os
import json
import time
import random
import tempfile
import sqlite3
import threading
from copy import deepcopy
from pathlib import Path
from functools import lru_cache
from SPARQLWrapper import SPARQLWrapper, JSON


INPUT_FOLDER = Path("...")
OUTPUT_FOLDER = Path("...")
# Cache in a local, non-synced folder by default. Override with WIKIDATA_CACHE_DIR if you want.
CACHE_FOLDER = Path(os.environ.get("WIKIDATA_CACHE_DIR",
                                   Path(tempfile.gettempdir()) / "wikidata_triples_cache"))
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 6
BASE_SLEEP = 0.5          # base backoff (seconds)
JITTER = (0.05, 0.25)     # jitter for backoff
SLEEP_BETWEEN_QUERIES = 0.0  # usually unnecessary with backoff

LANGS = ["en", "es", "it"]
MAX_ENTITIES_PER_FILE = 10
CONFIGS = [1, 2, 3]
FORCE_REBUILD = bool(int(os.environ.get("FORCE_REBUILD", "0")))  # set 1 to recompute everything

# --- Blacklists (by EN property label) ---
ALWAYS_BLACKLIST = {
    "topic's main category","subclass of","part of","image","coordinate location","model item",
    "Commons category","described by source","category combines topics","instance of",
    "topic has template","has list","on focus list of Wikimedia project",
    "category for maps or plans","category for eponymous categories",
    "category for the view of the item","category for alumni of educational institution",
    "category for the exterior of the item","category for recipients of this award"
}
CONDITIONAL_BLACKLIST = {
    "said to be the same as","related category","different from","sex or gender","part of","given name"
}

# ============ SPARQL client ============
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)
sparql.setMethod("POST")
sparql.addCustomHttpHeader("User-Agent", "https://github.com/liadraetta")

def run_sparql(query: str):
    """Run a SPARQL query with exponential backoff + jitter; return bindings list."""
    for attempt in range(MAX_RETRIES):
        try:
            sparql.setQuery(query)
            results = sparql.query().convert()
            return results.get("results", {}).get("bindings", [])
        except Exception as e:
            sleep_for = (BASE_SLEEP * (2 ** attempt)) + random.uniform(*JITTER)
            print(f"[WDQS retry {attempt+1}/{MAX_RETRIES}] {e}  -> sleeping {sleep_for:.2f}s")
            time.sleep(sleep_for)
    return []

# ============ SQLite-backed KV stores ============
class SQLiteKV:
    def __init__(self, path: Path):
        self.path = str(path)
        self._lock = threading.Lock()
        # timeout helps during brief write contention
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        with self.conn:
            # WAL supports concurrent readers well
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);")

    def get(self, key: str):
        with self._lock:
            cur = self.conn.execute("SELECT v FROM kv WHERE k=?;", (key,))
            row = cur.fetchone()
            return None if row is None else row[0]

    def mget(self, keys: list[str]):
        if not keys:
            return {}
        with self._lock:
            qmarks = ",".join("?" for _ in keys)
            cur = self.conn.execute(f"SELECT k, v FROM kv WHERE k IN ({qmarks});", tuple(keys))
            return {k: v for (k, v) in cur.fetchall()}

    def set(self, key: str, value: str):
        with self._lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?);", (key, value))

    def mset(self, items_dict: dict[str, str]):
        if not items_dict:
            return
        with self._lock, self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?);",
                list(items_dict.items())
            )

    def close(self):
        with self._lock:
            self.conn.close()

edges_kv  = SQLiteKV(CACHE_FOLDER / "edges.sqlite")   # stores JSON-encoded lists
labels_kv = SQLiteKV(CACHE_FOLDER / "labels.sqlite")  # stores plain strings

def _label_key(wd_id: str, lang: str) -> str:
    return f"{wd_id}::{lang}"

# ============ Caching helpers ============
@lru_cache(maxsize=100_000)
def _mem_label_get(wd_id: str, lang: str):
    return labels_kv.get(_label_key(wd_id, lang))

def _mem_label_set(wd_id: str, lang: str, value: str):
    # invalidate only this entry in LRU (simplest: clear)
    _mem_label_get.cache_clear()
    labels_kv.set(_label_key(wd_id, lang), value)

def get_claim_edges_cached(qid: str, limit: int = 10):
    """
    Get direct-claim edges (P, Q) for a QID; cached persistently.
    Returns: list[(pid, value_qid)]
    """
    val = edges_kv.get(qid)
    if val is not None:
        try:
            return json.loads(val)
        except Exception:
            pass  # fall through to refetch if cache was corrupted

    query = f"""
    SELECT ?propEntity ?value WHERE {{
        wd:{qid} ?prop ?value .
        FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/")) .
        FILTER(STRSTARTS(STR(?value), "http://www.wikidata.org/entity/Q")) .
        BIND(IRI(REPLACE(STR(?prop),
            "http://www.wikidata.org/prop/direct/","http://www.wikidata.org/entity/")) AS ?propEntity) .
    }} LIMIT {limit}
    """
    rows = run_sparql(query)
    edges = []
    for r in rows:
        pid = r["propEntity"]["value"].split("/")[-1]
        vq  = r["value"]["value"].split("/")[-1]
        edges.append((pid, vq))

    edges_kv.set(qid, json.dumps(edges, ensure_ascii=False))
    if SLEEP_BETWEEN_QUERIES > 0:
        time.sleep(SLEEP_BETWEEN_QUERIES)
    return edges

def get_labels_batched(ids, lang="en", chunk_size=200):
    """
    Fetch labels for a set/list of Wikidata IDs (Qxxx or Pxxx) in batches.
    Uses SQLite + in-memory caches; only queries missing items.
    Returns: dict[id] = label (fallback to id if not found)
    """
    if not ids:
        return {}
    ids = list(dict.fromkeys(ids))

    # 1) in-memory + SQLite cache
    found = {}
    still_missing = []
    for i in ids:
        v = _mem_label_get(i, lang)
        if v is not None:
            found[i] = v
        else:
            still_missing.append(i)

    # 2) query missing in chunks
    to_store = {}
    for i in range(0, len(still_missing), chunk_size):
        chunk = still_missing[i:i+chunk_size]
        values = " ".join(f"wd:{x}" for x in chunk)
        query = f"""
        SELECT ?id ?label WHERE {{
          VALUES ?id {{ {values} }}
          ?id rdfs:label ?label .
          FILTER(LANG(?label) = "{lang}")
        }}
        """
        rows = run_sparql(query)
        returned = set()
        for r in rows:
            _id = r["id"]["value"].split("/")[-1]
            _lab = r["label"]["value"]
            found[_id] = _lab
            to_store[_label_key(_id, lang)] = _lab
            _mem_label_set(_id, lang, _lab)
            returned.add(_id)
        # Fallback to ID string if no label in that lang
        for miss in chunk:
            if miss not in returned:
                found[miss] = miss
                to_store[_label_key(miss, lang)] = miss
                _mem_label_set(miss, lang, miss)

    # 3) persist
    if to_store:
        labels_kv.mset(to_store)

    return found

# ============ Utilities ============
def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic on Windows & POSIX

def safe_sample(lst, k):
    if not lst:
        return []
    if len(lst) <= k:
        return lst
    return random.sample(lst, k)

def filter_triples_by_blacklist_id(triples_ids):
    """
    triples_ids: list of (s_qid, p_pid, o_qid)
    Use EN property labels for blacklist checks (batched).
    """
    pids = list({p for _, p, _ in triples_ids})
    pid2en = get_labels_batched(pids, lang="en")

    subjects = {s for (s, _, _) in triples_ids}
    out = []
    for (s, p, o) in triples_ids:
        pen = pid2en.get(p, p)
        if pen in ALWAYS_BLACKLIST:
            continue
        if pen in CONDITIONAL_BLACKLIST and o not in subjects:
            continue
        out.append((s, p, o))
    return out

def keep_connected_triples_ids(triples_ids, root_qid):
    adj = {}
    for s, _, o in triples_ids:
        adj.setdefault(s, set()).add(o)
        adj.setdefault(o, set()).add(s)
    visited, stack = set(), [root_qid]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj.get(node, []))
    return [(s, p, o) for (s, p, o) in triples_ids if s in visited and o in visited]

# ============ Build configs once (ID space) ============
def build_config_triples_ids(start_qid, conf):
    """
    Build config once in ID space; enforce relation diversity by PID; cached edges.
    Returns list of (s_qid, p_pid, o_qid)
    """
    a_edges = get_claim_edges_cached(start_qid, limit=10)  # [(P, Q)]
    if len(a_edges) < 2:
        return []

    a_pick = safe_sample(a_edges, 3)
    if len(a_pick) < 2:
        return []

    triples = []

    if conf == 1:
        # a -> b, a -> c, a -> d
        for (p, b_qid) in a_pick:
            triples.append((start_qid, p, b_qid))

    elif conf == 2:
        # a -> b, a -> c, b -> d, b -> e
        (p_ab, b_qid), (p_ac, c_qid) = safe_sample(a_edges, 2)
        b_edges = get_claim_edges_cached(b_qid, limit=10)
        b_samples = safe_sample(b_edges, 2)
        if not b_samples:
            return []
        triples.append((start_qid, p_ab, b_qid))
        triples.append((start_qid, p_ac, c_qid))
        for (p_bd, d_qid) in b_samples:
            triples.append((b_qid, p_bd, d_qid))

    elif conf == 3:
        # a -> b, a -> c, b -> d, c -> e, d -> f, d -> g
        (p_ab, b_qid), (p_ac, c_qid) = safe_sample(a_edges, 2)
        b_edges = get_claim_edges_cached(b_qid, limit=10)
        c_edges = get_claim_edges_cached(c_qid, limit=10)
        d_edges = get_claim_edges_cached(b_qid, limit=10)  # matches your original logic
        b_samples = safe_sample(b_edges, 1)
        c_samples = safe_sample(c_edges, 1)
        d_samples = safe_sample(d_edges, 2)
        triples.append((start_qid, p_ab, b_qid))
        triples.append((start_qid, p_ac, c_qid))
        for (p_bd, d_qid) in b_samples:
            triples.append((b_qid, p_bd, d_qid))
        for (p_ce, e_qid) in c_samples:
            triples.append((c_qid, p_ce, e_qid))
        for (p_df, f_qid) in d_samples:
            triples.append((b_qid, p_df, f_qid))

    # De-dup by (s,p,o) and relation diversity by PID
    seen = set()
    used_rel = set()
    uniq = []
    for (s, p, o) in triples:
        if (s, p, o) in seen:
            continue
        if p in used_rel:
            continue
        seen.add((s, p, o))
        used_rel.add(p)
        uniq.append((s, p, o))

    filtered = filter_triples_by_blacklist_id(uniq)
    connected = keep_connected_triples_ids(filtered, start_qid)
    return connected

# ============ Render (labels) ============
def render_configs(triples_by_conf_ids, langs):
    """
    Render once per entity:
      - Gather all Q and P across ALL configs.
      - Fetch labels in *batches* per language.
      - Return {lang: {conf_k: [(s_lbl,p_lbl,o_lbl), ...]}}
    """
    all_q = set()
    all_p = set()
    for triples in triples_by_conf_ids.values():
        for (s, p, o) in triples:
            all_q.add(s); all_q.add(o); all_p.add(p)
    all_q = list(all_q)
    all_p = list(all_p)

    out = {}
    for lang in langs:
        q_map = get_labels_batched(all_q, lang=lang)
        p_map = get_labels_batched(all_p, lang=lang)
        rendered = {}
        for conf_k, triples in triples_by_conf_ids.items():
            rendered[conf_k] = [(q_map.get(s, s), p_map.get(p, p), q_map.get(o, o)) for (s, p, o) in triples]
        out[lang] = rendered
    return out

# ============ Main (incremental) ============
def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    for input_file in INPUT_FOLDER.glob("*.jsonl"):
        category_name = input_file.stem
        output_file = OUTPUT_FOLDER / f"{category_name}.json"
        print(f"\nProcessing category: {category_name}")

        # Skip whole file if already processed and up-to-date (unless FORCE_REBUILD=1)
        if output_file.exists() and not FORCE_REBUILD:
            try:
                if output_file.stat().st_mtime >= input_file.stat().st_mtime:
                    print(f"  - Up to date → skipping file: {output_file.name}")
                    continue
            except Exception:
                pass  # if stat fails, just process

        # Load existing results to enable per-QID skip
        existing = {}
        if output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        # Load QIDs from input
        entities = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entities.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        qids = [e["title"] for e in entities if "title" in e and isinstance(e["title"], str) and e["title"].startswith("Q")]
        qids = qids[:MAX_ENTITIES_PER_FILE]

        all_triples = deepcopy(existing)

        for idx, qid in enumerate(qids):
            # Per-QID skip if we already have all langs and configs
            if not FORCE_REBUILD and qid in existing:
                have = existing[qid]
                have_langs = all(lang in have for lang in LANGS)
                have_confs = have_langs and all(f"conf_{c}" in have.get("en", {}) for c in CONFIGS)
                if have_langs and have_confs:
                    print(f"[{idx+1}/{len(qids)}] {qid} (cached) – skipping")
                    continue

            print(f"[{idx+1}/{len(qids)}] {qid} (processing)")

            # Build each config ONCE in ID space
            configs_ids = {}
            for conf in CONFIGS:
                triples_ids = build_config_triples_ids(qid, conf)
                if triples_ids:
                    configs_ids[f"conf_{conf}"] = triples_ids

            if not configs_ids:
                # record empty to avoid retrying it on future runs
                all_triples[qid] = all_triples.get(qid, {})
                continue

            # Render equivalently for EN/ES/IT with batched labels
            lang_payload = render_configs(configs_ids, LANGS)
            all_triples[qid] = lang_payload

        # Atomic write
        atomic_write_json(output_file, all_triples)
        print(f"Saved triples to {output_file}")

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            edges_kv.close()
        except Exception:
            pass
        try:
            labels_kv.close()
        except Exception:
            pass