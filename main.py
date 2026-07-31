"""
VectorDB Engine — Built from scratch in Python
Implements HNSW, KD-Tree, and Brute Force search algorithms side-by-side,
plus a RAG pipeline powered by a local LLM via Ollama.
"""

import http.server
import json
import math
import random
import re
import socketserver
import sys
import threading
import time
import urllib.parse
import urllib.request
import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

DIMS = 16  # Demo vectors dimension

# =====================================================================
# DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]

# =====================================================================
# DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    s = sum((x - y) ** 2 for x, y in zip(a, b))
    return math.sqrt(s)

def cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - (dot / (math.sqrt(na) * math.sqrt(nb)))

def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

DistFn = Callable[[List[float], List[float]], float]

def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean

# Helper for max-heap using Python's min-heap (heapq)
class MaxHeapItem:
    def __init__(self, dist: float, node_id: int):
        self.dist = dist
        self.node_id = node_id

    def __lt__(self, other: "MaxHeapItem") -> bool:
        # Farthest distance comes first (is considered "smaller" for min-heap top)
        return self.dist > other.dist

# =====================================================================
# BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        r = [(dist(q, v.emb), v.id) for v in self.items]
        r.sort(key=lambda x: x[0])
        return r[:k]

    def remove(self, item_id: int):
        self.items = [v for v in self.items if v.id != item_id]

# =====================================================================
# KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, v: VectorItem):
        self.item: VectorItem = v
        self.left: Optional[KDNode] = None
        self.right: Optional[KDNode] = None

class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _ins(self, n: Optional[KDNode], v: VectorItem, d: int) -> KDNode:
        if n is None:
            return KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < n.item.emb[ax]:
            n.left = self._ins(n.left, v, d + 1)
        else:
            n.right = self._ins(n.right, v, d + 1)
        return n

    def insert(self, v: VectorItem):
        self.root = self._ins(self.root, v, 0)

    def _knn_rec(self, n: Optional[KDNode], q: List[float], k: int, d: int, dist: DistFn, heap: List[MaxHeapItem]):
        if n is None:
            return
        dn = dist(q, n.item.emb)
        if len(heap) < k or dn < heap[0].dist:
            heapq.heappush(heap, MaxHeapItem(dn, n.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax = d % self.dims
        diff = q[ax] - n.item.emb[ax]
        closer = n.left if diff < 0 else n.right
        farther = n.right if diff < 0 else n.left
        
        self._knn_rec(closer, q, k, d + 1, dist, heap)
        if len(heap) < k or abs(diff) < heap[0].dist:
            self._knn_rec(farther, q, k, d + 1, dist, heap)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        heap: List[MaxHeapItem] = []
        self._knn_rec(self.root, q, k, 0, dist, heap)
        res = [(item.dist, item.node_id) for item in heap]
        res.sort(key=lambda x: x[0])
        return res

    def rebuild(self, items: List[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
# HNSW — Hierarchical Navigable Small World
# =====================================================================

@dataclass
class HNSWNode:
    item: VectorItem
    max_lyr: int
    nbrs: List[List[int]]

class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_pt = -1
        self.rng = random.Random(42)
        self.G: Dict[int, HNSWNode] = {}

    def rand_level(self) -> int:
        u = self.rng.random()
        if u == 0:
            u = 1e-7
        return int(math.floor(-math.log(u) * self.mL))

    def search_layer(self, q: List[float], ep: int, ef: int, lyr: int, dist: DistFn) -> List[Tuple[float, int]]:
        vis = {ep: True}
        d0 = dist(q, self.G[ep].item.emb)
        
        # cands: min-heap of (dist, node_id)
        cands = [(d0, ep)]
        # found: max-heap of MaxHeapItem(dist, node_id)
        found = [MaxHeapItem(d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > found[0].dist:
                break
            if lyr >= len(self.G[cid].nbrs):
                continue

            for nid in self.G[cid].nbrs[lyr]:
                if vis.get(nid, False) or nid not in self.G:
                    continue
                vis[nid] = True
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < found[0].dist:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, MaxHeapItem(nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        res = [(item.dist, item.node_id) for item in found]
        res.sort(key=lambda x: x[0])
        return res

    def select_nbrs(self, cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [cid for d, cid in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn):
        item_id = item.id
        lvl = self.rand_level()
        self.G[item_id] = HNSWNode(
            item=item,
            max_lyr=lvl,
            nbrs=[[] for _ in range(lvl + 1)]
        )

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self.search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self.select_nbrs(W, max_m)
            self.G[item_id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                if len(self.G[nid].nbrs) <= lc:
                    self.G[nid].nbrs.extend([[] for _ in range(lc + 1 - len(self.G[nid].nbrs))])
                conn = self.G[nid].nbrs[lc]
                conn.append(item_id)
                if len(conn) > max_m:
                    ds = []
                    for c in conn:
                        if c in self.G:
                            ds.append((dist(self.G[nid].item.emb, self.G[c].item.emb), c))
                    ds.sort(key=lambda x: x[0])
                    self.G[nid].nbrs[lc] = [c for d, c in ds[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item_id

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self.search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, item_id: int):
        if item_id not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if item_id in layer:
                    layer.remove(item_id)
        if self.entry_pt == item_id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != item_id:
                    self.entry_pt = nid
                    break
        del self.G[item_id]

    def get_info(self) -> dict:
        top_layer = self.top_layer
        node_count = len(self.G)
        max_l = max(top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []

        for item_id, nd in self.G.items():
            nodes.append({
                "id": item_id,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if item_id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": item_id, "dst": nid, "lyr": lc})

        return {
            "topLayer": top_layer,
            "nodeCount": node_count,
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges
        }

    def size(self) -> int:
        return len(self.G)

# =====================================================================
# VECTOR DATABASE (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist: DistFn) -> int:
        with self.mu:
            v = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            rem = list(self.store.values())
            self.kdt.rebuild(rem)
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            t1 = time.perf_counter()
            us = int((t1 - t0) * 1_000_000)

            hits = []
            for d, item_id in raw:
                if item_id in self.store:
                    v = self.store[item_id]
                    hits.append({
                        "id": v.id,
                        "metadata": v.metadata,
                        "category": v.category,
                        "embedding": v.emb,
                        "distance": d
                    })
            return {
                "results": hits,
                "latencyUs": us,
                "algo": algo,
                "metric": metric
            }

    def benchmark(self, q: List[float], k: int, metric: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)

            def time_fn(fn):
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
                return int((t1 - t0) * 1_000_000)

            bf_us = time_fn(lambda: self.bf.knn(q, k, dfn))
            kd_us = time_fn(lambda: self.kdt.knn(q, k, dfn))
            hnsw_us = time_fn(lambda: self.hnsw.knn(q, k, 50, dfn))

            return {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
                "itemCount": len(self.store)
            }

    def all(self) -> List[VectorItem]:
        with self.mu:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.mu:
            return len(self.store)

# =====================================================================
# TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    if step <= 0:
        step = 1

    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
    return chunks

# =====================================================================
# OLLAMA CLIENT — wraps local Ollama REST API
# =====================================================================

import os

class OllamaClient:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or os.getenv("OLLAMA_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("OLLAMA_PORT", "11434"))
        self.embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.gen_model = os.getenv("GEN_MODEL", "llama3.2")

    def is_available(self) -> bool:
        url = f"http://{self.host}:{self.port}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        url = f"http://{self.host}:{self.port}/api/embeddings"
        body = json.dumps({"model": self.embed_model, "prompt": text}).encode('utf-8')
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode('utf-8'))
                    return data.get("embedding", [])
        except Exception:
            pass
        return []

    def generate(self, prompt: str) -> str:
        url = f"http://{self.host}:{self.port}/api/generate"
        body = json.dumps({
            "model": self.gen_model,
            "prompt": prompt,
            "stream": False
        }).encode('utf-8')
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode('utf-8'))
                    return data.get("response", "")
        except Exception:
            pass
        return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
# DOCUMENT DATABASE — HNSW over real Ollama embeddings
# =====================================================================

class DocumentDB:
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            doc = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[doc.id] = doc
            vi = VectorItem(doc.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return doc.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self.mu:
            if not self.store:
                return []
            raw = self.bf.knn(q, k, cosine) if len(self.store) < 10 else self.hnsw.knn(q, k, 50, cosine)
            out = []
            for d, doc_id in raw:
                if doc_id in self.store and d <= max_dist:
                    out.append((d, self.store[doc_id]))
            return out

    def remove(self, doc_id: int) -> bool:
        with self.mu:
            if doc_id not in self.store:
                return False
            del self.store[doc_id]
            self.hnsw.remove(doc_id)
            self.bf.remove(doc_id)
            return True

    def all(self) -> List[DocItem]:
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims

# =====================================================================
# DEMO DATA (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB):
    dist = get_dist_fn("cosine")
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    db.insert("Linked List: nodes connected by pointers", "cs",
        [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06], dist)
    db.insert("Binary Search Tree: O(log n) search and insert", "cs",
        [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10], dist)
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
        [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07], dist)
    db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
        [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07], dist)
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
        [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09], dist)
    db.insert("Calculus: derivatives integrals and limits", "math",
        [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10], dist)
    db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
        [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09], dist)
    db.insert("Probability: distributions random variables Bayes theorem", "math",
        [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08], dist)
    db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
        [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06], dist)
    db.insert("Combinatorics: permutations combinations generating functions", "math",
        [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07], dist)
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
        [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07], dist)
    db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
        [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08], dist)
    db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
        [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06], dist)
    db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
        [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09], dist)
    db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
        [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06], dist)
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
        [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72], dist)
    db.insert("Football: tackles touchdowns field goals and strategy", "sports",
        [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76], dist)
    db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
        [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82], dist)
    db.insert("Chess: openings endgames tactics strategic board game", "sports",
        [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90], dist)
    db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
        [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80], dist)

# =====================================================================
# HTTP SERVER
# =====================================================================

class VectorDBRequestHandler(http.server.BaseHTTPRequestHandler):
    db: VectorDB = None
    doc_db: DocumentDB = None
    ollama: OllamaClient = None

    def log_message(self, format, *args):
        pass

    def send_cors(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", content_type)

    def send_json(self, data: Any, status: int = 200):
        self.send_cors(status, "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_cors(204)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            try:
                with open("index.html", "rb") as f:
                    content = f.read()
                self.send_cors(200, "text/html")
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_cors(404, "text/plain")
                self.end_headers()
                self.wfile.write(b"404 Not Found")
            return

        if path == "/search":
            v_str = params.get("v", [""])[0]
            try:
                q = [float(x) for x in v_str.split(",") if x.strip()]
            except ValueError:
                q = []
            if len(q) != DIMS:
                self.send_json({"error": f"need {DIMS}D vector"}, status=400)
                return

            k = int(params.get("k", ["5"])[0])
            metric = params.get("metric", ["cosine"])[0]
            algo = params.get("algo", ["hnsw"])[0]

            out = self.db.search(q, k, metric, algo)
            self.send_json(out)
            return

        if path == "/items":
            items = self.db.all()
            res = [{
                "id": v.id,
                "metadata": v.metadata,
                "category": v.category,
                "embedding": v.emb
            } for v in items]
            self.send_json(res)
            return

        if path == "/benchmark":
            v_str = params.get("v", [""])[0]
            try:
                q = [float(x) for x in v_str.split(",") if x.strip()]
            except ValueError:
                q = []
            if len(q) != DIMS:
                self.send_json({"error": f"need {DIMS}D vector"}, status=400)
                return

            k = int(params.get("k", ["5"])[0])
            metric = params.get("metric", ["cosine"])[0]

            out = self.db.benchmark(q, k, metric)
            self.send_json(out)
            return

        if path == "/hnsw-info":
            info = self.db.hnsw_info()
            self.send_json(info)
            return

        if path == "/doc/list":
            docs = self.doc_db.all()
            res = []
            for d in docs:
                preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
                words = len(d.text.split())
                res.append({
                    "id": d.id,
                    "title": d.title,
                    "preview": preview,
                    "words": words
                })
            self.send_json(res)
            return

        if path == "/status":
            up = self.ollama.is_available()
            self.send_json({
                "ollamaAvailable": up,
                "embedModel": self.ollama.embed_model,
                "genModel": self.ollama.gen_model,
                "docCount": self.doc_db.size(),
                "docDims": self.doc_db.get_dims(),
                "demoDims": DIMS,
                "demoCount": self.db.size()
            })
            return

        if path == "/stats":
            self.send_json({
                "count": self.db.size(),
                "dims": DIMS,
                "algorithms": ["bruteforce", "kdtree", "hnsw"],
                "metrics": ["euclidean", "cosine", "manhattan"]
            })
            return

        self.send_json({"error": "Not Found"}, status=404)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body_bytes = self.rfile.read(content_length)
        try:
            body = json.loads(body_bytes.decode('utf-8'))
        except Exception:
            body = {}

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/insert":
            meta = body.get("metadata", "")
            cat = body.get("category", "")
            emb = body.get("embedding", [])
            if not meta or not emb or len(emb) != DIMS:
                self.send_json({"error": "invalid body"}, status=400)
                return
            item_id = self.db.insert(meta, cat, emb, get_dist_fn("cosine"))
            self.send_json({"id": item_id})
            return

        if path == "/doc/insert":
            title = body.get("title", "")
            text = body.get("text", "")
            if not title or not text:
                self.send_json({"error": "need title and text"}, status=400)
                return

            chunks = chunk_text(text, 250, 30)
            ids = []
            for i, chunk in enumerate(chunks):
                emb = self.ollama.embed(chunk)
                if not emb:
                    self.send_json({
                        "error": "Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2"
                    }, status=400)
                    return
                chunk_title = f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
                ids.append(self.doc_db.insert(chunk_title, chunk, emb))

            self.send_json({
                "ids": ids,
                "chunks": len(chunks),
                "dims": self.doc_db.get_dims()
            })
            return

        if path == "/doc/search":
            question = body.get("question", "")
            k = int(body.get("k", 3))
            if not question:
                self.send_json({"error": "need question"}, status=400)
                return

            q_emb = self.ollama.embed(question)
            if not q_emb:
                self.send_json({"error": "Ollama unavailable"}, status=400)
                return

            hits = self.doc_db.search(q_emb, k)
            contexts = [{
                "id": doc.id,
                "title": doc.title,
                "distance": round(d, 4)
            } for d, doc in hits]
            self.send_json({"contexts": contexts})
            return

        if path == "/doc/ask":
            question = body.get("question", "")
            k = int(body.get("k", 3))
            if not question:
                self.send_json({"error": "need question"}, status=400)
                return

            # Step 1: embed question
            q_emb = self.ollama.embed(question)
            if not q_emb:
                self.send_json({"error": "Ollama unavailable"}, status=400)
                return

            # Step 2: retrieve top-k chunks
            hits = self.doc_db.search(q_emb, k)

            # Step 3: build prompt
            ctx_str = ""
            for i, (d, doc) in enumerate(hits):
                ctx_str += f"[{i+1}] {doc.title}:\n{doc.text}\n\n"

            prompt = (
                "You are a helpful assistant. Answer the user's question directly. "
                "Use the provided context if it contains relevant information. "
                "If it doesn't, just use your own general knowledge. "
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                "Just answer the question naturally.\n\n"
                f"Context:\n{ctx_str}"
                f"Question: {question}\n\n"
                "Answer:"
            )

            # Step 4: generate answer
            answer = self.ollama.generate(prompt)

            # Step 5: return response
            contexts = [{
                "id": doc.id,
                "title": doc.title,
                "text": doc.text,
                "distance": round(d, 4)
            } for d, doc in hits]

            self.send_json({
                "answer": answer,
                "model": self.ollama.gen_model,
                "contexts": contexts,
                "docCount": self.doc_db.size()
            })
            return

        self.send_json({"error": "Not Found"}, status=404)

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        m_del = re.match(r"^/delete/(\d+)$", path)
        if m_del:
            item_id = int(m_del.group(1))
            ok = self.db.remove(item_id)
            self.send_json({"ok": ok})
            return

        m_doc_del = re.match(r"^/doc/delete/(\d+)$", path)
        if m_doc_del:
            doc_id = int(m_doc_del.group(1))
            ok = self.doc_db.remove(doc_id)
            self.send_json({"ok": ok})
            return

        self.send_json({"error": "Not Found"}, status=404)

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

def main():
    db = VectorDB(DIMS)
    doc_db = DocumentDB()
    ollama = OllamaClient()

    load_demo(db)

    VectorDBRequestHandler.db = db
    VectorDBRequestHandler.doc_db = doc_db
    VectorDBRequestHandler.ollama = ollama

    ollama_up = ollama.is_available()
    print("=== AI-Map Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    server = ThreadedHTTPServer(("0.0.0.0", 8080), VectorDBRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down.")
        server.server_close()

if __name__ == "__main__":
    main()
