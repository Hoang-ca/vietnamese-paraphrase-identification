"""
Data preprocessing pipeline for Vietnamese Paraphrase Identification.

Steps:
  1. Load four public datasets (VNPC, vnPara, ViSP, ViQP)
  2. Normalize text  +  VnCoreNLP word segmentation (for PhoBERT)
  3. Deduplicate with canonical pair ordering
  4. Union-Find grouping to prevent leakage in splits
  5. StratifiedGroupKFold → train / val / test
  6. TF-IDF hard-negative mining
  7. Save splits as Parquet

Usage:
    python -m src.preprocess          # from repo root
    python src/preprocess.py          # or directly
"""

import os, json, re, unicodedata, random, gc, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from datasets import load_dataset

from src.config import (
    SEED, MODEL_NAME, USE_WORD_SEG,
    VISP_SAMPLE_SIZE, VIQP_ENABLED,
    DO_HARD_NEG_MINING, HARD_NEG_PER_SENT, HARD_NEG_WEIGHT,
    PREP_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)
random.seed(SEED); np.random.seed(SEED)


# ─── Word Segmentation ──────────────────────────────────────────
def _init_segmenter():
    """Initialize VnCoreNLP word segmenter if PhoBERT backbone is used."""
    if not USE_WORD_SEG:
        print("Word segmentation: disabled (XLM-R mode)")
        return None
    import py_vncorenlp
    wseg_dir = os.getenv("VNPI_WSEG_DIR", "./vncorenlp")
    os.makedirs(wseg_dir, exist_ok=True)
    py_vncorenlp.download_model(save_dir=wseg_dir)
    seg = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=wseg_dir)
    print("VnCoreNLP word segmenter loaded ✓")
    return seg


_segmenter = None


def norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def word_segment(text: str) -> str:
    global _segmenter
    if _segmenter is None:
        return text
    if not text.strip():
        return text
    try:
        return " ".join(_segmenter.word_segment(text))
    except Exception:
        return text


def process_text(s: str) -> str:
    s = norm_text(s)
    return word_segment(s) if USE_WORD_SEG else s


def canonical_pair(a: str, b: str):
    return (a, b) if a <= b else (b, a)


# ─── Dataset Loading ────────────────────────────────────────────
def _load_vnpc_vnpara() -> pd.DataFrame:
    print("Loading VNPC + vnPara...")
    ds_vnpc = load_dataset("chienpham/vnpc-train", split="train")
    ds_vnpara = load_dataset("chienpham/vnpara", split="train")

    df_vnpc = ds_vnpc.to_pandas(); df_vnpc["source"] = "vnpc"
    df_vnpara = ds_vnpara.to_pandas(); df_vnpara["source"] = "vnpara"

    if "score" in df_vnpara.columns and "label" not in df_vnpara.columns:
        df_vnpara = df_vnpara.rename(columns={"score": "label"})
    if "Unnamed: 0" in df_vnpara.columns:
        df_vnpara = df_vnpara.drop(columns=["Unnamed: 0"])

    print(f"  vnpc:   {len(df_vnpc)}, labels={df_vnpc['label'].value_counts().to_dict()}")
    print(f"  vnpara: {len(df_vnpara)}, labels={df_vnpara['label'].value_counts().to_dict()}")
    return pd.concat([df_vnpc, df_vnpara], ignore_index=True)


def _load_visp() -> pd.DataFrame:
    print(f"\nLoading ViSP (sampling {VISP_SAMPLE_SIZE})...")
    visp_dir = os.getenv("VNPI_VISP_DIR", "./data/ViSP")
    if not os.path.exists(visp_dir):
        os.system(f"git clone --depth 1 https://github.com/ngwgsang/ViSP.git {visp_dir}")

    visp_files = []
    for root, _, files in os.walk(visp_dir):
        visp_files.extend(
            os.path.join(root, f) for f in files
            if f.endswith((".json", ".jsonl"))
        )
    print(f"  Found: {[os.path.basename(f) for f in visp_files]}")

    visp_pairs = []
    for fpath in visp_files:
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                content = fp.read().strip()
                items = (
                    json.loads(content) if content.startswith("[")
                    else [json.loads(l) for l in content.split("\n") if l.strip()]
                )
            for item in items:
                orig = item.get("original", "")
                for p in item.get("paraphrases", []):
                    pt = p.get("text", p) if isinstance(p, dict) else str(p)
                    if orig and pt:
                        visp_pairs.append({"sentence1": orig, "sentence2": pt,
                                           "label": 1, "source": "visp"})
        except Exception as e:
            print(f"  Warning: {os.path.basename(fpath)}: {e}")

    print(f"  Total ViSP pairs: {len(visp_pairs)}")
    if len(visp_pairs) > VISP_SAMPLE_SIZE:
        random.shuffle(visp_pairs)
        visp_pairs = visp_pairs[:VISP_SAMPLE_SIZE]
        print(f"  Sampled to: {len(visp_pairs)}")

    return pd.DataFrame(visp_pairs) if visp_pairs else pd.DataFrame(
        columns=["sentence1", "sentence2", "label", "source"])


def _load_viqp() -> pd.DataFrame:
    if not VIQP_ENABLED:
        return pd.DataFrame(columns=["sentence1", "sentence2", "label", "source"])
    print("\nLoading ViQP...")
    try:
        ds_viqp = load_dataset("SCM-LAB/ViQP")
        viqp_pairs = []
        for split_name in ds_viqp:
            for item in ds_viqp[split_name]:
                sq = item.get("source", "")
                targets = item.get("target", [])
                if isinstance(targets, str):
                    targets = [targets]
                for t in targets:
                    if sq and t:
                        viqp_pairs.append({"sentence1": sq, "sentence2": t,
                                           "label": 1, "source": "viqp"})
        df_viqp = pd.DataFrame(viqp_pairs)
        print(f"  ViQP pairs: {len(df_viqp)}")
        return df_viqp
    except Exception as e:
        print(f"  ViQP failed: {e}")
        return pd.DataFrame(columns=["sentence1", "sentence2", "label", "source"])


# ─── Dedup + Union-Find ─────────────────────────────────────────
def _dedup_and_group(df: pd.DataFrame) -> pd.DataFrame:
    print("Dedup...")
    pairs = df.apply(lambda r: canonical_pair(r["sentence1"], r["sentence2"]),
                     axis=1, result_type="expand")
    df["a"], df["b"] = pairs[0], pairs[1]
    df["pair_key"] = df["a"] + "\u0001" + df["b"]

    nuniq = df.groupby("pair_key")["label"].nunique()
    confl = set(nuniq[nuniq > 1].index)
    if confl:
        print(f"  Removing {len(confl)} conflicts")
        df = df[~df["pair_key"].isin(confl)].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["pair_key", "label"]).reset_index(drop=True)
    print(f"  {before} → {len(df)}")

    # Union-Find for connected components
    print("Building connected components...")
    all_s = pd.concat([df["a"], df["b"]]).unique()
    sid = {s: i for i, s in enumerate(all_s)}
    parent = list(range(len(all_s)))
    uf_rank = [0] * len(all_s)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if uf_rank[rx] < uf_rank[ry]:
            parent[rx] = ry
        elif uf_rank[rx] > uf_rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            uf_rank[rx] += 1

    for a, b in zip(df["a"], df["b"]):
        union(sid[a], sid[b])

    comp = {s: find(i) for s, i in sid.items()}
    df["group"] = df["a"].map(comp).astype(int)
    print(f"  Components: {df['group'].nunique()}")
    return df


# ─── Splitting ──────────────────────────────────────────────────
def _split(df: pd.DataFrame):
    print("Split train/val/test...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tv_i, ts_i = list(sgkf.split(df, df["label"], df["group"]))[0]
    df_trainval = df.iloc[tv_i].reset_index(drop=True)
    df_test = df.iloc[ts_i].reset_index(drop=True)

    sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    tr_i, vl_i = list(sgkf2.split(df_trainval, df_trainval["label"],
                                   df_trainval["group"]))[0]
    df_train = df_trainval.iloc[tr_i].reset_index(drop=True)
    df_val = df_trainval.iloc[vl_i].reset_index(drop=True)

    for nm, dx in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"  [{nm}] n={len(dx)}, labels={dx['label'].value_counts().to_dict()}")
    return df_train, df_val, df_test


# ─── Hard-Negative Mining ───────────────────────────────────────
def _mine_hard_negatives(df_train: pd.DataFrame) -> pd.DataFrame:
    df_train = df_train.copy()
    df_train["sample_weight"] = 1.0

    if not DO_HARD_NEG_MINING:
        return df_train

    print("\nHard-neg mining...")
    known = set(df_train["pair_key"])
    pos_k = set(df_train.loc[df_train["label"] == 1, "pair_key"])
    slist = pd.concat([df_train["a"], df_train["b"]]).unique().tolist()
    MAX_MINE = 10_000
    msents = random.sample(slist, min(MAX_MINE, len(slist)))
    print(f"  Mining from {len(msents)} of {len(slist)} sents")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                          sublinear_tf=True)
    X = vec.fit_transform(slist)
    s2i = {s: i for i, s in enumerate(slist)}

    nn = NearestNeighbors(n_neighbors=min(20, len(slist)), metric="cosine",
                          algorithm="brute", n_jobs=-1)
    nn.fit(X)
    _, idxs = nn.kneighbors(X[[s2i[s] for s in msents]])

    mined = []
    used = set(known)
    for qi, s in enumerate(msents):
        found = 0
        for j in idxs[qi]:
            if j == s2i[s]:
                continue
            t = slist[j]
            a, b = canonical_pair(s, t)
            key = a + "\u0001" + b
            if key in used or key in pos_k:
                continue
            mined.append({"sentence1": a, "sentence2": b, "label": 0,
                          "source": "mined", "sample_weight": HARD_NEG_WEIGHT})
            used.add(key)
            found += 1
            if found >= HARD_NEG_PER_SENT:
                break

    print(f"  Mined {len(mined)} hard negatives")
    if mined:
        keep = ["sentence1", "sentence2", "label", "source", "sample_weight"]
        df_train = pd.concat([df_train[keep], pd.DataFrame(mined)[keep]],
                             ignore_index=True)
    return df_train


# ─── Main ───────────────────────────────────────────────────────
def main():
    global _segmenter
    _segmenter = _init_segmenter()

    # Load all datasets
    df_core = _load_vnpc_vnpara()
    df_visp = _load_visp()
    df_viqp = _load_viqp()

    # Merge
    all_dfs = [
        dfx[["sentence1", "sentence2", "label", "source"]]
        for dfx in [df_core, df_visp, df_viqp]
        if len(dfx) > 0
    ]
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nMerged: {len(df)} rows")

    # Label cleanup
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    bad = df["label"].isna() | (~df["label"].isin([0, 1]))
    if bad.sum() > 0:
        print(f"Removing {bad.sum()} bad labels")
        df = df[~bad].copy()
    df["label"] = df["label"].astype(int)
    print(f"Labels: {df['label'].value_counts().to_dict()}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")

    # Process text
    print("\nProcessing text (norm + word-seg)...")
    df["sentence1"] = df["sentence1"].map(process_text)
    df["sentence2"] = df["sentence2"].map(process_text)
    empty = (df["sentence1"].str.len() == 0) | (df["sentence2"].str.len() == 0)
    if empty.any():
        df = df[~empty].copy()

    # Dedup + group
    df = _dedup_and_group(df)

    # Split
    df_train, df_val, df_test = _split(df)

    # Hard-neg mining
    df_train = _mine_hard_negatives(df_train)

    # Finalize columns
    keep = ["sentence1", "sentence2", "label", "source", "sample_weight"]
    df_train = df_train[keep].copy()
    df_val = df_val[["sentence1", "sentence2", "label", "source"]].copy()
    df_val["sample_weight"] = 1.0
    df_test = df_test[["sentence1", "sentence2", "label", "source"]].copy()
    df_test["sample_weight"] = 1.0

    print(f"\nFinal: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    print(f"Train labels: {df_train['label'].value_counts().to_dict()}")

    # Save
    df_train.to_parquet(os.path.join(PREP_DIR, "train.parquet"), index=False)
    df_val.to_parquet(os.path.join(PREP_DIR, "val.parquet"), index=False)
    df_test.to_parquet(os.path.join(PREP_DIR, "test.parquet"), index=False)

    del df, df_core, df_visp, df_viqp
    gc.collect()
    print("Data done ✓")


if __name__ == "__main__":
    main()
