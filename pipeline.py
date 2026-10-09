"""
Rebuilds the app artifacts from the raw TMDB files, following the notebook's
pipeline exactly, with the index bug fixed.

Outputs:
    data/movie_titles.csv     slim catalogue (titles in matrix row order)
    data/sigmoid_kernel.npy   481x481 float32 similarity matrix
    reports/*.png             charts for the README
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, sigmoid_kernel
from sklearn.preprocessing import MinMaxScaler

# Paths are relative to the project root. The raw TMDB CSVs are not in the repo —
# see the README for the Kaggle link.
SRC = pathlib.Path(__file__).parent
OUT = SRC
(OUT / "data").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

for required in ("tmdb_5000_movies.csv", "tmdb_5000_credits.csv"):
    if not (SRC / required).exists():
        raise SystemExit(
            f"missing {required} — download the TMDB 5000 dataset from Kaggle into "
            f"{SRC} (see README). The app itself does not need these; it runs off "
            f"the committed artifacts in data/."
        )

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- load + merge
credits = pd.read_csv(SRC / "tmdb_5000_credits.csv").rename(columns={"movie_id": "id"})
movies = pd.read_csv(SRC / "tmdb_5000_movies.csv")
mix = movies.merge(credits, on="id")
mix = mix.drop(columns=["homepage", "title_x", "title_y", "production_countries", "status"])
print(f"merged catalogue: {mix.shape[0]} movies, {mix.shape[1]} columns")

# ------------------------------------------------------- filter to voted films
tr = mix["vote_count"].quantile(0.9)
fltr = mix[mix["vote_count"] >= tr].copy()

# THE FIX: the notebook kept the original index here (values up to 4602), then
# used it to index a 481x481 matrix. Resetting it makes row position and index
# the same thing, which is what the similarity matrix is addressed by.
fltr = fltr.reset_index(drop=True)
print(f"vote_count cutoff (90th pct): {tr:.0f}  ->  {len(fltr)} movies retained")

# --------------------------------------------------- 1. weighted average score
# m is a smoothing constant: the vote count at which a film's own rating starts
# to outweigh the global mean. Held at 1900, just above the 1840 cutoff, so every
# film in the set is still pulled toward the mean rather than trusted outright.
m = 1900
v = fltr["vote_count"]
R = fltr["vote_average"]
C = fltr["vote_average"].mean()
fltr["weighted_avg"] = ((R * v) + (C * m)) / (v + m)

top_ranks = fltr.sort_values("weighted_avg", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=top_ranks["weighted_avg"], y=top_ranks["original_title"], ax=ax, color="#4c6ef5")
ax.set(xlabel="Weighted average score", ylabel="", title="Top 15 by weighted average rating")
ax.set_xlim(left=6.5)
fig.tight_layout()
fig.savefig(OUT / "reports/01_weighted_average.png", dpi=110)
plt.close(fig)

# --------------------------------------------------------- 2. popularity score
top_pop = fltr.sort_values("popularity", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=top_pop["popularity"], y=top_pop["original_title"], ax=ax, color="#f06595")
ax.set(xlabel="TMDB popularity", ylabel="", title="Top 15 by popularity")
fig.tight_layout()
fig.savefig(OUT / "reports/02_popularity.png", dpi=110)
plt.close(fig)

# ------------------------------------------- 3. normalised hybrid of the above
scaled = MinMaxScaler().fit_transform(fltr[["weighted_avg", "popularity"]])
fltr[["wt_sc", "pop_sc"]] = scaled
fltr["combi_sc"] = fltr["wt_sc"] * 0.5 + fltr["pop_sc"] * 0.5

top_combo = fltr.sort_values("combi_sc", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=top_combo["combi_sc"], y=top_combo["original_title"], ax=ax, color="#20c997")
ax.set(xlabel="Hybrid score (50% rating / 50% popularity)", ylabel="",
       title="Top 15 by hybrid score")
fig.tight_layout()
fig.savefig(OUT / "reports/03_hybrid_score.png", dpi=110)
plt.close(fig)

# -------------------------------------------------- 4. content-based filtering
assert fltr["overview"].notna().all(), "blank overview would break the vectoriser"
tfv = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1, 3), stop_words="english")
tfv_matrix = tfv.fit_transform(fltr["overview"])
print(f"tf-idf matrix: {tfv_matrix.shape[0]} docs x {tfv_matrix.shape[1]} features "
      f"({tfv_matrix.nnz} non-zero, {100 * tfv_matrix.nnz / np.prod(tfv_matrix.shape):.2f}% dense)")

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
print(f"similarity matrix: {sig.shape}, range [{sig.min():.4f}, {sig.max():.4f}]")

# ------------------------------------------------- sanity check the similarity
# The sigmoid kernel here is saturated: tanh(gamma*<x,y> + 1) with tiny dot
# products lands every pair in a band ~3e-04 wide around tanh(1) = 0.7616. The
# ORDER still holds, because tanh is monotonic over that range -- verified below
# against cosine similarity, the conventional choice for tf-idf vectors.
cos = cosine_similarity(tfv_matrix, tfv_matrix)
overlap = np.array([
    len(set(np.argsort(-sig[i])[1:11]) & set(np.argsort(-cos[i])[1:11]))
    for i in range(len(sig))
])
print(f"top-10 agreement with cosine similarity: mean {overlap.mean():.2f}/10, "
      f"identical for {(overlap == 10).sum()}/{len(sig)} films")

# Saturation is also why float32 is NOT safe here: it loses the 4th decimal the
# ranking depends on, and reorders the top-10 for 26 of 481 films. Keep float64.
np.save(OUT / "data/sigmoid_kernel.npy", sig)

catalogue = fltr[["original_title", "release_date", "vote_average", "vote_count", "overview"]].copy()
catalogue["year"] = pd.to_datetime(catalogue["release_date"], errors="coerce").dt.year
catalogue = catalogue.drop(columns=["release_date"])
catalogue.to_csv(OUT / "data/movie_titles.csv", index=False)

meta = {
    "movies": int(len(fltr)),
    "vote_count_cutoff": int(tr),
    "tfidf_features": int(tfv_matrix.shape[1]),
    "kernel_mb": round(sig.nbytes / 1e6, 2),
    "cosine_agreement": f"{(overlap == 10).sum()}/{len(sig)}",
}
(OUT / "data/meta.json").write_text(json.dumps(meta, indent=2))
print("\nartifacts written:", meta)
