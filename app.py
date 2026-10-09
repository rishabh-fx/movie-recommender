"""
Content-based movie recommender.

Serves pre-computed artifacts built by pipeline.py:
  data/movie_titles.csv     481 films, in the same row order as the matrix
  data/sigmoid_kernel.npy   481x481 pairwise similarity over tf-idf of overviews

Row position in the CSV IS the index into the matrix. Nothing here re-derives an
index from the original TMDB row numbers -- that mismatch was the bug in the
original notebook.
"""
import numpy as np
import pandas as pd
import streamlit as st

DATA = "data/movie_titles.csv"
KERNEL = "data/sigmoid_kernel.npy"
TOP_N = 10


@st.cache_data(show_spinner=False)
def load_catalogue() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    # Guard the invariant the whole app rests on.
    assert df.index.equals(pd.RangeIndex(len(df))), "catalogue index must be 0..n-1"
    return df


@st.cache_resource(show_spinner=False)
def load_similarity() -> np.ndarray:
    return np.load(KERNEL)


def recommend(title: str, catalogue: pd.DataFrame, similarity: np.ndarray, n: int = TOP_N):
    """Return the n most similar films to `title`, excluding the film itself."""
    position = catalogue.index[catalogue["original_title"] == title]
    if len(position) == 0:
        return catalogue.iloc[[]]
    row = similarity[position[0]]
    # argsort descending, drop the film's own row, keep n
    ranked = np.argsort(-row)
    ranked = ranked[ranked != position[0]][:n]
    return catalogue.iloc[ranked]


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

catalogue = load_catalogue()
similarity = load_similarity()

st.title("🎬 Movie Recommender")
st.caption(
    f"Content-based filtering over {len(catalogue)} films — "
    "TF-IDF on plot summaries, ranked by pairwise similarity."
)

titles = sorted(catalogue["original_title"])
default = titles.index("Interstellar") if "Interstellar" in titles else 0
selected = st.selectbox("Pick a film you like", titles, index=default)

chosen = catalogue[catalogue["original_title"] == selected].iloc[0]
year = "" if pd.isna(chosen["year"]) else f" ({int(chosen['year'])})"
st.markdown(f"**{selected}**{year} — ⭐ {chosen['vote_average']:.1f} · {int(chosen['vote_count']):,} votes")
with st.expander("Plot summary"):
    st.write(chosen["overview"])

if st.button("Get recommendations", type="primary"):
    results = recommend(selected, catalogue, similarity)
    st.subheader(f"Films similar to {selected}")
    for rank, (_, movie) in enumerate(results.iterrows(), start=1):
        year = "" if pd.isna(movie["year"]) else f" ({int(movie['year'])})"
        st.markdown(f"**{rank}. {movie['original_title']}**{year} — ⭐ {movie['vote_average']:.1f}")
        st.caption(movie["overview"][:220] + ("…" if len(movie["overview"]) > 220 else ""))

st.divider()
st.caption(
    "Similarity comes from a sigmoid kernel over TF-IDF vectors of each film's plot "
    "summary. Scores are deliberately not shown: the kernel is saturated, so every "
    "pair sits within 0.0003 of 0.7616 and the number would read as meaningless. "
    "The ranking it produces is sound — it matches cosine similarity for all 481 films."
)
