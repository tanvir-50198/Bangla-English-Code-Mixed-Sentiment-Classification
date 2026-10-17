"""
Bangla-English (Banglish) Code-Mixed Sentiment Classification - Streamlit app
Bag-of-Words vs TF-IDF  x  Naive Bayes / Logistic Regression / SVM

Run:  streamlit run app.py
Data: put Banglish_Dataset.csv (columns: text, sentiment) next to this file,
      or upload it from the sidebar.
"""
import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn deprecation notices

SEED = 42
DEFAULT_CSV = Path(__file__).parent / "Banglish_Dataset.csv"
MODEL_LABELS = {"NaiveBayes": "Naive Bayes", "LogisticRegression": "Logistic Regression", "SVM": "SVM (linear)"}
FEATURE_LABELS = {"BoW": "Bag-of-Words", "TFIDF": "TF-IDF"}

EXAMPLES = {
    "(write your own)": "",
    "Positive-looking": "movie ta darun hoyeche, ekdom moja pailam. Great acting!",
    "Negative-looking": "ekdom baje movie, boring and waste of time. kharap!!!",
    "Neutral-looking": "movie ta thik thak chilo, average story and normal acting",
}

# --------------------------------------------------------------------------
# Preprocessing (same logic as the notebook / R script)
# --------------------------------------------------------------------------
ENGLISH_STOPWORDS = set(
    """
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
she her hers herself it its itself they them their theirs themselves what which who whom
this that these those am is are was were be been being have has had having do does did doing
would should could ought i'm you're he's she's it's we're they're i've you've we've they've
i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll isn't aren't wasn't
weren't hasn't haven't hadn't doesn't don't didn't won't wouldn't shan't shouldn't can't
cannot couldn't mustn't let's that's who's what's here's there's when's where's why's how's
a an the and but if or because as until while of at by for with about against between into
through during before after above below to from up down in out on off over under again
further then once here there when where why how all any both each few more most other some
such no nor not only own same so than too very
""".split()
)
BANGLISH_STOPWORDS = {
    "ta", "er", "e", "ei", "oi", "amar", "ami", "tumi", "apni", "se",
    "amra", "tara", "eta", "oita", "ekta", "hoy", "hoyeche", "hocche",
    "hoise", "korte", "kore", "korlam", "koreche", "hobe", "chilo",
    "hote", "diye", "niye", "jabo", "gelo", "gelam", "asche", "ache",
    "nai", "na", "ekhon", "ekdom", "sotti", "onek", "khub",
}
ALL_STOPWORDS = ENGLISH_STOPWORDS | BANGLISH_STOPWORDS


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return " ".join(text.split())


def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in ALL_STOPWORDS)


def quick_clean(s: str) -> str:
    return " ".join(re.sub(r"[^a-z\s]", " ", s.lower()).split())


@st.cache_data(show_spinner=False)
def prepare(data: bytes) -> pd.DataFrame:
    """Load CSV bytes, drop blanks/duplicates, clean text and remove stopwords."""
    import io

    df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    if not {"text", "sentiment"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'sentiment' columns.")
    df = df[df["text"].notna() & df["sentiment"].notna() & (df["text"].astype(str).str.strip() != "")]
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["word_count"] = df["text"].astype(str).str.count(r"\S+")
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    df["tokens"] = df["clean_text"].map(remove_stopwords)
    return df


@st.cache_data(show_spinner=False)
def make_wordcloud(data: bytes) -> np.ndarray:
    df = prepare(data)
    freq = Counter(w for t in df["text"].astype(str).map(quick_clean) for w in t.split())
    freq = {w: c for w, c in freq.items() if len(w) > 2}
    wc = WordCloud(width=900, height=450, background_color="white", max_words=100,
                   colormap="Dark2", random_state=SEED).generate_from_frequencies(freq)
    return wc.to_array()


# --------------------------------------------------------------------------
# Features: BoW + TF-IDF (mirrors R's tm: min 3 docs, removeSparseTerms 0.995,
# tf = count/doc_length, idf = log2(N/df))
# --------------------------------------------------------------------------
class FeatureBuilder:
    def __init__(self, sparse_cut: float = 0.995):
        self.sparse_cut = sparse_cut

    def fit(self, tokens: pd.Series):
        self.cv = CountVectorizer(token_pattern=r"(?u)\b[a-z]{2,}\b", min_df=3, lowercase=False)
        counts = self.cv.fit_transform(tokens).tocsr().astype(np.float64)
        n = counts.shape[0]
        doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf = np.log2(n / doc_freq)
        self.keep = doc_freq > n * (1 - self.sparse_cut)
        self.vocab = np.array(self.cv.get_feature_names_out())[self.keep]
        return self._weights(counts)

    def transform(self, tokens: pd.Series):
        return self._weights(self.cv.transform(tokens).tocsr().astype(np.float64))

    def _weights(self, counts):
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        doc_len[doc_len == 0] = 1.0
        tfidf = ((sp.diags(1.0 / doc_len) @ counts) @ sp.diags(self.idf)).tocsr()
        return counts[:, self.keep].tocsr(), tfidf[:, self.keep].tocsr()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_model(name: str, fast_lr: bool):
    if name == "NaiveBayes":
        return BernoulliNB(alpha=1.0, binarize=0.0)
    if name == "LogisticRegression":
        Cs, cv = (np.logspace(-3, 1, 6), 3) if fast_lr else (np.logspace(-3, 1, 12), 5)
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegressionCV(Cs=Cs, cv=cv, penalty="l1", solver="saga", scoring="neg_log_loss",
                                 max_iter=300, tol=1e-3, n_jobs=-1, random_state=SEED),
        )
    return SVC(kernel="linear", C=1.0)


def compute_metrics(preds, actual, classes):
    p, r, f1, _ = precision_recall_fscore_support(actual, preds, labels=classes, average="macro", zero_division=0)
    cm = confusion_matrix(actual, preds, labels=classes).T  # rows = Prediction, cols = Reference
    cm_df = pd.DataFrame(cm, index=pd.Index(classes, name="Prediction"), columns=pd.Index(classes, name="Reference"))
    return {"confusion_matrix": cm_df, "accuracy": accuracy_score(actual, preds),
            "precision": p, "recall": r, "f1": f1}


@st.cache_resource(show_spinner=False)
def train_all(data: bytes, test_size: float, fast_lr: bool):
    df = prepare(data)
    builder = FeatureBuilder()
    X_bow_full, X_tfidf_full = builder.fit(df["tokens"])

    keep_rows = (np.asarray(X_bow_full.sum(axis=1)).ravel() > 0) & (np.asarray(X_tfidf_full.sum(axis=1)).ravel() > 0)
    X = {"BoW": X_bow_full[keep_rows], "TFIDF": X_tfidf_full[keep_rows]}
    labels = df.loc[keep_rows, "sentiment"].reset_index(drop=True)
    classes = sorted(labels.unique())

    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=test_size, stratify=labels, random_state=SEED)
    y_train, y_test = labels.iloc[tr].values, labels.iloc[te].values

    models, results = {}, {}
    for feat, Xf in X.items():
        models[feat], results[feat] = {}, {}
        for mname in MODEL_LABELS:
            model = build_model(mname, fast_lr)
            model.fit(Xf[tr], y_train)
            models[feat][mname] = model
            results[feat][mname] = compute_metrics(model.predict(Xf[te]), y_test, classes)

    rows = [
        {"Feature": f, "Model": m, "Accuracy": round(r["accuracy"], 4), "Precision": round(r["precision"], 4),
         "Recall": round(r["recall"], 4), "F1": round(r["f1"], 4)}
        for f, ms in results.items() for m, r in ms.items()
    ]
    return {
        "builder": builder, "models": models, "results": results, "summary": pd.DataFrame(rows),
        "classes": classes, "n_docs": int(keep_rows.sum()), "n_dropped": int((~keep_rows).sum()),
        "n_train": len(tr), "n_test": len(te), "n_features": X["BoW"].shape[1],
        "test_size": test_size, "fast_lr": fast_lr,
    }


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def fig_class_distribution(df):
    order = sorted(df["sentiment"].unique())
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.countplot(data=df, x="sentiment", order=order, hue="sentiment", hue_order=order, legend=False, ax=ax)
    ax.set(title="Class Distribution of Sentiment Labels", xlabel="Sentiment", ylabel="Count")
    fig.tight_layout()
    return fig


def fig_wordcount(df):
    order = sorted(df["sentiment"].unique())
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.histplot(data=df, x="word_count", hue="sentiment", hue_order=order, binwidth=1,
                 multiple="layer", alpha=0.7, ax=ax)
    ax.set(title="Distribution of Text Length (in words)", xlabel="Word Count", ylabel="Frequency")
    fig.tight_layout()
    return fig


def fig_f1_compare(summary):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=summary, x="Model", y="F1", hue="Feature", ax=ax)
    ax.set(title="Macro F1-Score: BoW vs TF-IDF across Models", xlabel="Classifier", ylabel="Macro F1-score")
    fig.tight_layout()
    return fig


def fig_confusion(cm_df, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor="white",
                annot_kws={"size": 13, "weight": "bold"}, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Reference", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    fig.tight_layout()
    return fig


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="Banglish Sentiment Classifier", page_icon="💬", layout="wide")
st.title("💬 Banglish Sentiment Classification")
st.caption("Bangla-English code-mixed text · Bag-of-Words vs TF-IDF · Naive Bayes / Logistic Regression / SVM")

# ---- Sidebar: data + training settings ----
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Banglish_Dataset.csv", type="csv", help="Needs `text` and `sentiment` columns.")
    st.header("Training settings")
    test_pct = st.slider("Test set size (%)", 10, 40, 20, step=5)
    fast_lr = st.checkbox("Fast Logistic Regression", value=False,
                          help="3-fold CV and fewer C values. Quicker, slightly less thorough.")
    train_clicked = st.button("🏋️ Train / retrain models", type="primary")

if uploaded is not None:
    data_bytes = uploaded.getvalue()
elif DEFAULT_CSV.exists():
    data_bytes = DEFAULT_CSV.read_bytes()
else:
    st.info("Upload `Banglish_Dataset.csv` in the sidebar, or place it next to `app.py`.")
    st.stop()

try:
    df = prepare(data_bytes)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the dataset: {e}")
    st.stop()

if train_clicked:
    st.session_state["train_cfg"] = (test_pct / 100, fast_lr)

art = None
if "train_cfg" in st.session_state:
    cfg = st.session_state["train_cfg"]
    with st.spinner("Training 6 models (2 feature sets × 3 classifiers)… the first run can take a few minutes."):
        art = train_all(data_bytes, *cfg)

with st.sidebar:
    if art is None:
        st.info("Press **Train / retrain models** to unlock the evaluation and prediction tabs.")
    else:
        st.success(f"Trained · {art['n_train']:,} train / {art['n_test']:,} test · {art['n_features']} features")

tab_explore, tab_prep, tab_eval, tab_predict, tab_interp = st.tabs(
    ["📊 Explore", "🧹 Preprocessing", "🏆 Evaluation", "🔮 Predict", "📝 Interpretation"]
)

# ---- Explore ----
with tab_explore:
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents (after cleaning)", f"{len(df):,}")
    c2.metric("Classes", df["sentiment"].nunique())
    c3.metric("Avg. words per text", f"{df['word_count'].mean():.1f}")

    left, right = st.columns(2)
    with left:
        show_fig(fig_class_distribution(df))
        dist = df["sentiment"].value_counts().sort_index().rename("Count").to_frame()
        dist["Share (%)"] = (dist["Count"] / dist["Count"].sum() * 100).round(2)
        st.dataframe(dist)
    with right:
        show_fig(fig_wordcount(df))
        st.dataframe(df["word_count"].describe().round(2).rename("word_count").to_frame())

    st.subheader("Word cloud")
    st.image(make_wordcloud(data_bytes))
    st.subheader("Sample rows")
    st.dataframe(df[["text", "sentiment"]].head(50), hide_index=True)

# ---- Preprocessing ----
with tab_prep:
    st.markdown(
        "Cleaning steps for romanised Bangla + English text: strip emojis/non-ASCII, remove URLs, mentions and `#`, "
        "keep letters only, lowercase, collapse 3+ repeated letters, then remove English + Banglish stopwords."
    )
    demo = st.text_input("Try the cleaner", value="Movie ta boringgggg chilo 😡 http://x.com @user #waste")
    d1, d2, d3 = st.columns(3)
    d1.markdown("**Raw**")
    d1.code(demo or " ", language=None)
    cleaned = clean_text(demo)
    d2.markdown("**Cleaned**")
    d2.code(cleaned or " ", language=None)
    d3.markdown("**After stopword removal**")
    d3.code(remove_stopwords(cleaned) or " ", language=None)

    st.subheader("Raw vs cleaned (dataset sample)")
    st.dataframe(df[["text", "clean_text", "tokens"]].head(20), hide_index=True)
    with st.expander("Banglish stopword list"):
        st.write(", ".join(sorted(BANGLISH_STOPWORDS)))

# ---- Evaluation ----
with tab_eval:
    if art is None:
        st.info("Train the models from the sidebar to see results.")
    else:
        s = art["summary"].copy()
        s["Feature"] = s["Feature"].map(FEATURE_LABELS)
        s["Model"] = s["Model"].map(MODEL_LABELS)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Documents used", f"{art['n_docs']:,}", help=f"{art['n_dropped']} all-zero rows dropped")
        m2.metric("Train / Test", f"{art['n_train']:,} / {art['n_test']:,}")
        m3.metric("Vocabulary size", art["n_features"])
        best = s.loc[s["F1"].idxmax()]
        m4.metric("Best macro F1", f"{best['F1']:.4f}", f"{best['Feature']} + {best['Model']}", delta_color="off")

        st.subheader("Performance summary")
        st.dataframe(s, hide_index=True)
        st.download_button("Download CSV", art["summary"].to_csv(index=False).encode(),
                           file_name="model_performance_summary.csv", mime="text/csv")

        c1, c2 = st.columns([3, 2])
        with c1:
            show_fig(fig_f1_compare(art["summary"].assign(
                Feature=art["summary"]["Feature"].map(FEATURE_LABELS),
                Model=art["summary"]["Model"].map(MODEL_LABELS))))
        with c2:
            st.markdown("**Average by feature representation**")
            avg = (art["summary"].groupby("Feature").agg(Avg_Accuracy=("Accuracy", "mean"), Avg_F1=("F1", "mean"))
                   .round(4).reset_index())
            avg["Feature"] = avg["Feature"].map(FEATURE_LABELS)
            st.dataframe(avg, hide_index=True)

        st.subheader("Confusion matrices")
        cc1, cc2 = st.columns(2)
        feat_sel = cc1.selectbox("Feature representation", list(FEATURE_LABELS), format_func=FEATURE_LABELS.get,
                                 index=1, key="cm_feat")
        model_sel = cc2.selectbox("Classifier", list(MODEL_LABELS), format_func=MODEL_LABELS.get,
                                  index=2, key="cm_model")
        cm_df = art["results"][feat_sel][model_sel]["confusion_matrix"]
        g1, g2 = st.columns([1, 1])
        with g1:
            show_fig(fig_confusion(cm_df, f"{FEATURE_LABELS[feat_sel]} + {MODEL_LABELS[model_sel]}"))
        with g2:
            r = art["results"][feat_sel][model_sel]
            st.dataframe(cm_df)
            st.write(f"Accuracy **{r['accuracy']:.4f}** · Precision **{r['precision']:.4f}** · "
                     f"Recall **{r['recall']:.4f}** · F1 **{r['f1']:.4f}**")
            st.caption("Rows = predicted class, columns = true class (caret layout).")

# ---- Predict ----
with tab_predict:
    if art is None:
        st.info("Train the models from the sidebar to try predictions.")
    else:
        ex = st.selectbox("Try an example", list(EXAMPLES))
        text = st.text_area("Enter Banglish / English text", value=EXAMPLES[ex], key=f"txt_{ex}", height=100,
                            placeholder="e.g. movie ta darun hoyeche, moja pailam")
        p1, p2 = st.columns(2)
        pf = p1.selectbox("Feature representation", list(FEATURE_LABELS), format_func=FEATURE_LABELS.get,
                          index=1, key="pred_feat")
        pm = p2.selectbox("Classifier", list(MODEL_LABELS), format_func=MODEL_LABELS.get, index=2, key="pred_model")

        if text.strip():
            cleaned = clean_text(text)
            tokens = remove_stopwords(cleaned)
            builder = art["builder"]
            Xb, Xt = builder.transform(pd.Series([tokens]))
            Xs = {"BoW": Xb, "TFIDF": Xt}

            known = [w for w in tokens.split() if w in set(builder.vocab)]
            st.markdown(f"**Cleaned:** `{cleaned or '—'}`  \n**Tokens after stopwords:** `{tokens or '—'}`  \n"
                        f"**Words the model knows:** `{', '.join(known) if known else 'none'}`")

            if not known:
                st.warning("None of the words are in the model's vocabulary, so a prediction would be meaningless. "
                           "Try different words.")
            else:
                model = art["models"][pf][pm]
                X = Xs[pf]
                pred = model.predict(X)[0]
                st.success(f"Prediction: **{pred.upper()}**  ({FEATURE_LABELS[pf]} + {MODEL_LABELS[pm]})")

                if hasattr(model, "predict_proba"):
                    probs = pd.Series(model.predict_proba(X)[0], index=model.classes_, name="Probability")
                    st.bar_chart(probs)
                else:
                    st.caption("The linear SVM here does not produce probabilities.")

                if pm == "LogisticRegression":
                    scaler = model.named_steps["standardscaler"]
                    lr = model.named_steps["logisticregressioncv"]
                    xd = X.toarray().ravel() / scaler.scale_
                    ci = list(lr.classes_).index(pred)
                    nz = np.nonzero(xd)[0]
                    contrib = pd.Series(lr.coef_[ci][nz] * xd[nz], index=builder.vocab[nz]).sort_values()
                    st.markdown(f"**Word contributions toward `{pred}`** (positive = pushes toward this class)")
                    st.bar_chart(contrib)

                st.subheader("All six models")
                rows = []
                for f in FEATURE_LABELS:
                    for m in MODEL_LABELS:
                        mod = art["models"][f][m]
                        row = {"Feature": FEATURE_LABELS[f], "Model": MODEL_LABELS[m], "Prediction": mod.predict(Xs[f])[0]}
                        if hasattr(mod, "predict_proba"):
                            row["Confidence"] = f"{mod.predict_proba(Xs[f]).max():.1%}"
                        else:
                            row["Confidence"] = "n/a"
                        rows.append(row)
                st.dataframe(pd.DataFrame(rows), hide_index=True)

# ---- Interpretation ----
with tab_interp:
    if art is None:
        st.info("Train the models from the sidebar to see the interpretation.")
    else:
        summ = art["summary"]
        avg = summ.groupby("Feature")["F1"].mean()
        better = FEATURE_LABELS[avg.idxmax()]
        b = summ.loc[summ["F1"].idxmax()]
        st.markdown(
            f"On this Bangla-English code-mixed sentiment dataset (**{art['n_docs']:,} documents**, "
            f"**{len(art['classes'])} classes**: {', '.join(art['classes'])}), **{better}** achieved the higher average "
            f"macro F1 across the three classifiers. The best single configuration was "
            f"**{FEATURE_LABELS[b['Feature']]} + {MODEL_LABELS[b['Model']]}** with accuracy "
            f"**{b['Accuracy']:.4f}** and macro F1 **{b['F1']:.4f}**."
        )
        st.markdown(
            """
**Interpretation notes**
- TF-IDF down-weights words that appear in many documents, which usually helps linear models such as Logistic Regression and SVM.
- Bag-of-Words can stay competitive for Naive Bayes, whose likelihood is tied to term presence/frequency.
- Romanised Bangla is noisy (spelling variation, elongated letters, transliteration, emojis). The cleaning step and Banglish stopword list aim to reduce that noise.
- Confusion matrices show where models struggle, typically between *neutral* and *positive/negative*.

**Next steps:** character n-gram features, fastText embeddings trained on code-mixed text, or fine-tuned multilingual transformers (mBERT / XLM-R).
"""
        )
