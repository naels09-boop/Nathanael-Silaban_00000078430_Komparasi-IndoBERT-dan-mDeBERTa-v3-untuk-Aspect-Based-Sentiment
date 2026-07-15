# ============================================================
# STREAMLIT DEPLOYMENT — ABSA SENTIMENT GOJEK
# Sumber data hanya Google Play Store
# ABSA Positive/Negative/Neutral + Summary + SNA Per Aspek
# ============================================================

import os
import re
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Google Play Scraper
try:
    from google_play_scraper import reviews, Sort
    GOOGLE_PLAY_AVAILABLE = True
except Exception:
    GOOGLE_PLAY_AVAILABLE = False


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ABSA Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Deployment ABSA Sentiment Analysis")
st.caption(
    "Analisis sentimen berbasis aspek pada ulasan aplikasi Gojek "
    "yang bersumber dari Google Play Store."
)


# ============================================================
# 2. GLOBAL CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Asus\Documents\Project Skripsi\data_scraping_gojek_50000"
LABEL_DIR = os.path.join(BASE_DIR, "labeling_local_llm_50000")
OUTPUT_DIR = os.path.join(LABEL_DIR, "hasil_modeling_final")

MODEL_DIR_CANDIDATES = [
    os.path.join(OUTPUT_DIR, "best_indobert_absa_model_full"),
    os.path.join(OUTPUT_DIR, "best_indobert_absa_model"),
    os.path.join(OUTPUT_DIR, "best_indobert_model_full"),
    os.path.join(OUTPUT_DIR, "best_indobert_model"),
    os.path.join(OUTPUT_DIR, "indobert_absa_model_full"),
    os.path.join(OUTPUT_DIR, "indobert_absa_model"),

    # Jika nanti dipindahkan ke folder deployment/VPS/Streamlit Cloud,
    # letakkan folder model di salah satu path berikut:
    "model/best_indobert_absa_model",
    "model/best_indobert_absa_model_full",
    "best_indobert_absa_model",
    "best_indobert_absa_model_full"
]

ASPECTS = [
    "Aplikasi",
    "Layanan",
    "Driver",
    "Tarif & Harga",
    "GoPay & Pembayaran",
    "Promo & Iklan"
]

LABEL_MAP = {
    0: "negatif",
    1: "netral",
    2: "positif"
}

LABEL_ORDER = ["negatif", "netral", "positif"]


# ============================================================
# 3. HELPER MODEL
# ============================================================

def find_model_dir():
    for path in MODEL_DIR_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource
def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_one(text, aspect, tokenizer, model, device, max_length=128):
    model_input = f"aspek: {aspect} | teks: {text}"

    inputs = tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = LABEL_MAP.get(pred_id, str(pred_id))

    return {
        "sentiment": pred_label,
        "prob_negatif": float(probs[0]),
        "prob_netral": float(probs[1]),
        "prob_positif": float(probs[2])
    }


def run_absa(df, text_col, tokenizer, model, device, max_rows=None):
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str).apply(clean_text)
    df = df[df[text_col].str.strip() != ""].reset_index(drop=True)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    results = []

    progress = st.progress(0)
    total_steps = len(df) * len(ASPECTS)
    step = 0

    for idx, row in df.iterrows():
        text = row[text_col]

        source_id = row.get("source_id", idx)
        created_at = row.get("created_at", "")
        source = row.get("source", "Google Play Store")

        for aspect in ASPECTS:
            pred = predict_one(
                text=text,
                aspect=aspect,
                tokenizer=tokenizer,
                model=model,
                device=device
            )

            results.append({
                "source": source,
                "source_id": source_id,
                "created_at": created_at,
                "review_text": text,
                "aspek": aspect,
                "sentiment": pred["sentiment"],
                "prob_negatif": pred["prob_negatif"],
                "prob_netral": pred["prob_netral"],
                "prob_positif": pred["prob_positif"]
            })

            step += 1
            progress.progress(min(step / total_steps, 1.0))

    progress.empty()
    return pd.DataFrame(results)


# ============================================================
# 4. HELPER SNA PER ASPEK
# ============================================================

STOPWORDS_SNA = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "atau",
    "karena", "jadi", "juga", "saya", "aku", "gue", "gw", "nya", "aja", "sih",
    "dong", "deh", "kok", "mah", "lah", "pun", "para", "pada", "dalam", "akan",
    "sudah", "belum", "masih", "lebih", "kurang", "sangat", "banget", "bgt",
    "ga", "gak", "nggak", "tidak", "tak", "bukan", "ada", "adalah", "kalau",
    "kalo", "kan", "ya", "nih", "tuh", "the", "is", "are", "a", "an", "to",
    "for", "of", "in", "on", "at", "and", "or", "aplikasi", "gojek", "gojeknya",
    "sama", "mau", "lagi", "bisa", "buat", "pakai", "pake", "terus", "jadi"
}


def clean_token_for_sna(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        token for token in tokens
        if token not in STOPWORDS_SNA and len(token) > 2
    ]

    return tokens


def build_sna_aspect(df_pred, selected_aspect, selected_sentiments=None, top_n_words=15, top_n_edges=20):
    if selected_sentiments is None:
        selected_sentiments = ["positif", "negatif"]

    df_aspect = df_pred[
        (df_pred["aspek"] == selected_aspect) &
        (df_pred["sentiment"].isin(selected_sentiments))
    ].copy()

    if df_aspect.empty:
        return pd.DataFrame(), pd.DataFrame(), nx.Graph(), df_aspect

    all_tokens = []
    edge_counter = Counter()

    for text in df_aspect["review_text"].dropna().astype(str):
        tokens = clean_token_for_sna(text)
        unique_tokens = list(dict.fromkeys(tokens))

        all_tokens.extend(unique_tokens)

        for word1, word2 in itertools.combinations(sorted(unique_tokens), 2):
            if word1 != word2:
                edge_counter[(word1, word2)] += 1

    word_counter = Counter(all_tokens)

    df_top_words = pd.DataFrame(
        word_counter.most_common(top_n_words),
        columns=["kata", "frekuensi"]
    )

    df_edges = pd.DataFrame(
        [
            {"kata_1": pair[0], "kata_2": pair[1], "bobot": weight}
            for pair, weight in edge_counter.most_common(top_n_edges)
        ]
    )

    G = nx.Graph()

    if not df_edges.empty:
        for _, row in df_edges.iterrows():
            G.add_edge(row["kata_1"], row["kata_2"], weight=row["bobot"])

    return df_top_words, df_edges, G, df_aspect


def plot_sna_network(G, title):
    if G.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(G, seed=42, k=0.9)

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.8, color="#999999"),
        hoverinfo="none",
        mode="lines"
    )

    degree_dict = dict(G.degree())
    max_degree = max(degree_dict.values()) if degree_dict else 1

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        degree = degree_dict.get(node, 1)

        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Degree: {degree}")
        node_size.append(18 + (degree / max_degree) * 35)
        node_color.append(degree)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title=dict(text="Degree"),
                xanchor="left"
            ),
            line=dict(width=1, color="white")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=title,
                font=dict(size=18)
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=650
        )
    )

    return fig


def get_dynamic_sna_context(df_pred):
    dynamic_sna_context = {}

    for aspect in ASPECTS:
        df_words_tmp, df_edges_tmp, _, df_sna_tmp = build_sna_aspect(
            df_pred=df_pred,
            selected_aspect=aspect,
            selected_sentiments=["positif", "negatif"],
            top_n_words=10,
            top_n_edges=10
        )

        if not df_words_tmp.empty:
            kata_sna = ", ".join(df_words_tmp["kata"].head(5).tolist())
        else:
            kata_sna = "-"

        if not df_edges_tmp.empty:
            edge_sna = ", ".join(
                [
                    f"{row['kata_1']}–{row['kata_2']}"
                    for _, row in df_edges_tmp.head(3).iterrows()
                ]
            )
        else:
            edge_sna = "-"

        dynamic_sna_context[aspect] = {
            "kata_sna": kata_sna,
            "edge_sna": edge_sna,
            "jumlah_data_sna": int(len(df_sna_tmp)),
            "temuan_sna": (
                f"Kata dominan pada aspek {aspect} adalah {kata_sna}. "
                f"Hubungan kata yang menonjol adalah {edge_sna}. "
                f"Hasil ini menunjukkan konteks kata yang membentuk opini pengguna pada aspek {aspect}."
            )
        }

    return dynamic_sna_context


# ============================================================
# 5. HELPER KESIMPULAN ABSA
# ============================================================

def build_interpretative_conclusion(aspect, dominant_sentiment, dominant_pct, counts):
    negatif = counts.get("negatif", 0)
    netral = counts.get("netral", 0)

    if aspect == "Aplikasi":
        if dominant_sentiment == "positif":
            return (
                f"Aspek Aplikasi didominasi sentimen positif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa pengalaman penggunaan aplikasi cenderung diterima dengan baik. "
                f"Namun, masih terdapat {negatif} prediksi negatif dan {netral} prediksi netral, "
                f"sehingga kendala teknis atau proses pemesanan tetap perlu diperhatikan."
            )
        if dominant_sentiment == "negatif":
            return (
                f"Aspek Aplikasi didominasi sentimen negatif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa pengalaman teknis penggunaan aplikasi masih menjadi perhatian, "
                f"terutama terkait kendala akses, respons aplikasi, atau proses pemesanan."
            )
        return (
            f"Aspek Aplikasi didominasi sentimen netral sebesar {dominant_pct}%. "
            f"Hasil ini menunjukkan bahwa ulasan pada aspek aplikasi cenderung informatif "
            f"atau belum menunjukkan opini yang kuat."
        )

    if aspect == "Layanan":
        if dominant_sentiment == "positif":
            return (
                f"Aspek Layanan didominasi sentimen positif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa layanan umum Gojek cenderung memberikan pengalaman yang baik. "
                f"Namun, masih terdapat {negatif} prediksi negatif dan {netral} prediksi netral, "
                f"sehingga pengalaman layanan tetap perlu dipantau."
            )
        if dominant_sentiment == "negatif":
            return (
                f"Aspek Layanan didominasi sentimen negatif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa kualitas layanan umum masih menjadi sumber keluhan pengguna, "
                f"seperti keterlambatan layanan, respons layanan, atau kesulitan memperoleh bantuan."
            )
        return (
            f"Aspek Layanan didominasi sentimen netral sebesar {dominant_pct}%. "
            f"Hasil ini menunjukkan bahwa ulasan pada aspek layanan cenderung tidak menunjukkan penilaian yang sangat kuat."
        )

    if aspect == "Driver":
        if dominant_sentiment == "positif":
            return (
                f"Aspek Driver didominasi sentimen positif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa pengalaman pengguna terhadap mitra pengemudi cenderung dinilai baik. "
                f"Namun, aspek Driver tetap perlu diperhatikan karena hasil SNA dapat menunjukkan isu waktu tunggu, "
                f"pembatalan, dan ketersediaan driver."
            )
        if dominant_sentiment == "negatif":
            return (
                f"Aspek Driver didominasi sentimen negatif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa pengalaman terhadap mitra pengemudi menjadi perhatian utama, "
                f"terutama terkait waktu tunggu, pembatalan pesanan, atau kesulitan memperoleh driver."
            )
        return (
            f"Aspek Driver didominasi sentimen netral sebesar {dominant_pct}%. "
            f"Hasil ini menunjukkan bahwa ulasan terkait driver cenderung informatif."
        )

    if aspect == "Tarif & Harga":
        if dominant_sentiment == "positif":
            return (
                f"Aspek Tarif & Harga didominasi sentimen positif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa pengguna cenderung masih dapat menerima tarif atau biaya layanan. "
                f"Namun, persepsi harga tetap perlu dibaca secara hati-hati karena dipengaruhi oleh jarak, waktu, promo, dan kondisi layanan."
            )
        if dominant_sentiment == "negatif":
            return (
                f"Aspek Tarif & Harga didominasi sentimen negatif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa persepsi biaya menjadi isu yang menonjol, "
                f"seperti tarif mahal, ongkos layanan, atau ketidaksesuaian harga dengan pengalaman layanan."
            )
        return (
            f"Aspek Tarif & Harga didominasi sentimen netral sebesar {dominant_pct}%. "
            f"Hasil ini menunjukkan bahwa pembahasan tarif dan harga cenderung informatif."
        )

    if aspect == "GoPay & Pembayaran":
        if dominant_sentiment == "positif":
            return (
                f"Aspek GoPay & Pembayaran didominasi sentimen positif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa pengalaman pembayaran cenderung berjalan baik, "
                f"terutama terkait proses transaksi, kemudahan pembayaran, atau penggunaan GoPay."
            )
        if dominant_sentiment == "negatif":
            return (
                f"Aspek GoPay & Pembayaran didominasi sentimen negatif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa proses pembayaran menjadi sumber keluhan, "
                f"seperti saldo, transaksi gagal, metode pembayaran, atau kendala penggunaan GoPay."
            )
        return (
            f"Aspek GoPay & Pembayaran didominasi sentimen netral sebesar {dominant_pct}%. "
            f"Hasil ini menunjukkan bahwa pembahasan pembayaran cenderung informatif."
        )

    if aspect == "Promo & Iklan":
        if dominant_sentiment == "positif":
            return (
                f"Aspek Promo & Iklan didominasi sentimen positif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa promosi, voucher, diskon, atau informasi penawaran "
                f"masih dipersepsikan bernilai oleh pengguna."
            )
        if dominant_sentiment == "negatif":
            return (
                f"Aspek Promo & Iklan didominasi sentimen negatif sebesar {dominant_pct}%. "
                f"Hasil ini menunjukkan bahwa promosi atau iklan dapat menjadi sumber ketidakpuasan, "
                f"seperti promo tidak sesuai, voucher sulit digunakan, atau iklan mengganggu."
            )
        return (
            f"Aspek Promo & Iklan didominasi sentimen netral sebesar {dominant_pct}%. "
            f"Hasil ini menunjukkan bahwa pembahasan promosi dan iklan cenderung informatif."
        )

    return (
        f"Aspek {aspect} didominasi sentimen {dominant_sentiment} sebesar {dominant_pct}%. "
        f"Hasil ini menunjukkan kecenderungan opini pengguna terhadap aspek tersebut."
    )


# ============================================================
# 6. CRAWLER GOOGLE PLAY STORE
# ============================================================

def crawl_google_play(app_id, count, lang="id", country="id"):
    if not GOOGLE_PLAY_AVAILABLE:
        raise RuntimeError(
            "Library google-play-scraper belum terinstall. "
            "Install dengan perintah: pip install google-play-scraper"
        )

    result, _ = reviews(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=count
    )

    rows = []

    for item in result:
        rows.append({
            "source": "Google Play Store",
            "source_id": item.get("reviewId", ""),
            "created_at": item.get("at", ""),
            "review_text": item.get("content", ""),
            "rating": item.get("score", "")
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Pengaturan")

model_dir = find_model_dir()

if model_dir is None:
    st.sidebar.error("Folder model IndoBERT final tidak ditemukan.")
    st.error(
        "Folder model IndoBERT final belum ditemukan. "
        "Cek kembali MODEL_DIR_CANDIDATES di app.py."
    )
    st.stop()

st.sidebar.success("Model IndoBERT ditemukan")
st.sidebar.caption(model_dir)

tokenizer, model, device = load_model(model_dir)
st.sidebar.write("Device:", device)

st.sidebar.markdown("### Sumber Data")
st.sidebar.info(
    "Sumber data deployment dibatasi hanya dari Google Play Store "
    "sesuai ruang lingkup penelitian skripsi."
)

max_rows_predict = st.sidebar.number_input(
    "Maksimal data dianalisis",
    min_value=1,
    max_value=1000,
    value=100,
    step=10
)


# ============================================================
# 8. DATA INPUT — GOOGLE PLAY STORE ONLY
# ============================================================

df_input = None

st.subheader("1. Pengambilan Data dari Google Play Store")

st.write(
    "Pada deployment ini, sumber data dibatasi hanya dari Google Play Store "
    "karena sesuai dengan ruang lingkup penelitian skripsi, yaitu ulasan aplikasi Gojek "
    "pada Google Play Store."
)

app_id = st.text_input("App ID", value="com.gojek.app")
count = st.number_input("Jumlah ulasan", min_value=10, max_value=5000, value=100, step=10)
lang = st.text_input("Language", value="id")
country = st.text_input("Country", value="id")

if st.button("Crawl Google Play Store"):
    with st.spinner("Mengambil data ulasan dari Google Play Store..."):
        try:
            df_input = crawl_google_play(app_id, count, lang, country)
            st.session_state["df_input"] = df_input
            st.success(f"Berhasil mengambil {len(df_input)} ulasan dari Google Play Store.")
        except Exception as e:
            st.error(str(e))

if df_input is None and "df_input" in st.session_state:
    df_input = st.session_state["df_input"]

if df_input is not None:
    st.write("Data ulasan Google Play Store yang akan dianalisis:")
    st.dataframe(df_input.head(20), use_container_width=True)


# ============================================================
# 9. RUN ABSA
# ============================================================

st.subheader("2. Analisis ABSA Positif, Negatif, dan Netral")

if df_input is not None:
    if st.button("Jalankan Analisis ABSA"):
        with st.spinner("Model sedang menganalisis sentimen per aspek..."):
            df_result = run_absa(
                df=df_input,
                text_col="review_text",
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_rows=max_rows_predict
            )

            st.session_state["df_result"] = df_result
            st.success("Analisis ABSA selesai.")


# ============================================================
# 10. OUTPUT ANALYSIS
# ============================================================

if "df_result" in st.session_state:
    df_result = st.session_state["df_result"].copy()

    st.subheader("3. Hasil Prediksi ABSA")
    st.dataframe(df_result, use_container_width=True)

    csv_result = df_result.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download Hasil ABSA CSV",
        data=csv_result,
        file_name="hasil_absa_streamlit.csv",
        mime="text/csv"
    )

    # ========================================================
    # 4. RINGKASAN SENTIMEN PER ASPEK
    # ========================================================

    st.subheader("4. Ringkasan Sentimen Per Aspek")

    pivot = pd.crosstab(
        df_result["aspek"],
        df_result["sentiment"]
    ).reindex(index=ASPECTS, columns=LABEL_ORDER, fill_value=0)

    pivot["total"] = pivot.sum(axis=1)

    for label in LABEL_ORDER:
        pivot[f"{label}_%"] = (
            pivot[label] / pivot["total"].replace(0, np.nan) * 100
        ).fillna(0).round(2)

    st.dataframe(pivot, use_container_width=True)

    pivot_reset = pivot.reset_index()

    fig = px.bar(
        pivot_reset,
        x="aspek",
        y=LABEL_ORDER,
        barmode="group",
        title="Distribusi Sentimen Positif, Negatif, dan Netral per Aspek",
        labels={
            "value": "Jumlah",
            "variable": "Sentimen",
            "aspek": "Aspek"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # 5. KESIMPULAN ABSA PER ASPEK
    # ========================================================

    st.subheader("5. Kesimpulan ABSA Per Aspek")

    conclusion_rows = []

    for aspect, row in pivot.iterrows():
        counts = {
            "negatif": int(row.get("negatif", 0)),
            "netral": int(row.get("netral", 0)),
            "positif": int(row.get("positif", 0))
        }

        dominant_sentiment = max(counts, key=counts.get)
        dominant_count = counts[dominant_sentiment]
        total = int(row["total"])
        dominant_pct = round((dominant_count / total * 100), 2) if total > 0 else 0

        conclusion = build_interpretative_conclusion(
            aspect=aspect,
            dominant_sentiment=dominant_sentiment,
            dominant_pct=dominant_pct,
            counts=counts
        )

        conclusion_rows.append({
            "aspek": aspect,
            "sentimen_dominan": dominant_sentiment,
            "jumlah_dominan": dominant_count,
            "persentase_dominan_%": dominant_pct,
            "kesimpulan_absa": conclusion
        })

    df_conclusion = pd.DataFrame(conclusion_rows)

    st.dataframe(df_conclusion, use_container_width=True)

    for _, row in df_conclusion.iterrows():
        st.markdown(
            f"**{row['aspek']}** — dominan **{row['sentimen_dominan']}** "
            f"({row['persentase_dominan_%']}%)."
        )
        st.write(row["kesimpulan_absa"])

    csv_conclusion = df_conclusion.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download Kesimpulan ABSA CSV",
        data=csv_conclusion,
        file_name="kesimpulan_absa_per_aspek.csv",
        mime="text/csv"
    )

    # ========================================================
    # 6. RINGKASAN UMUM SENTIMEN
    # ========================================================

    st.subheader("6. Ringkasan Umum Sentimen")

    total_sentiment = (
        df_result["sentiment"]
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
    )

    total_predictions = int(total_sentiment.sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Negatif", int(total_sentiment["negatif"]))
    col2.metric("Netral", int(total_sentiment["netral"]))
    col3.metric("Positif", int(total_sentiment["positif"]))

    total_sentiment_df = total_sentiment.reset_index()
    total_sentiment_df.columns = ["sentiment", "jumlah"]

    if total_predictions > 0:
        total_sentiment_df["persentase_%"] = (
            total_sentiment_df["jumlah"] / total_predictions * 100
        ).round(2)
    else:
        total_sentiment_df["persentase_%"] = 0

    st.dataframe(total_sentiment_df, use_container_width=True)

    fig_overall = px.pie(
        total_sentiment_df,
        names="sentiment",
        values="jumlah",
        title="Proporsi Sentimen Keseluruhan"
    )

    st.plotly_chart(fig_overall, use_container_width=True)

    dominant_overall = total_sentiment.idxmax()
    dominant_overall_count = int(total_sentiment.max())

    dominant_overall_pct = (
        dominant_overall_count / total_predictions * 100
    ) if total_predictions > 0 else 0

    st.info(
        f"Sentimen dominan secara umum adalah **{dominant_overall}** "
        f"dengan jumlah **{dominant_overall_count}** prediksi "
        f"atau **{dominant_overall_pct:.2f}%** dari seluruh hasil prediksi aspek-sentimen."
    )

    # ========================================================
    # 7. SEMANTIC NETWORK ANALYSIS PER ASPEK
    # ========================================================

    st.subheader("7. Semantic Network Analysis Per Aspek")

    st.write(
        "Bagian ini menampilkan hasil Semantic Network Analysis berdasarkan aspek hasil ABSA. "
        "ABSA digunakan untuk menentukan aspek dan sentimen, sedangkan SNA digunakan untuk "
        "melihat kata serta hubungan antarkata yang membentuk opini pada setiap aspek. "
        "Analisis utama SNA difokuskan pada sentimen positif dan negatif karena keduanya memiliki arah opini yang jelas."
    )

    selected_sna_aspect = st.selectbox(
        "Pilih aspek untuk dianalisis dengan SNA",
        ASPECTS
    )

    selected_sna_sentiments = st.multiselect(
        "Pilih sentimen untuk SNA",
        options=LABEL_ORDER,
        default=["negatif", "positif"]
    )

    if not selected_sna_sentiments:
        st.warning("Pilih minimal satu sentimen untuk membentuk SNA.")
    else:
        if "netral" in selected_sna_sentiments:
            st.warning(
                "Sentimen netral dapat ditampilkan untuk eksplorasi, tetapi interpretasi utama SNA "
                "disarankan pada sentimen positif dan negatif karena keduanya memiliki arah opini yang lebih jelas."
            )

        df_top_words, df_edges, G_aspect, df_sna_aspect = build_sna_aspect(
            df_pred=df_result,
            selected_aspect=selected_sna_aspect,
            selected_sentiments=selected_sna_sentiments,
            top_n_words=15,
            top_n_edges=20
        )

        st.markdown(f"### Hasil SNA untuk Aspek {selected_sna_aspect}")

        col_sna_1, col_sna_2, col_sna_3 = st.columns(3)

        with col_sna_1:
            st.metric("Jumlah Data Aspek", len(df_result[df_result["aspek"] == selected_sna_aspect]))

        with col_sna_2:
            st.metric("Data untuk SNA", len(df_sna_aspect))

        with col_sna_3:
            dominant_sna_sentiment = (
                df_sna_aspect["sentiment"].value_counts().idxmax()
                if not df_sna_aspect.empty else "-"
            )
            st.metric("Sentimen Dominan SNA", dominant_sna_sentiment)

        if df_sna_aspect.empty:
            st.info("Tidak ada data yang tersedia untuk kombinasi aspek dan sentimen yang dipilih.")
        else:
            st.markdown("#### Distribusi Sentimen pada Aspek Terpilih")

            sentiment_aspect_summary = (
                df_result[df_result["aspek"] == selected_sna_aspect]["sentiment"]
                .value_counts()
                .reindex(LABEL_ORDER, fill_value=0)
                .reset_index()
            )
            sentiment_aspect_summary.columns = ["sentiment", "jumlah"]
            sentiment_aspect_total = int(sentiment_aspect_summary["jumlah"].sum())
            sentiment_aspect_summary["persentase_%"] = (
                sentiment_aspect_summary["jumlah"] / sentiment_aspect_total * 100
            ).round(2) if sentiment_aspect_total > 0 else 0

            st.dataframe(sentiment_aspect_summary, use_container_width=True)

            fig_sentiment_aspect = px.bar(
                sentiment_aspect_summary,
                x="sentiment",
                y="jumlah",
                text="jumlah",
                title=f"Distribusi Sentimen pada Aspek {selected_sna_aspect}",
                labels={"sentiment": "Sentimen", "jumlah": "Jumlah"}
            )
            st.plotly_chart(fig_sentiment_aspect, use_container_width=True)

            st.markdown("#### Kata Dominan SNA per Aspek")

            if df_top_words.empty:
                st.info("Tidak ada kata dominan yang dapat ditampilkan.")
            else:
                st.dataframe(df_top_words, use_container_width=True)

                fig_top_words = px.bar(
                    df_top_words,
                    x="frekuensi",
                    y="kata",
                    orientation="h",
                    text="frekuensi",
                    title=f"Kata Dominan pada Aspek {selected_sna_aspect}",
                    labels={"frekuensi": "Frekuensi", "kata": "Kata"}
                )
                fig_top_words.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_top_words, use_container_width=True)

            st.markdown("#### Hubungan Kata Co-occurrence per Aspek")

            if df_edges.empty:
                st.info("Tidak ada hubungan co-occurrence yang dapat ditampilkan.")
            else:
                st.dataframe(df_edges, use_container_width=True)

            st.markdown("#### Visualisasi Network Co-occurrence per Aspek")

            fig_network_aspect = plot_sna_network(
                G_aspect,
                title=f"Network Co-occurrence Kata pada Aspek {selected_sna_aspect}"
            )

            if fig_network_aspect is not None:
                st.plotly_chart(fig_network_aspect, use_container_width=True)
            else:
                st.info("Network belum dapat dibentuk karena jumlah hubungan kata terlalu sedikit.")

            st.markdown("#### Kesimpulan Hubungan ABSA dan SNA per Aspek")

            top_words_text = ", ".join(df_top_words["kata"].head(5).tolist()) if not df_top_words.empty else "-"

            top_edges_text = ", ".join(
                [
                    f"{row['kata_1']}–{row['kata_2']}"
                    for _, row in df_edges.head(3).iterrows()
                ]
            ) if not df_edges.empty else "-"

            dominant_absa_sentiment = (
                df_conclusion[df_conclusion["aspek"] == selected_sna_aspect]["sentimen_dominan"].iloc[0]
                if selected_sna_aspect in df_conclusion["aspek"].values
                else dominant_sna_sentiment
            )

            st.info(
                f"Aspek **{selected_sna_aspect}** memiliki sentimen dominan ABSA **{dominant_absa_sentiment}**. "
                f"Hasil SNA pada aspek ini menunjukkan kata dominan seperti **{top_words_text}**. "
                f"Hubungan kata yang menonjol terlihat pada pasangan **{top_edges_text}**. "
                f"Dengan demikian, ABSA menjelaskan arah sentimen pada aspek {selected_sna_aspect}, sedangkan SNA "
                f"memperjelas kata dan isu yang membentuk sentimen tersebut."
            )

            csv_top_words = df_top_words.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label=f"Download Kata Dominan SNA Aspek {selected_sna_aspect}",
                data=csv_top_words,
                file_name=f"sna_top_words_{selected_sna_aspect}.csv",
                mime="text/csv"
            )

            csv_edges = df_edges.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label=f"Download Edge SNA Aspek {selected_sna_aspect}",
                data=csv_edges,
                file_name=f"sna_edges_{selected_sna_aspect}.csv",
                mime="text/csv"
            )

    dynamic_sna_context = get_dynamic_sna_context(df_result)

    # ========================================================
    # 8. INTEGRASI HASIL ABSA DAN SNA
    # ========================================================

    st.subheader("8. Integrasi Hasil ABSA dan Semantic Network Analysis")

    st.write(
        "Bagian ini menghubungkan hasil ABSA dengan temuan SNA per aspek. "
        "ABSA digunakan untuk mengetahui sentimen dominan pada setiap aspek, sedangkan SNA "
        "digunakan untuk menjelaskan kata dominan dan hubungan co-occurrence yang membentuk opini pengguna."
    )

    integration_rows = []

    for _, row in df_conclusion.iterrows():
        aspect = row["aspek"]
        sentiment_dom = row["sentimen_dominan"]
        absa_conclusion = row["kesimpulan_absa"]

        sna_info = dynamic_sna_context.get(
            aspect,
            {
                "kata_sna": "-",
                "edge_sna": "-",
                "temuan_sna": "Temuan SNA untuk aspek ini belum tersedia.",
                "jumlah_data_sna": 0
            }
        )

        integration_rows.append({
            "aspek_absa": aspect,
            "sentimen_dominan_absa": sentiment_dom,
            "jumlah_data_sna_positif_negatif": sna_info["jumlah_data_sna"],
            "kata_terkait_sna": sna_info["kata_sna"],
            "edge_terkait_sna": sna_info["edge_sna"],
            "temuan_sna": sna_info["temuan_sna"],
            "kesimpulan_integratif": (
                f"{absa_conclusion} Temuan SNA menunjukkan bahwa {sna_info['temuan_sna']} "
                f"Dengan demikian, ABSA menunjukkan arah sentimen aspek, sedangkan SNA menjelaskan kata dan hubungan kata yang membentuk opini pengguna."
            )
        })

    df_integration = pd.DataFrame(integration_rows)

    st.dataframe(df_integration, use_container_width=True)

    csv_integration = df_integration.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download Integrasi ABSA dan SNA CSV",
        data=csv_integration,
        file_name="integrasi_absa_sna_streamlit.csv",
        mime="text/csv"
    )

    # ========================================================
    # 9. KESIMPULAN AKHIR DEPLOYMENT
    # ========================================================

    st.subheader("9. Kesimpulan Akhir Deployment")

    if not df_conclusion.empty:
        top_aspect_row = df_conclusion.loc[df_conclusion["persentase_dominan_%"].idxmax()]

        st.success(
            f"Dashboard berhasil menghasilkan **{total_predictions} prediksi aspek-sentimen**. "
            f"Jumlah prediksi positif adalah **{int(total_sentiment['positif'])}**, "
            f"netral **{int(total_sentiment['netral'])}**, dan negatif **{int(total_sentiment['negatif'])}**. "
            f"Sentimen dominan secara umum adalah **{dominant_overall}**."
        )

        st.write(
            f"Berdasarkan kesimpulan ABSA per aspek, aspek **{top_aspect_row['aspek']}** "
            f"memiliki sentimen dominan **{top_aspect_row['sentimen_dominan']}** "
            f"dengan persentase **{top_aspect_row['persentase_dominan_%']}%**. "
            f"Hasil ini menunjukkan bahwa aspek tersebut menjadi aspek dengan kecenderungan sentimen paling kuat pada data yang dianalisis."
        )

        st.markdown("### Ringkasan Kata dan Edge SNA per Aspek")

        sna_summary_rows = []

        for _, row in df_conclusion.iterrows():
            aspect = row["aspek"]
            sentiment_dom = row["sentimen_dominan"]
            pct = row["persentase_dominan_%"]

            sna_info = dynamic_sna_context.get(
                aspect,
                {
                    "kata_sna": "-",
                    "edge_sna": "-",
                    "temuan_sna": "Temuan SNA untuk aspek ini belum tersedia.",
                    "jumlah_data_sna": 0
                }
            )

            sna_summary_rows.append({
                "aspek": aspect,
                "sentimen_dominan_absa": sentiment_dom,
                "persentase_dominan_%": pct,
                "jumlah_data_sna_positif_negatif": sna_info["jumlah_data_sna"],
                "kata_terkait_sna": sna_info["kata_sna"],
                "edge_terkait_sna": sna_info["edge_sna"],
                "makna_temuan_sna": sna_info["temuan_sna"]
            })

        df_sna_summary_final = pd.DataFrame(sna_summary_rows)

        st.dataframe(df_sna_summary_final, use_container_width=True)

        st.write(
            "Tabel di atas menunjukkan bahwa kesimpulan akhir tidak hanya didasarkan pada sentimen dominan ABSA, "
            "tetapi juga diperkuat oleh kata dominan dan hubungan co-occurrence yang muncul pada hasil SNA per aspek. "
            "Dengan demikian, setiap aspek memiliki konteks kata yang dapat digunakan untuk menjelaskan alasan di balik interpretasi hasil."
        )

        st.markdown("### Kesimpulan Integratif per Aspek")

        for _, row in df_sna_summary_final.iterrows():
            aspect = row["aspek"]
            sentiment_dom = row["sentimen_dominan_absa"]
            pct = row["persentase_dominan_%"]
            kata_sna = row["kata_terkait_sna"]
            edge_sna = row["edge_terkait_sna"]
            makna_sna = row["makna_temuan_sna"]

            st.markdown(f"**{aspect}**")
            st.write(
                f"Aspek **{aspect}** memiliki sentimen dominan **{sentiment_dom}** dengan persentase **{pct}%**. "
                f"Kata terkait SNA pada aspek ini adalah **{kata_sna}**, dengan hubungan kata menonjol **{edge_sna}**. "
                f"{makna_sna} Dengan demikian, kesimpulan pada aspek **{aspect}** tidak hanya dilihat dari jumlah sentimen, "
                f"tetapi juga dari kata dan hubungan kata yang membentuk konteks opini pengguna."
            )

        st.markdown("### Kesimpulan Keseluruhan")

        st.write(
            "Secara keseluruhan, hasil deployment menunjukkan kecenderungan sentimen pengguna berdasarkan hasil ABSA dan konteks kata berdasarkan SNA. "
            "ABSA menunjukkan aspek dan arah sentimen dominan, sedangkan SNA per aspek menunjukkan kata dominan serta hubungan co-occurrence yang membentuk opini pengguna. "
            "Dengan demikian, deployment ini tidak hanya menampilkan prediksi sentimen, tetapi juga memberikan kesimpulan berbasis aspek yang diperkuat oleh analisis jaringan kata."
        )

        csv_sna_summary_final = df_sna_summary_final.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Download Ringkasan Akhir ABSA dan SNA CSV",
            data=csv_sna_summary_final,
            file_name="ringkasan_akhir_absa_sna.csv",
            mime="text/csv"
        )

else:
    st.info("Silakan ambil data dari Google Play Store terlebih dahulu, lalu jalankan analisis ABSA.")