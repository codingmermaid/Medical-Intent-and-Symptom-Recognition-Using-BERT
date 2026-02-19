import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_DIR = BASE_DIR / "model_artifacts" / "intent_bert"
LABELS_PATH = MODEL_DIR / "labels.json"
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "")
if not HF_MODEL_ID and hasattr(st, "secrets"):
    HF_MODEL_ID = st.secrets.get("HF_MODEL_ID", "")

st.set_page_config(page_title="Medical Intent Classifier", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      .stAppDeployButton {display:none;}
      footer {visibility: hidden;}

      .stApp {
        background:
          radial-gradient(circle at 8% 8%, rgba(255, 207, 163, 0.28), transparent 28%),
          radial-gradient(circle at 92% 10%, rgba(128, 181, 255, 0.22), transparent 30%),
          linear-gradient(180deg, #f8fbff 0%, #f5faf7 100%);
      }

      .stMainBlockContainer {
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
        max-width: 1020px;
      }

      .hero {
        border: 1px solid rgba(23, 62, 102, 0.18);
        border-radius: 22px;
        padding: 22px 24px;
        background: linear-gradient(120deg, rgba(255,255,255,0.90), rgba(241,248,255,0.94));
        box-shadow: 0 14px 42px rgba(14, 28, 47, 0.09);
        margin-bottom: 16px;
      }

      .hero h1 {
        margin: 0;
        color: #17345d;
        letter-spacing: 0.2px;
        font-size: 2rem;
      }

      .hero p {
        margin: 8px 0 0 0;
        color: #38536e;
        font-size: 1rem;
      }

      .panel {
        border: 1px solid rgba(43, 78, 53, 0.20);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 8px 20px rgba(21, 46, 33, 0.06);
      }

      .prediction-card {
        border: 1px solid rgba(21, 101, 72, 0.22);
        border-radius: 16px;
        padding: 16px 18px;
        background: linear-gradient(120deg, rgba(229, 249, 236, 0.95), rgba(242, 255, 247, 0.98));
        margin: 10px 0 12px 0;
      }

      .intent-name {
        margin: 0;
        color: #0e5d43;
        font-size: 1.15rem;
        font-weight: 700;
      }

      .intent-meta {
        margin-top: 4px;
        color: #235d4c;
        font-size: 0.95rem;
      }

      .small-note {
        color: #4e6377;
        font-size: 0.88rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_label_names(data_path: Path, labels_path: Path) -> list[str]:
    if labels_path.exists():
        data = json.loads(labels_path.read_text())
        if isinstance(data, list) and data:
            return data

    if data_path.exists():
        df = pd.read_csv(data_path)
        if "prompt" in df.columns:
            return sorted(df["prompt"].astype(str).dropna().unique().tolist())

    return []


@st.cache_resource
def load_model_and_tokenizer(model_dir: Path, reload_token: int):
    _ = reload_token
    local_error = None

    if model_dir.exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            model.eval()
            return model, tokenizer, None
        except Exception as exc:  # noqa: BLE001
            local_error = str(exc)
    else:
        local_error = "Local model directory not found."

    if HF_MODEL_ID:
        try:
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
            model.eval()
            return model, tokenizer, None
        except Exception as exc:  # noqa: BLE001
            return None, None, f"Load failed: {exc}"

    return None, None, f"Load failed: {local_error}"


def predict(model, tokenizer, text: str, label_names: list[str], top_k: int = 5) -> pd.DataFrame:
    encoded = tokenizer([text], padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        output = model(**encoded)
        probs = torch.softmax(output.logits, dim=1).squeeze(0)

    top_k = min(top_k, probs.shape[0])
    values, indices = torch.topk(probs, k=top_k)

    rows = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        label = label_names[idx] if idx < len(label_names) else f"Class {idx}"
        rows.append({"intent": label, "confidence": score})

    return pd.DataFrame(rows)


if "model_reload_token" not in st.session_state:
    st.session_state["model_reload_token"] = 0
if "symptom_text" not in st.session_state:
    st.session_state["symptom_text"] = ""

model, tokenizer, model_error = load_model_and_tokenizer(MODEL_DIR, st.session_state["model_reload_token"])
label_names = load_label_names(DATA_PATH, LABELS_PATH)

if not label_names and model is not None:
    id2label = getattr(model.config, "id2label", {}) or {}
    if id2label:
        sorted_keys = sorted(id2label.keys(), key=lambda k: int(k))
        label_names = [id2label[k] for k in sorted_keys]

st.markdown(
    """
    <div class="hero">
      <h1>Medical Intent Explorer</h1>
      <p>Describe symptoms in natural language and get ranked intent predictions with confidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_main, col_side = st.columns([2.35, 1], gap="large")

with col_side:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("System")
    if model is None:
        st.error("Model unavailable")
        st.caption("Check deployment config and model files.")
    else:
        st.success("Model ready")
    st.caption(f"Intent labels: {len(label_names)}")
    if st.button("Reload", use_container_width=True):
        st.session_state["model_reload_token"] += 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Quick prompt ideas**")
    st.markdown("<div class='small-note'>Try one short symptom sentence, then submit.</div>", unsafe_allow_html=True)

    quick_prompts = [
        "I have a severe headache since morning.",
        "My chest hurts when I breathe deeply.",
        "I feel dizzy and weak today.",
        "I injured my knee while running.",
    ]
    for i, qp in enumerate(quick_prompts):
        if st.button(qp, key=f"qp_{i}"):
            st.session_state["symptom_text"] = qp
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col_main:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.text_area(
            "Describe what you feel",
            key="symptom_text",
            height=150,
            placeholder="Example: I have a headache and feel dizzy.",
        )
        submit = st.form_submit_button("Analyze Symptoms", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        user_text = st.session_state["symptom_text"].strip()

        if not user_text:
            st.warning("Please enter symptoms first.")
        elif model is None or tokenizer is None:
            st.error("Prediction unavailable right now.")
            st.caption(model_error if model_error else "Model failed to initialize.")
        else:
            result_df = predict(model, tokenizer, user_text, label_names, top_k=5)
            best = result_df.iloc[0]

            st.markdown(
                f"""
                <div class="prediction-card">
                  <p class="intent-name">Top Match: {best['intent']}</p>
                  <p class="intent-meta">Confidence: {best['confidence'] * 100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(float(best["confidence"]))

            if best["confidence"] < 0.45:
                st.warning("Low confidence. Consider describing symptoms with more detail.")

            st.markdown("### Ranked Intents")
            plot_df = result_df.copy()
            plot_df["confidence_pct"] = plot_df["confidence"] * 100

            for _, row in plot_df.iterrows():
                st.markdown(f"**{row['intent']}**")
                st.progress(float(row["confidence_pct"]/100.0), text=f"{row['confidence_pct']:.2f}%")

            display_df = plot_df[["intent", "confidence_pct"]].rename(columns={"confidence_pct": "confidence (%)"})
            st.dataframe(
                display_df.style.format({"confidence (%)": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )
