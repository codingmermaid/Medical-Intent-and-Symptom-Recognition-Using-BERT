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
      .stMainBlockContainer {padding-top: 1.3rem; padding-bottom: 1.6rem; max-width: 980px;}
      .card {
        border: 1px solid rgba(49, 89, 64, 0.25);
        border-radius: 14px;
        padding: 14px 16px;
        background: linear-gradient(180deg, rgba(240,248,241,0.9), rgba(255,255,255,1));
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_examples(path: Path, n_examples: int = 200) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "phrase" not in df.columns:
        return []
    examples = (
        df["phrase"]
        .astype(str)
        .str.strip()
        .dropna()
        .drop_duplicates()
        .head(n_examples)
        .tolist()
    )
    return examples


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
            return model, tokenizer, None, "local"
        except Exception as exc:  # noqa: BLE001
            local_error = str(exc)
    else:
        local_error = "Local model directory not found."

    if HF_MODEL_ID:
        try:
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
            model.eval()
            return model, tokenizer, None, "huggingface"
        except Exception as exc:  # noqa: BLE001
            return None, None, f"Local load failed: {local_error} | HF load failed: {exc}", None

    return None, None, f"Local load failed: {local_error}. Set HF_MODEL_ID for cloud fallback.", None


def predict(model, tokenizer, text: str, label_names: list[str], top_k: int = 5) -> pd.DataFrame:
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

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


st.title("Medical Intent Classifier Dashboard")
st.caption("Enter symptoms in plain language and get predicted medical intent categories.")

col_main, col_side = st.columns([2.1, 1], gap="large")

with col_side:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Status")
    st.write(f"Expected model path: `{MODEL_DIR}`")
    st.write(f"Hugging Face fallback: `{HF_MODEL_ID or 'not configured'}`")

    if "model_reload_token" not in st.session_state:
        st.session_state["model_reload_token"] = 0
    if st.button("Reload model"):
        st.session_state["model_reload_token"] += 1

    model, tokenizer, model_error, model_source = load_model_and_tokenizer(
        MODEL_DIR, st.session_state["model_reload_token"]
    )
    label_names = load_label_names(DATA_PATH, LABELS_PATH)
    if not label_names and model is not None:
        id2label = getattr(model.config, "id2label", {}) or {}
        if id2label:
            sorted_keys = sorted(id2label.keys(), key=lambda k: int(k))
            label_names = [id2label[k] for k in sorted_keys]

    if model is None:
        st.error("Model is not loaded.")
        st.write(model_error)
    else:
        if model_source == "local":
            st.success("Model loaded from local artifacts.")
        else:
            st.success("Model loaded from Hugging Face Hub.")

    st.write(f"Detected labels: `{len(label_names)}`")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("How to save your trained model"):
        st.code(
            """
# Run this in your training notebook after training:
output_dir = "model_artifacts/intent_bert"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

import json
with open(f"{output_dir}/labels.json", "w") as f:
    json.dump(list(LE.classes_), f)
            """.strip(),
            language="python",
        )
    with st.expander("Cloud deploy model fallback"):
        st.markdown(
            "Set `HF_MODEL_ID` to your Hub repo (for example: `username/medical-intent-bert`). "
            "You can set it in Streamlit Cloud app settings or `.streamlit/secrets.toml`."
        )

with col_main:
    examples = load_examples(DATA_PATH)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    selected = st.selectbox(
        "Pick an example sentence (optional)",
        options=[""] + examples,
        index=0,
    )

    if "symptom_text" not in st.session_state:
        st.session_state["symptom_text"] = ""

    if st.button("Use selected example") and selected:
        st.session_state["symptom_text"] = selected

    st.text_area(
        "Describe what you feel",
        key="symptom_text",
        height=130,
        placeholder="Example: I have a headache and feel dizzy.",
    )

    submit = st.button("Submit", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        user_text = st.session_state["symptom_text"].strip()

        if not user_text:
            st.warning("Please enter symptoms first.")
        elif model is None or tokenizer is None:
            st.error("Cannot run prediction because model is not loaded.")
        else:
            result_df = predict(model, tokenizer, user_text, label_names, top_k=5)

            best = result_df.iloc[0]
            st.subheader("Prediction")
            st.success(f"Most likely intent: **{best['intent']}** ({best['confidence'] * 100:.2f}%)")

            if best["confidence"] < 0.45:
                st.warning("Low confidence prediction. Consider collecting more examples for similar symptoms.")

            st.subheader("Top-5 intents")
            plot_df = result_df.copy()
            plot_df["confidence"] = plot_df["confidence"] * 100
            st.bar_chart(plot_df.set_index("intent")["confidence"], height=290)
            st.dataframe(
                plot_df.rename(columns={"confidence": "confidence_%"}).style.format({"confidence_%": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )
