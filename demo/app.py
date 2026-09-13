"""Live demo: EN→NL software-domain translation with MarianMT.

Loads the Helsinki-NLP/opus-mt-en-nl encoder-decoder model (the base model
this repository's pipeline fine-tunes) and translates English input to Dutch
on CPU. The fine-tuned checkpoints themselves are training artifacts and are
not shipped in the repository, so this demo serves the baseline model and
shows the domain test samples the evaluation harness scores.
"""

import streamlit as st

st.set_page_config(page_title="EN→NL Domain Translation", page_icon="🌐", layout="centered")

st.title("🌐 EN→NL Software-Domain Translation")
st.caption(
    "MarianMT (Helsinki-NLP/opus-mt-en-nl, 148M parameters) — the base model of this "
    "repository's dual-track fine-tuning pipeline (full fine-tune + BLOOM-560m LoRA). "
    "Evaluation harness: BLEU · chrF · TER · COMET."
)

SAMPLES = [
    "Disconnected {1}",
    "Increased contrast",
    "Update window",
    "The configuration file could not be parsed.",
    "Restart the service to apply the new settings.",
]


@st.cache_resource(show_spinner="Loading MarianMT (first run downloads ~300 MB)…")
def load_model():
    from transformers import MarianMTModel, MarianTokenizer

    name = "Helsinki-NLP/opus-mt-en-nl"
    tokenizer = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    model.eval()
    return tokenizer, model


def translate(text: str) -> str:
    tokenizer, model = load_model()
    batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
    generated = model.generate(**batch, max_length=512, num_beams=4)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


sample = st.selectbox("Try a software-domain sample", ["(type your own below)"] + SAMPLES)
default_text = "" if sample == "(type your own below)" else sample
text = st.text_area("English input", value=default_text, height=120,
                    placeholder="Enter English text to translate to Dutch…")

if st.button("Translate", type="primary") and text.strip():
    try:
        with st.spinner("Translating…"):
            result = translate(text.strip())
        st.subheader("Dutch translation")
        st.success(result)
    except Exception as exc:  # noqa: BLE001 - surface any load/runtime failure to the viewer
        st.error(f"Translation failed: {exc}")

with st.expander("About this pipeline"):
    st.markdown(
        "- **Task 1** — full-parameter fine-tuning of MarianMT in PyTorch Lightning\n"
        "- **Task 2** — LoRA instruction tuning of BLOOM-560m (1.2M trainable parameters, 0.2%)\n"
        "- **Task 3** — evaluation on FLORES-devtest and an 84-sample software-domain test set\n"
        "  with BLEU, chrF, TER, and COMET; per-sample translation audits in `outputs/evaluation/`"
    )
