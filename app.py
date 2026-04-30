import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import whisper
from audiorecorder import audiorecorder
from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Multimodal Sarcasm Detection",
    page_icon="🎭",
    layout="wide"
)

# =====================================================
# CUSTOM CSS (ATTRACTIVE UI)
# =====================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

.header {
    background: linear-gradient(90deg,#4facfe,#00f2fe,#7b2ff7);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 20px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.10);
    margin-bottom: 18px;
}

.result-box {
    padding: 18px;
    border-radius: 12px;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}

.pos {
    background: #d4edda;
    color: #155724;
}

.neg {
    background: #f8d7da;
    color: #721c24;
}

.neu {
    background: #fff3cd;
    color: #856404;
}

.sar {
    background: #ffe6f2;
    color: #c2185b;
}

.small {
    font-size: 18px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    '<div class="header">🎭 Multimodal Emotion & Sarcasm Detection</div>',
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODELS
# =====================================================

# Add FFmpeg path
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-8.1-essentials_build\bin"

@st.cache_resource
def load_audio_model():
    return load_model("audio_emotion_model.keras")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

audio_model = load_audio_model()
whisper_model = load_whisper_model()

# =====================================================
# LABEL MAP
# =====================================================
audio_to_sentiment = {
    0: "negative",   # anger
    1: "negative",   # disgust
    2: "negative",   # fear
    3: "positive",   # joy
    4: "neutral",    # neutral
    5: "negative",   # sadness
    6: "positive"    # surprise
}

# =====================================================
# AUDIO FEATURE EXTRACTION
# =====================================================
def get_mel_image(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)

    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)
    delta = librosa.util.fix_length(delta, size=128, axis=1)
    delta2 = librosa.util.fix_length(delta2, size=128, axis=1)

    img = np.stack([mel_db, delta, delta2], axis=-1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img.astype(np.float32)

# =====================================================
# TEXT SENTIMENT (TEMPORARY RULE BASED)
# Replace later with your real text model
# =====================================================
def predict_text_sentiment(text):
    text = text.lower()

    positive_words = ["happy","great","love","excited","good","awesome","job"]
    negative_words = ["bad","hate","angry","sad","terrible","worst"]

    pos_score = sum(word in text for word in positive_words)
    neg_score = sum(word in text for word in negative_words)

    if pos_score > neg_score:
        return "positive", 0.92
    elif neg_score > pos_score:
        return "negative", 0.91
    else:
        return "neutral", 0.80

# =====================================================
# AUDIO SENTIMENT
# =====================================================
def predict_audio_sentiment(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    feature = get_mel_image(y, sr)

    X = np.expand_dims(feature, axis=0)

    probs = audio_model.predict(X, verbose=0)[0]
    pred_class = np.argmax(probs)

    label = audio_to_sentiment[pred_class]
    conf = float(np.max(probs))

    return label, conf

# =====================================================
# TRANSCRIBE AUDIO
# =====================================================
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# =====================================================
# SARCASM LOGIC
# =====================================================
def detect_sarcasm(text_label, audio_label, text_conf, audio_conf):
    if text_label != audio_label:
        sarcasm = "YES 😏"
        score = round(((text_conf + audio_conf) / 2) * 100, 2)
    else:
        sarcasm = "NO 🙂"
        score = round((1 - abs(text_conf - audio_conf)) * 20, 2)

    return sarcasm, score

# =====================================================
# HELPER FOR COLORS
# =====================================================
def get_class(label):
    if label == "positive":
        return "pos"
    elif label == "negative":
        return "neg"
    else:
        return "neu"

# =====================================================
# LAYOUT
# =====================================================
left, center, right = st.columns([1.1,2,1])

# Storage variables
transcript = ""
text_label = ""
audio_label = ""
sarcasm = ""
score = 0

# =====================================================
# LEFT PANEL
# =====================================================
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎤 Input Section")

    mode = st.radio(
        "Choose Input Type",
        ["Text", "Audio Upload", "Live Microphone"]
    )

    analyze = False

    # TEXT MODE
    if mode == "Text":
        user_text = st.text_area("Enter Text")
        if st.button("🚀 Analyze Text"):
            if user_text.strip() != "":
                text_label, text_conf = predict_text_sentiment(user_text)
                transcript = user_text
                audio_label = "neutral"
                sarcasm, score = detect_sarcasm(
                    text_label, audio_label,
                    text_conf, 0.80
                )
                analyze = True

    # AUDIO UPLOAD
    elif mode == "Audio Upload":
        audio_file = st.file_uploader(
            "Upload Audio",
            type=["wav","mp3","aac"]
        )

        if audio_file is not None:
            st.audio(audio_file)

            if st.button("🚀 Analyze Audio"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    temp_path = tmp.name

                transcript = transcribe_audio(temp_path)
                text_label, text_conf = predict_text_sentiment(transcript)
                audio_label, audio_conf = predict_audio_sentiment(temp_path)
                sarcasm, score = detect_sarcasm(
                    text_label, audio_label,
                    text_conf, audio_conf
                )

                os.remove(temp_path)
                analyze = True

    # MICROPHONE
    else:
        st.write("Record from microphone")

        audio = audiorecorder("🎤 Start Recording", "⏹ Stop Recording")

        if len(audio) > 0:
            audio.export("live_audio.wav", format="wav")
            st.audio("live_audio.wav")

            if st.button("🚀 Analyze Recording"):
                transcript = transcribe_audio("live_audio.wav")
                text_label, text_conf = predict_text_sentiment(transcript)
                audio_label, audio_conf = predict_audio_sentiment("live_audio.wav")
                sarcasm, score = detect_sarcasm(
                    text_label, audio_label,
                    text_conf, audio_conf
                )
                analyze = True

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# CENTER PANEL
# =====================================================
with center:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Analysis Result")

    # waveform placeholder
    st.line_chart(np.random.randn(80,1))

    st.markdown("### 📝 Transcript")

    if transcript:
        st.info(transcript)
    else:
        st.info("Transcript will appear here...")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RIGHT PANEL
# =====================================================
with right:

    # Text Emotion
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Text Emotion")

    if text_label:
        st.markdown(
            f'<div class="result-box {get_class(text_label)}">{text_label.upper()}</div>',
            unsafe_allow_html=True
        )
    else:
        st.write("Waiting...")

    st.markdown('</div>', unsafe_allow_html=True)

    # Audio Emotion
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎧 Audio Emotion")

    if audio_label:
        st.markdown(
            f'<div class="result-box {get_class(audio_label)}">{audio_label.upper()}</div>',
            unsafe_allow_html=True
        )
    else:
        st.write("Waiting...")

    st.markdown('</div>', unsafe_allow_html=True)

    # Sarcasm
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("😏 Sarcasm Result")

    if sarcasm:
        st.markdown(
            f'<div class="result-box sar">{sarcasm}</div>',
            unsafe_allow_html=True
        )
        st.progress(int(score))
        st.write(f"### Score: {score}%")
    else:
        st.write("Waiting...")

    st.markdown('</div>', unsafe_allow_html=True)