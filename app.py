import streamlit as st
import cv2
import numpy as np
import librosa
from moviepy import VideoFileClip
import tempfile
import os

# ── CONFIGURATION ────────────────────────────────────────
st.set_page_config(page_title="AnimScore", page_icon="🎬", layout="centered")

# ── STYLE NUTRISCORE ─────────────────────────────────────
NUTRISCORE_COLORS = {
    "A": "#00813A",
    "B": "#85BB2F",
    "C": "#FECB02",
    "D": "#EF7D00",
    "E": "#E63312",
}

def nutriscore_badge(lettre):
    color = NUTRISCORE_COLORS.get(lettre, "#999999")
    return f"""
    <div style="
        background-color: {color};
        color: white;
        font-size: 48px;
        font-weight: bold;
        width: 80px;
        height: 80px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: auto;
        font-family: Arial;
    ">{lettre}</div>
    """

# ── BORNES ───────────────────────────────────────────────
BORNES = {
    "cuts_par_minute"   : (0, 30),
    "flicker"           : (0, 1.5),
    "saturation_moyenne": (0, 255),
    "ecart_dynamique"   : (0, 50),
    "onset_rate"        : (0, 320),
    "variabilite_onset" : (0, 30),
    "ratio_silence"     : (0, 1),
}

def normaliser(valeur, min_val, max_val):
    return max(0.0, min(100.0, (valeur - min_val) / (max_val - min_val) * 100))

def attribuer_lettre(score):
    if score >= 80:   return "A"
    elif score >= 60: return "B"
    elif score >= 40: return "C"
    elif score >= 20: return "D"
    else:             return "E"

# ── ANALYSE ──────────────────────────────────────────────
def analyser_video(video_path, progress_bar, status_text):

    # VIDÉO
    status_text.text("⏳ Analyse vidéo en cours...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    frame_precedente = None
    hist_precedent = None
    changements_de_plan = 0
    luminosites_brutes = []
    saturations = []
    SEUIL_SAD = 20
    SEUIL_HIST = 0.3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luminosites_brutes.append(np.mean(gris))

        if frame_count % 5 == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturations.append(np.mean(hsv[:, :, 1]))

        hist = cv2.calcHist([gris], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if frame_precedente is not None:
            diff = cv2.absdiff(gris, frame_precedente)
            sad = np.mean(diff)
            diff_hist = cv2.compareHist(hist, hist_precedent, cv2.HISTCMP_BHATTACHARYYA)
            if sad > SEUIL_SAD and diff_hist > SEUIL_HIST:
                changements_de_plan += 1

        frame_precedente = gris
        hist_precedent = hist
        frame_count += 1

        if frame_count % 100 == 0:
            progress_bar.progress(min(frame_count / total_frames * 0.6, 0.6))

    cap.release()

    duree_minutes = (frame_count / fps) / 60
    cuts_par_minute = changements_de_plan / duree_minutes
    flicker = np.mean(np.abs(np.diff(luminosites_brutes)))
    saturation_moyenne = np.mean(saturations)

    # AUDIO
    status_text.text("⏳ Analyse audio en cours...")
    progress_bar.progress(0.65)

    video = VideoFileClip(video_path)
    import tempfile
    audio_path = tempfile.mktemp(suffix=".wav")
    video.audio.write_audiofile(audio_path, logger=None)
    video.close()

    audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    os.remove(audio_path)

    progress_bar.progress(0.75)

    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    onset_rate = len(onsets) / (len(audio) / sr / 60)

    rms = librosa.feature.rms(y=audio)[0]
    rms_db = 20 * np.log10(rms + 1e-6)
    ecart_dynamique = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

    fenetre = sr * 10
    nb_fenetres = len(audio) // fenetre
    if nb_fenetres > 1:
        fenetres = [audio[i*fenetre:(i+1)*fenetre] for i in range(nb_fenetres)]
        rates = [len(librosa.onset.onset_detect(y=f, sr=sr)) for f in fenetres]
        variabilite_onset = float(np.std(rates))
    else:
        variabilite_onset = 0.0

    ratio_silence = float(np.sum(np.abs(audio) < 0.01) / len(audio))

    progress_bar.progress(1.0)
    status_text.text("✅ Analyse terminée !")

    return {
        "duree_minutes"     : round(duree_minutes, 1),
        "cuts_par_minute"   : round(cuts_par_minute, 1),
        "flicker"           : round(flicker, 2),
        "saturation_moyenne": round(float(saturation_moyenne), 1),
        "onset_rate"        : round(onset_rate, 1),
        "ecart_dynamique"   : round(ecart_dynamique, 1),
        "variabilite_onset" : round(variabilite_onset, 1),
        "ratio_silence"     : round(ratio_silence, 2),
    }

# ── INTERFACE ────────────────────────────────────────────
st.title("🎬 AnimScore")
st.caption("Analyse l'excitation visuelle et sonore d'un dessin animé")
st.divider()

fichier = st.file_uploader("Choisir une vidéo", type=["mp4", "mkv", "avi"])

if fichier is not None:
    st.video(fichier)
    st.divider()

    if st.button("▶ Lancer l'analyse", type="primary", use_container_width=True):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(fichier.read())
            tmp_path = tmp.name

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            resultats = analyser_video(tmp_path, progress_bar, status_text)
            os.remove(tmp_path)

        except MemoryError:
            os.remove(tmp_path)
            st.error("❌ Mémoire insuffisante pour analyser cette vidéo. Essayez avec un fichier plus court.")
            st.stop()

        except Exception as e:
            os.remove(tmp_path)
            st.error(f"❌ Erreur pendant l'analyse : {str(e)}")
            st.stop()

        # SCORES
        n = {k: normaliser(resultats[k], *BORNES[k]) for k in BORNES}

        score_visuel = round(
            0.40 * (100 - n["cuts_par_minute"]) +
            0.40 * (100 - n["flicker"]) +
            0.20 * (100 - n["saturation_moyenne"]), 1)

        score_sonore = round(
            0.45 * n["ecart_dynamique"] +
            0.35 * (100 - n["onset_rate"]) +
            0.15 * (100 - n["variabilite_onset"]) +
            0.05 * n["ratio_silence"], 1)

        score_global = round((score_visuel + score_sonore) / 2, 1)

        lettre_v = attribuer_lettre(score_visuel)
        lettre_s = attribuer_lettre(score_sonore)
        lettre_g = attribuer_lettre(score_global)

        # AFFICHAGE SCORES
        st.divider()
        st.subheader("🎯 Scores")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Visuel**")
            st.markdown(nutriscore_badge(lettre_v), unsafe_allow_html=True)
            st.metric("", f"{score_visuel} / 100")
        with col2:
            st.markdown("**Sonore**")
            st.markdown(nutriscore_badge(lettre_s), unsafe_allow_html=True)
            st.metric("", f"{score_sonore} / 100")
        with col3:
            st.markdown("**Global**")
            st.markdown(nutriscore_badge(lettre_g), unsafe_allow_html=True)
            st.metric("", f"{score_global} / 100")

        # AFFICHAGE MÉTRIQUES
        st.divider()
        st.subheader("📊 Détail des métriques")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Durée", f"{resultats['duree_minutes']} min")
            st.metric("Cuts / minute", resultats["cuts_par_minute"])
            st.metric("Flicker", resultats["flicker"])
        with col5:
            st.metric("Saturation", resultats["saturation_moyenne"])
            st.metric("Onset rate", resultats["onset_rate"])
        with col6:
            st.metric("Écart dynamique", f"{resultats['ecart_dynamique']} dB")
            st.metric("Variabilité onset", resultats["variabilite_onset"])
            st.metric("Ratio silence", resultats["ratio_silence"])