import whisper
import asyncio
import aiohttp
import base64
import cv2
import numpy as np
import json
import os

# ── MODÈLE WHISPER ───────────────────────────────────────
whisper_model = whisper.load_model("small")

# ── EXTRACTION DES FRAMES CLÉS ───────────────────────────
def extraire_frames(video_path, nb_frames=8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, nb_frames, dtype=int)
    frames_b64 = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append(b64)
    cap.release()
    return frames_b64

# ── TRANSCRIPTION WHISPER ────────────────────────────────
def transcrire(audio_path):
    result = whisper_model.transcribe(audio_path, language="fr")
    return result["text"]

# ── APPEL API MAMMOUTH ───────────────────────────────────
async def appeler_llm(session, messages, model, api_key):
    url = "https://api.mammouth.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {"model": model, "messages": messages, "max_tokens": 1000}
    async with session.post(url, headers=headers, json=data) as response:
        result = await response.json()
        return result["choices"][0]["message"]["content"]

# ── ANALYSE PARALLÈLE ────────────────────────────────────
async def analyser_ia_async(video_path, audio_path, api_key):

    # Transcription
    transcription = transcrire(audio_path)

    # Frames
    frames_b64 = extraire_frames(video_path, nb_frames=8)

    async with aiohttp.ClientSession() as session:
        resultats = await asyncio.gather(

            # Claude — analyse narrative
            appeler_llm(session, [{
                "role": "user",
                "content": f"""Tu es expert en protection de l'enfance.
Analyse ce script de dessin animé et réponds UNIQUEMENT en JSON valide :
{{
  "themes_sensibles": [],
  "niveau_violence": 0,
  "niveau_peur": 0,
  "complexite_langage": 0,
  "valeurs_vehiculees": [],
  "note_narrative": "A",
  "resume": ""
}}
Les notes vont de A (très bien) à E (préoccupant).
Les niveaux sont de 0 à 10.

Script : {transcription[:3000]}"""
            }], "claude-sonnet-4-6", api_key),

            # Gemini — analyse visuelle frames
            appeler_llm(session, [{
                "role": "user",
                "content": [
                    {"type": "text", "text": """Tu es expert en protection de l'enfance.
Analyse ces frames de dessin animé et réponds UNIQUEMENT en JSON valide :
{
  "ambiance_generale": "calme",
  "violence_visuelle": false,
  "personnages_menaçants": false,
  "note_visuelle_ia": "A",
  "description": ""
}
Les notes vont de A (très bien) à E (préoccupant)."""},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
                      for f in frames_b64]
                ]
            }], "gemini-2.5-flash", api_key),

            # GPT — analyse vocabulaire
            appeler_llm(session, [{
                "role": "user",
                "content": f"""Tu es expert en protection de l'enfance.
Analyse le vocabulaire de ce script et réponds UNIQUEMENT en JSON valide :
{{
  "age_recommande": "3-6 ans",
  "mots_agressifs": [],
  "niveau_agressivite_verbale": 0,
  "note_vocabulaire": "A"
}}
Les notes vont de A (très bien) à E (préoccupant).
Les niveaux sont de 0 à 10.

Script : {transcription[:3000]}"""
            }], "gpt-4.1", api_key),
        )

        # Parser les JSON
        def parse_json(text):
            try:
                text = text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                return json.loads(text)
            except:
                return {"erreur": "Impossible de parser la réponse", "raw": text[:200]}

        return {
            "transcription" : transcription,
            "narrative"     : parse_json(resultats[0]),
            "visuelle_ia"   : parse_json(resultats[1]),
            "vocabulaire"   : parse_json(resultats[2]),
        }

# ── FONCTION PRINCIPALE ──────────────────────────────────
def analyser_ia(video_path, audio_path, api_key):
    return asyncio.run(analyser_ia_async(video_path, audio_path, api_key))
