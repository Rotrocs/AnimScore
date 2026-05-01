# AnimScore
AnimScore — A Nutriscore-inspired rating system for cartoons. Automatically analyzes cuts, motion, flicker, sound dynamics and gives each cartoon a score from A to E based on its visual and auditory stimulation level. Designed for researchers, pediatricians, and parents.

> Like Nutriscore, but for cartoons.

AnimScore is an open-source tool that objectively measures the visual and auditory excitation level of animated content. It automatically analyzes each video and assigns a score from **A** (calm) to **E** (very stimulating).

![AnimScore grades from A to E]
![Python]
![License]

---

## Why AnimScore ?

Current content rating systems (like age classifications) are **subjective** and decided by the broadcasters themselves. AnimScore provides an **objective, reproducible, and real-time** measurement based on scientifically grounded audio-visual metrics.

Unlike existing parental control tools, AnimScore goes beyond narrative content classification. 
It measures the **neurosensory stimulation level** of a cartoon through audio-visual analysis, and will soon incorporate script and dialogue analysis via AI to provide a comprehensive assessment of animated content.

---

## How it works

### Visual metrics
| Metric | Description |
|---|---|
| **Cuts per minute** | Scene change detection using SAD + Bhattacharyya histogram comparison |
| **Flicker** | Frame-by-frame luminosity variation |
| **Colour saturation** | Mean HSV saturation |

### Audio metrics
| Metric | Description |
|---|---|
| **Dynamic range** | P95–P05 RMS difference in dB |
| **Onset rate** | Sound attacks per minute (librosa) |
| **Onset variability** | Standard deviation of onset rate across 10s windows |
| **Silence ratio** | Proportion of near-silent frames |

### Scoring
Each metric is normalised against **absolute bounds** (independent of the corpus), then combined into two composite scores — visual and auditory — and a global score displayed as a Nutriscore-inspired **A–E grade**.

| Grade | Score | Meaning |
|---|---|---|
| 🟢 A | 80–100 | Very calm |
| 🟡 B | 60–80 | Calm |
| 🟠 C | 40–60 | Moderate |
| 🔴 D | 20–40 | Stimulating |
| 🔴 E | 0–20 | Very stimulating |

---

## Installation

```bash
git clone https://github.com/Rotrocs/AnimScore.git
cd animscore
pip install -r requirements.txt
```

---

## Usage

### Web interface
```bash
streamlit run app.py
```

Open your browser, upload a video file (MP4, MKV, AVI) and click **Analyse**.

### Batch analysis (full folder → CSV export)
```bash
python analyse.py --dossier "/path/to/videos" --export results.csv
```

The script automatically resumes if interrupted — already analysed videos are skipped.

---

## Project roadmap

- [x] Batch video analysis
- [x] Web interface (Streamlit)
- [ ] Public database of cartoon ratings
- [ ] AnimScore displayed on all animated content (TV, streaming)
- [ ] Live stream analysis
- [ ] Raspberry Pi physical device with LED display
- [ ] API endpoint

---

## Scientific background

This tool draws on research in **Music Emotion Recognition (MER)** and **media psychology** :

- Onset rate based on **spectral flux analysis**
- Dynamic range measurement inspired by **EBU R128** loudness standard
- Scene change detection using **SAD (Sum of Absolute Differences)** + Bhattacharyya histogram comparison
- Flicker measurement based on frame-by-frame luminosity variation

---

## Contributing

Contributions are welcome ! Feel free to :
- Open an issue to report a bug or suggest a feature
- Submit a pull request
- Share your results on a new cartoon corpus

---

## License

AnimScore is dual-licensed :

- **Non-commercial use** (personal, academic, public institutions) → [MIT Non-Commercial](LICENSE)
- **Commercial use** → [Commercial License](LICENSE-COMMERCIAL)

For commercial licensing inquiries, please open an issue or contact me directly via GitHub : [@Rotrocs](https://github.com/Rotrocs)

---

## Author

Developed by [@Rotrocs](https://github.com/Rotrocs)  
Tested on a corpus of 100+ French and international cartoons.
