# When Memories Compete: How Similarity Affects Memory Retrieval in Hopfield Networks

**Blessing Ndeh** · Computational Neuroscience, Fall 2026 (PSYC 40 / COSC 16 / COGS 21)

A classical Hopfield network stores real handwritten digits. The project asks how the **similarity** (pixel overlap) between stored memories, and their **number**, affect whether each memory can be retrieved *exactly*, and what the network does when retrieval fails.

**➡️ Start here: [`final_report.ipynb`](final_report.ipynb)** (the project documentation: data, objective, model, metrics, results and discussion).

## Main findings
1. With 3 stored images, exact stability falls from 100% to 0% as the images' mean pixel overlap rises from about 0.44 to 0.56. Noisy cues fail at slightly lower overlap.
2. Every failed three-memory retrieval (22,599 of 22,599) ended in the pixel-wise **majority vote** of the stored images, never in another stored memory.
3. A stability rule derived from the Hebbian weights explains which memories survive. Its assumptions were checked for all 540 sampled images, and it predicts 540 of 540.

Raw pixel overlap is **not** perceived visual similarity: much of it comes from shared white background. Details are in the report.

## Repository layout
| Path | Contents |
|---|---|
| `final_report.ipynb` | Project documentation (reads saved results; runs in seconds) |
| `src/optdigits.py` | Parser for the 32×32 Optdigits bitmap file |
| `src/hopfield.py` | Hopfield network: Hebbian storage, asynchronous recall, energy, corruption, overlap |
| `notebooks/01_dataset_exploration.ipynb` | File format, image checks, sample digits, counts per digit |
| `notebooks/02_patterns_and_similarity.ipynb` | ±1 representation, overlap measure, background bias |
| `notebooks/03_hopfield_verification.ipynb` | Implementation checks (weights, stability, noise repair, random-pattern control) |
| `notebooks/04_similarity_experiment.ipynb` | Experiment 1: groups of 5 memories |
| `notebooks/05_group_size_pilot.ipynb` | Pilot: groups of 2, 3 and 4 memories |
| `notebooks/06_three_memory_experiment.ipynb` | Experiment 2 (main): 180 groups of 3 memories, statistics, mathematical validation, overlap decomposition |
| `scripts/make_report_figures.py` | Rebuilds the report figures from saved results |
| `results/` | Saved results (CSV), configs with random seeds, bit-packed final states, figures |
| `reports/` | Detailed interpretation (`step5_interpretation.md`), human-memory comparison, report outline |
| `data/optdigits-orig.cv` | Dataset (see Data and licence below) |

## How to run
Requires Python 3.12 (tested with Anaconda, Python 3.12.4).

```bash
pip install -r requirements.txt
jupyter notebook final_report.ipynb      # or open it in VS Code / JupyterLab
```

- Every experiment notebook **loads saved results by default**. To rerun an experiment from scratch, set `RECOMPUTE = True` in its settings cell. Experiment 2 takes about 10 minutes, Experiment 1 about 10, and the pilot about 3.
- All randomness uses fixed seeds, recorded in `results/*_config.json`, so reruns reproduce the saved results.
- To rebuild the report figures: `python scripts/make_report_figures.py`

## Data and licence
*Optical Recognition of Handwritten Digits*, original 32×32 bitmaps (`optdigits-orig.cv`, decompressed from `optdigits-orig.cv.Z`).
Alpaydin, E., & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49.
The dataset is licensed under CC BY 4.0 and is included here with attribution.
