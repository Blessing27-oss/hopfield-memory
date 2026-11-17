# Final Submission Outline (restructured to the course format; not the report)

**Working title:** *When Memories Compete: How Similarity Affects Memory Retrieval in Hopfield Networks*

**Status:** for approval. Drafting waits until you approve.

## Course requirements (as provided)

**Five required parts:**
1. Data description and visualisation, including any reformatting or preprocessing.
2. Objective.
3. Model description: why this model, and what it is specialised for, at a high level with no heavy maths.
4. Metric.
5. Interpretation and discussion: what the results show, limitations, possible improvements.

**Tone:** "doesn't have to be a full-on research paper… help us understand your thought processes."

**Format:** anything downloadable or compilable, e.g. a Jupyter notebook, Colab or GitHub.

## What changes compared with the paper-style outline
- **The five required headings become the backbone.** The paper sections (abstract, introduction, separate mathematical analysis, human-memory section) are folded into them or moved to linked material.
- **The thought process is shown on purpose:** why the design changed (5-memory floor effect → load pilot → 3-memory main experiment), and which checks caught problems (the int8 overflow bug; the corrected per-image overlap formula; the stability rule being conditional).
- **Maths stays light in the main text.** The stability rule is explained in words plus one formula, with its assumptions. The full derivations stay in notebook 06 §12, linked.
- **The human-memory comparison is shortened** to a motivation paragraph and a short discussion paragraph, linking to `reports/human_memory_comparison.md`.
- **Unchanged:** the three central findings, all accuracy distinctions, and the five figures.

---

## Proposed submission package

```
hopfield-memory/
├── README.md                      ← overview, how to run, file map
├── final_report.ipynb             ← THE documentation (5 required sections)
├── requirements.txt               ← numpy, pandas, scipy, matplotlib, nbformat (versions pinned from this environment)
├── data/optdigits-orig.cv         ← included (CC BY 4.0, with attribution)
├── src/  hopfield.py, optdigits.py
├── scripts/make_report_figures.py
├── notebooks/ 01–06               ← full experiments, with outputs saved
├── results/  CSVs, configs, figures/
└── reports/  step5_interpretation.md, human_memory_comparison.md
```

- **`final_report.ipynb`:** mostly Markdown, with embedded figures. A few small code cells load the saved CSVs to show key tables, so readers can see where numbers come from without re-running the 15-minute experiment.
- **Delivery:** GitHub repository, or a zip of the same folder. **Your choice; see Questions.**

---

## `final_report.ipynb`: section plan

### 0. Summary (short, at the top)
- One paragraph: question, approach, and the three findings, F1–F3.
- **F1.** Higher raw pixel overlap is associated with lower clean stability and lower exact retrieval. With 3 memories, the 50% point is at mean overlap 0.497 (clean) and 0.477 (20% noise).
- **F2.** All 22,599 failed three-memory retrievals ended exactly in the majority-vote mixture, never in another stored memory.
- **F3.** A stability condition from the Hebbian weights explains which memories survive. Its assumptions were checked for all 540 sampled images, and it matches 540/540.
- **One caution:** "overlap" here is raw pixel overlap, much of it shared blank background, not perceived similarity.

### 1. Objective
- **What we want to know:** when a classical Hopfield network stores several real images that resemble each other, does it still retrieve each one exactly? How does that depend on (a) how similar the stored images are and (b) how many are stored?
- **Neuroscience motivation**, one paragraph: the hippocampus is thought to use *pattern separation* (DG) and *pattern completion* (CA3 recurrent network) to keep similar memories apart (Yassa & Stark 2011). In humans, more similar lures are harder to tell apart (Reagh & Yassa 2014). A Hopfield network is a completion-only memory with no separation stage, so it shows what happens when similar memories are stored *without* being separated.
- **Thought process:** the goal was refined along the way. Five memories almost never worked, so the main question was narrowed to three memories, where outcomes actually vary.

### 2. Data description and visualisation
**Dataset**
- UCI *Optical Recognition of Handwritten Digits*, original 32×32 bitmaps (`optdigits-orig.cv`; Alpaydin & Kaynak 1998).
- File format: a 21-line header, then 946 blocks of 32 rows of 0/1 characters followed by a label line.
- 946 images; 85–114 per digit; about 30.5% ink pixels.

**Visualisations (existing)**
- Sample digits (appendix `01_…cell09`).
- Examples per digit (`01_…cell11`).
- Per-pixel ink map showing that the edges are almost always blank (`02_…cell18`).

**Preprocessing (and why)**
1. Parse the file and validate it (every row exactly 32 binary characters; counts match the header).
2. Convert 0/1 → −1/+1 and flatten to 1,024 "neurons". The Hopfield model uses ±1 units.
3. **Typical-image filter:** keep the more typical half of each digit (476/946). *Why:* without it, the highest-similarity groups were thick, unreadable blobs. Results therefore apply to this subset.
4. **Similarity measure:** overlap $m = \frac{1}{N}\sum_i \xi_i\eta_i$, computed in float64. *Thought process:* the int8 version silently overflowed, giving m(x,x) = 0; this was caught and fixed.
5. **What the overlap contains:**
   - shared blank border pixels add about 0.35 to every overlap;
   - random ink placement would give ≈0.14, versus 0.49 observed;
   - the ordering still tracks digit shape (ρ = 0.965 with ink-only overlap).
   - **Fig. 5** `report_fig5_overlap_decomposition.png`.

### 3. Model description (high level)
- **Model:** classical Hopfield network (Hopfield 1982).
  - Memories are stored with the Hebbian rule ("neurons active together connect").
  - Recall starts from a cue and updates neurons one at a time until nothing changes. The network settles into a stable state (an *attractor*).
- **Why this model:**
  - It is the standard model of *content-addressable* (auto-associative) memory: it recovers a whole memory from a partial or noisy cue.
  - It is the classic abstraction of CA3-style pattern completion.
  - Its behaviour can be analysed exactly, so failures can be explained, not just observed.
- **What it is specialised for, and its known weak spot:**
  - It works well for *uncorrelated* patterns, roughly 0.138·N ≈ 141 at N = 1024.
  - It also has spurious "mixture" states (Amit, Gutfreund & Sompolinsky 1985).
  - Real images are correlated, and that is exactly what we test.
- **Implementation checks** (short):
  - weights symmetric with zero diagonal;
  - a single image is stable and repairs itself from up to 30% noise; energy never increases;
  - 5 random patterns are all retrieved.
  - These show that later failures come from the data, not bugs. (notebook 03)

### 4. Metric
- **Primary: clean stability.** The fraction of stored images that are *exact* fixed points (start at the memory; does it stay exactly the same?).
- **Second: exact retrieval accuracy.** The fraction of trials where the network, started from the memory with 5/10/20% of pixels flipped, ends **pixel-for-pixel** on the original. 20 trials per image per level.
- **Why exact, not "close enough":** failed trials still end at an overlap of 0.58–0.93 with the target. A lenient threshold would count blends as successes, and the most similar groups would look *best*. Every end state was either exactly the target or exactly the mixture, so exact scoring is unambiguous here.
- **Also recorded (descriptive):** outcome category (correct / wrong memory / inverted / other state / not converged); final overlap; whether different memories end in the same state.
- **How results are summarised:**
  - per group of memories (the unit of analysis);
  - Spearman correlation with overlap;
  - descriptive logistic curves giving a "50% point", fitted only after pre-set checks.

### 5. Results, interpretation and discussion

**5.1 How the design evolved (thought process)**
- **Experiment 1 (5 memories, 84 groups):** clean stability 0.5%; exact retrieval ≈0.3%. That is a floor effect: there was nothing to compare.
- **Pilot (2–4 memories):** 2 memories always stable; 4 rarely; 3 gave real variation.
- **Experiment 2 (3 memories):** 180 groups, sampled from all 12.9M possible groups, denser in the transition region.
- **Fig. 2** `report_fig2_memory_load.png`: supporting evidence only, because the groups were sampled differently.

**5.2 F1: higher overlap, less stability and retrieval**
- Stability is 1.00 below mean overlap 0.44, falls 0.84 → 0.07 across 0.44–0.56, and is 0 above 0.60.
- Spearman ρ −0.93 to −0.96. More noise moves the 50% point lower: 0.497 → 0.477.
- The result holds without reused images and without digit 4.
- **Fig. 3** `report_fig3_overlap_retrieval.png`.

**5.3 F2: failures merge, they don't swap**
- 22,599/22,599 failures end exactly in the pixel-wise majority vote of the three images.
- 0 wrong-memory outcomes. Above overlap 0.60, all three memories collapse into that one state.
- **Fig. 1** `report_fig1_retrieval_examples.png`.

**5.4 F3: why (light maths, with assumptions)**
- In words: a stored image survives only if it differs from the other two on at least 514 of its 1,024 pixels in total. Equivalently, $m_{ab}+m_{ac} < 1 - 3/N$.
- This holds **under assumptions**:
  - A1: some pixel where both other images disagree with it;
  - A2: its two overlaps not too different;
  - A3: their sum not very negative;
  - A4: N even, so no ties.
- All four were verified for the 540 sampled images, and the rule matches 540/540.
- It is **not** a general theorem: counterexamples exist when the assumptions fail.
- On average, an image's two overlaps add up to 2 × the group mean. That is why the change sits near 0.5 *for this sample*; it is not a universal threshold.
- From a clean start, an unstable image provably slides into the mixture. For noisy starts this was observed, not proven.
- Images that are only barely stable are the ones lost to noise. That pattern is observed; the explanation is an interpretation.
- **Fig. 4** `report_fig4_stability_analysis.png`. Full derivations: notebook 06 §12.

**5.5 What it means**
- Correlation, not the classical capacity limit, is what limits this network: about 2–3 digit images, versus ≈141 random patterns.
- Interference shows up as *merging into what the memories share*.
- **Neuroscience link (conceptual only):** this illustrates why the brain might need a separation stage (DG) before completion (CA3).
  - It does not show that the brain computes majority votes.
  - Human similarity ratings, lure discrimination and fMRI cannot be numerically compared with exact reconstruction.
  - Details: `reports/human_memory_comparison.md` (Bakker et al. 2008; Lacy et al. 2011; Reagh & Yassa 2014).

**5.6 Limitations, and how to improve**

| Limitation | Possible improvement |
|---|---|
| Similarity = raw pixel overlap, inflated by shared background; not perceived similarity | A learning rule or coding that discounts background (e.g. centred or sparse patterns) |
| Typical-image subset; only 3 different digits per group | Include atypical images; same-digit groups |
| Digit identity and overlap vary together; real images can't have overlap set independently, so association, not causation | Synthetic patterns with controlled overlap |
| Image reuse makes confidence intervals optimistic | Resampling that accounts for shared images |
| Load comparison uses different sampling designs | A matched-design load experiment |
| Five-memory failure states not characterised | Save and classify the 5-memory end states (Step B) |
| Classical model only: binary units, one-shot learning, no separation stage | Add a pattern-separation preprocessing stage; sparse coding |

### 6. References (verified)
- Alpaydin & Kaynak (1998) [Dataset], UCI, doi:10.24432/C50P49
- Hopfield (1982), doi:10.1073/pnas.79.8.2554
- Amit, Gutfreund & Sompolinsky (1985), doi:10.1103/PhysRevA.32.1007
- Yassa & Stark (2011), doi:10.1016/j.tins.2011.06.006
- Bakker et al. (2008), doi:10.1126/science.1152882
- Lacy et al. (2011), doi:10.1101/lm.1971111
- Reagh & Yassa (2014), doi:10.1073/pnas.1411250111

Full entries are in `reports/human_memory_comparison.md` §7. Citation style is not specified by the course, so a simple consistent author–year style is proposed.

### Appendix (links rather than content)
- Notebook guide: 01 data, 02 similarity, 03 model checks, 04 Experiment 1, 05 pilot, 06 Experiment 2 and the maths.
- Sampling diagnostics and appendix figures (listed in the previous outline version; files in `results/figures/` and `results/figures/appendix/`).

---

## Accuracy rules carried into the draft
- Raw pixel overlap ≠ perceived similarity; background is a large share of it.
- The ≈0.5 transition is specific to this sample and explained by the stability condition; it is not universal.
- The simplified rule always appears with A1–A4.
- Mixture convergence is proven for clean starts only; observed for noisy starts.
- No mixture claim for five memories.
- Clean stability and noisy retrieval are reported separately.
- No numerical equivalence with human data; no causal claims.
- Observations, deductions and interpretations are labelled.

## Preparation still needed before submission
1. Re-run notebook 05 in VS Code so its outputs are saved (the others already have outputs).
2. Write `README.md` and `requirements.txt`.
3. Confirm the dataset may be included in the repository (CC BY 4.0 allows it with attribution).

## Questions before drafting
1. **Delivery:** a GitHub repository, or a zip/folder with the notebook? If GitHub, public or private?
2. **Length:** the course says "not a full research paper". Is about 6–10 screens for `final_report.ipynb` right, with details linked to the experiment notebooks?
3. **Title block:** your name, course name and date?
