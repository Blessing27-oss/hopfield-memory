# Connecting the Hopfield Results to Human Memory Research

*Project: When Memories Compete: How Similarity Affects Memory Retrieval in Hopfield Networks. Step 6: comparison with human-memory studies. Draft for review.*

**Computational basis:** notebooks 04–06 and `reports/step5_interpretation.md`. No new simulations were run.
**Verification:** every citation was checked against PubMed Central full text, the publisher's page or bibliographic records (§7). Study details below come only from those sources. Anything that could not be checked is marked **[not verified]**.

---

## 1. The model results being compared

| Result | Value | Source |
|---|---|---|
| Groups of 3 memories: clean stability vs. group mean overlap $\bar m$ | 50% point at $\bar m = 0.497$ (95% CI 0.493–0.502). Spearman ρ = −0.93 | notebook 06, §6–8 |
| Exact retrieval at 5 / 10 / 20% corruption | 50% points at 0.490 / 0.485 / 0.477. ρ = −0.95 to −0.96 | notebook 06, §7–8 |
| How retrieval fails | 22,599 / 22,599 failed trials end **exactly** in the pixel-wise majority-vote ("mixture") state. 0 trials end in another stored image | notebook 06, §12.1 |
| Group size | clean stability: 2 memories 100%, 3 memories 51.5%, 4 memories 6%, 5 memories 0.5% | notebooks 04–06 |
| Mechanism (exact for our 540 stored images, under stated assumptions) | image *a* is stable ⇔ $m_{ab}+m_{ac} < 1 - 3/N$ ⇔ it differs from the other two on ≥ 514 pixels in total | notebook 06, §12.3, 13.1 |
| What "overlap" contains | about 0.35 of every pairwise overlap comes from shared blank border pixels. Randomly placed ink would give about 0.14 instead of the observed 0.49 | notebook 06, §13 |

---

## 2. The human studies

### 2.1 Bakker, Kirwan, Miller & Stark (2008), *Science*
**Citation:** Bakker A, Kirwan CB, Miller M, Stark CEL. Pattern separation in the human hippocampal CA3 and dentate gyrus. *Science* 2008;319(5870):1640–1642. doi:10.1126/science.1152882 (PMCID PMC2829853).

- **Question:** Does the human CA3/dentate gyrus (DG) region show signals consistent with *pattern separation*, while other medial-temporal regions show *pattern completion*?
- **Participants:** 18 adults. Ages not reported on the source page.
- **Task:** incidental encoding. Participants judged pictures of everyday objects as indoor or outdoor; they were never asked to remember them. Trial types:
  - first presentations;
  - exact repetitions;
  - **lures**: slightly different pictures of an object seen earlier, from 144 sets of similar pictures.
- **Measurements:** high-resolution fMRI (1.5 mm voxels); field strength **[not verified]**. The key measure is a bias score, (first − lure)/(first − repetition):
  - near 0: the region treats a lure like a new item (separation-like);
  - near 1: it treats a lure like a repeat (completion-like).
  - No behavioural memory measure was collected.
- **Main findings:**
  - CA3/DG bias score **0.15**: lures were treated almost like new items.
  - CA1, subiculum, entorhinal and parahippocampal regions: **0.59–0.83**, toward completion; the authors call this pattern "somewhat mixed".
  - CA3 and DG could not be separated at this resolution and were analysed together.

### 2.2 Lacy, Yassa, Stark, Muftuler & Stark (2011), *Learning & Memory*
**Citation:** Lacy JW, Yassa MA, Stark SM, Muftuler LT, Stark CEL. Distinct pattern separation related transfer functions in human CA3/dentate and CA1 revealed using high-resolution fMRI and variable mnemonic similarity. *Learning & Memory* 2011;18(1):15–18. doi:10.1101/lm.1971111 (PMCID PMC3023966).

- **Question:** How does activity in CA1 and in CA3/DG change as the **amount** of change in the input (lure similarity) varies?
- **Participants:** 18 (6 male, 12 female); mean age 20.8 (SD 2.8).
- **Task:** the same incidental indoor/outdoor design, with first presentations, repetitions and lures. Lures were rated separately for **mnemonic similarity**, then split at the median into high- and low-similarity sets; a reliability check used four bins. How the ratings were made is described only in the supplementary material **[not verified]**.
- **Measurements:** 3T high-resolution fMRI (1.5 mm isotropic); response amplitudes compared across trial types and regions.
- **Main findings:**
  - **CA1 responded in a graded way** as the input changed from none (repetition) to small (high-similarity lure) to large (new item).
  - **CA3/DG responded in a step-like way**: activity rose sharply even for small changes.
  - For high-similarity lures, CA3/DG activity was higher than CA1 (t(17) = 2.22, p < .05); there was no difference for low-similarity lures.
  - The paper reports that behaviour shifted gradually from completion-like to separation-like responses, but the details are in the supplement **[not verified]**.

### 2.3 Reagh & Yassa (2014), *PNAS*
**Citation:** Reagh ZM, Yassa MA. Object and spatial mnemonic interference differentially engage lateral and medial entorhinal cortex in humans. *Proc Natl Acad Sci USA* 2014;111(40):E4264–E4273. doi:10.1073/pnas.1411250111 (PMCID PMC4210036).

- **Question:** Do the lateral and medial entorhinal cortex (LEC, MEC) respond differently to **object** versus **spatial** interference? How does the DG/CA3 region respond to both?
- **Participants:** 22 recruited; 18 analysed (10 female, 8 male; mean age 22.31, SD 1.27). Four were excluded: two for poor performance, two for incomplete brain coverage.
- **Task:**
  - *Incidental encoding:* 260 object images, each shown at one of 31 positions on a hidden 7×5 grid.
  - *Surprise test:* repeated items, object lures, spatial lures and new foils. Responses: "No Change", "Object Change", "Location Change" or "New".
  - *Spatial lures:* the object moved one grid space (high similarity) or two (low similarity).
  - *Object lures:* similarity levels taken from an earlier rating study (n > 100).
  - 36 high- and 36 low-similarity trials per lure type.
- **Measurements:** 3T fMRI (1.5 mm isotropic). Lure discrimination index (LDI) = p("change" | lure) − p("change" | new foil).
- **Main findings:**
  - **High-similarity lures were harder to identify correctly than low-similarity ones** (F(1,17) = 184.61, p < .001).
  - LEC and perirhinal cortex were more engaged by object interference; MEC and parahippocampal cortex by spatial interference. Region × domain interaction: F(1,17) = 20.88, p < .001.
  - **DG/CA3 was engaged by both object and spatial lures.** It was the only region whose activity correlated with behaviour: spatial LDI R² = 0.284, p = .023; object LDI R² = 0.211, p = .055 (marginal).

### 2.4 Theoretical background (not empirical comparisons)
- **Yassa & Stark (2011), review in *Trends in Neurosciences*.**
  - Pattern separation means similar representations are stored "in a distinct, non-overlapping (orthogonalized) fashion". Pattern completion means filling in "incomplete or degraded representations … based on previously stored representations".
  - The review stresses that "pattern separation is a neural computation", not a behaviour.
  - It describes the DG as forcing pattern-separated representations onto CA3, and CA3's recurrent connections as an auto-associative network that supports completion "via attractor dynamics".
- **Hopfield (1982)** introduced the network we use.
- **Amit, Gutfreund & Sompolinsky (1985)** showed that such networks also have additional stable states that "correspond to specific mixings of the embedded patterns". These are the classical mixture states our failed trials converge to.

---

## 3. How the findings relate (conceptually)

1. **More similarity, more interference. The direction agrees; the measures do not.** In Reagh & Yassa (2014), lures closer to a studied item were harder to identify as changed. In our model, groups with higher raw pixel overlap had lower exact stability and retrieval.
   Both show that **similarity between stored and competing information makes them harder to keep apart**. But the dependent measures differ entirely: a behavioural lure judgement in humans, versus the exact end state of a network started from a stored or corrupted pattern in the model.
2. **Our network is a completion-only system with no separation stage.** The literature (Yassa & Stark 2011; Bakker et al. 2008) gives DG/CA3 a role in making similar inputs *less* similar before or during storage, with CA3's recurrent network then completing patterns.
   Our Hopfield network stores the raw patterns directly: there is nothing that decorrelates them. Our results can therefore be read as a demonstration of **what an auto-associative store does when similar inputs are *not* pattern-separated**. Interference rises with input overlap, and the memories eventually merge.
   This supports the *computational motivation* for pattern separation. It does **not** test whether the human DG performs it.
3. **A step-like change in the model and step-like CA3/DG responses: a similarity in shape only.** Lacy et al. (2011) found a step-like CA3/DG response and a graded CA1 response as input similarity changed.
   Our clean-stability curve is also sharp: it falls from ~1 to ~0 between $\bar m$ ≈ 0.44 and 0.56, and at the level of single images it is an exact threshold.
   But these are different things: a BOLD signal across a region versus a deterministic stability rule about one network's fixed points. The resemblance in shape should **not** be presented as evidence that they share a mechanism.
4. **Mixture attractors as a concrete form of "merging" interference.** When our network fails, it never retrieves a *different* stored memory; it settles into a state that blends features of all three.
   This makes one form of interference concrete: competing traces collapse into what they have in common, instead of one replacing another.
   The human lure tasks do not measure internal states this way. A human error (calling a lure "old" or "no change") is a behavioural response, often described as completion toward the studied item. So the mixture result is a **model-level** observation with no direct counterpart in these data.

---

## 4. Human similarity vs. our pixel-overlap measure

| | Human studies | Our model |
|---|---|---|
| What "similarity" is | **Mnemonic or perceptual** similarity between a lure and a studied item: independent human ratings (Lacy; Reagh, object lures) or spatial displacement on a grid (Reagh, spatial lures) | **Bipolar pixel overlap** $m = 1 - 2p_{dis}$ between 32×32 binary images: no perception, ratings or meaning involved |
| What it is a property of | the stimuli as people perceive them | the input code; about 0.35 of each value is shared blank background |
| Which comparison | a lure vs. **one** studied item | among the **stored** memories (3 per network) |
| What is measured | behaviour (lure discrimination) and region-level BOLD signals | the exact network state after recall |

**Important:** the hippocampus is not claimed to compute majority votes, and our overlap is not claimed to equal human similarity. The two studies do not measure the same thing.
Our own decomposition shows that raw overlap ranks image pairs almost the same way as ink-only shape overlap (ρ = 0.965). But *where* the transition happens is set mostly by shared background, a property of the ±1 code that human similarity ratings would not reflect.

---

## 5. Where the analogy helps, and where it breaks down

**Where it helps**
- **Shows how overlap causes interference:** in Hebbian auto-associative storage, each memory's stability depends on how much it overlaps with the others (exact rule in §1). This is a clear, testable version of the idea that similar memories compete.
- **Separates two kinds of failure:** *storage instability* (a memory cannot be retrieved even from a perfect cue) versus *failure to recover* (a stored memory is lost when the cue is degraded). The model can observe this split directly; behavioural tasks cannot.
- **Shows that the code matters:** overlap in the stored *representation*, not raw stimulus similarity, determines interference. That is the core argument for a decorrelating stage such as the DG (Yassa & Stark 2011).

**Where it breaks down**
- **No pattern separation:** the model has no DG-like orthogonalisation, no sparse coding and no separate CA3/CA1 regions. The human findings are mostly *about* those differences between regions.
- **Neurons and learning:** binary units, symmetric weights, one-shot Hebbian storage with no forgetting, consolidation or neurogenesis; 1,024 "neurons" with about 30% active. Hippocampal codes are thought to be much sparser.
- **Task:** our cue is a stored image with random pixel flips. The human tasks present a *meaningful* lure (a slightly different object, or the same object moved) and require a recognition decision.
- **Scoring:** exact pixel-for-pixel recovery, versus a behavioural choice or a BOLD response. These are on different scales and cannot be compared numerically.
- **Thresholds:** our 50% points (≈0.48–0.50) depend on the ±1 coding and the shared background. They are **not** predictions of human similarity thresholds.
- **Failure mode:** the model never retrieved the wrong stored memory. In the lure paradigms above, accepting a lure as the studied item is interpreted as completion-like responding (the framing in Bakker et al. 2008 and Yassa & Stark 2011), which is a different kind of error from merging into a blend.

---

## 6. Comparison table

| Topic | Model result | Human finding | Similarity | Limitation |
|---|---|---|---|---|
| Similarity and interference | Exact stability and retrieval fall as group overlap rises (ρ −0.93 to −0.96) | High-similarity lures harder to discriminate than low (F = 184.61; Reagh & Yassa 2014) | Same direction: more similarity, more interference | Different similarity measures, tasks and outcomes |
| Separation vs. completion | Completion-only network; similar inputs merge | CA3/DG separation-like (bias 0.15); CA1 and cortex completion-like (0.59–0.83) (Bakker et al. 2008) | The model shows what happens without a separation stage | Not a test of DG function; fMRI is region-level |
| Shape of the similarity response | Sharp change in stability between $\bar m$ ≈ 0.44 and 0.56; exact threshold per image | CA3/DG step-like, CA1 graded (Lacy et al. 2011) | Both non-linear | BOLD amplitude ≠ fixed-point stability; similar shape does not mean same mechanism |
| What failure looks like | 100% of failures end in the exact majority-vote mixture state; 0 wrong-memory | Lure errors = behavioural false "no change"/"old" responses | Both can be described as completion toward existing traces | The mixture is a model-internal state with no direct human measure; human errors favour the studied item |
| Load | 2 memories always stable; 4–5 rarely | Not tested in these studies | None | No human comparison for number of memories |
| Meaning of similarity | Raw pixel overlap; ~0.35 is shared background | Rated mnemonic or perceptual similarity; spatial displacement | Both are graded | Not the same construct |

---

## 7. References (all checked)

1. Amit DJ, Gutfreund H, Sompolinsky H. Spin-glass models of neural networks. *Physical Review A* 1985;32:1007. doi:10.1103/PhysRevA.32.1007. *(Checked against the APS abstract page.)*
2. Bakker A, Kirwan CB, Miller M, Stark CEL. Pattern separation in the human hippocampal CA3 and dentate gyrus. *Science* 2008;319(5870):1640–1642. doi:10.1126/science.1152882. *(Checked against PMC full text, PMC2829853.)*
3. Hopfield JJ. Neural networks and physical systems with emergent collective computational abilities. *Proc Natl Acad Sci USA* 1982;79(8):2554–2558. doi:10.1073/pnas.79.8.2554. *(Checked against the PMC listing and the Caltech repository record.)*
4. Lacy JW, Yassa MA, Stark SM, Muftuler LT, Stark CEL. Distinct pattern separation related transfer functions in human CA3/dentate and CA1 revealed using high-resolution fMRI and variable mnemonic similarity. *Learning & Memory* 2011;18(1):15–18. doi:10.1101/lm.1971111. *(Checked against PMC full text, PMC3023966.)*
5. Reagh ZM, Yassa MA. Object and spatial mnemonic interference differentially engage lateral and medial entorhinal cortex in humans. *Proc Natl Acad Sci USA* 2014;111(40):E4264–E4273. doi:10.1073/pnas.1411250111. *(Checked against PMC full text, PMC4210036.)*
6. Yassa MA, Stark CEL. Pattern separation in the hippocampus. *Trends in Neurosciences* 2011;34(10):515–525. doi:10.1016/j.tins.2011.06.006. *(Checked against PMC full text, PMC3183227.)*

**Not verified:**
- Bakker et al. (2008): scanner field strength and participant ages were not on the source page, so they are left out.
- Lacy et al. (2011): how the similarity ratings were made, and the behavioural results, are only in the supplementary material, which I did not access.
