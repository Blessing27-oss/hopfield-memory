# Validation and Interpretation of the Hopfield Similarity Experiments

*Project: When Memories Compete: How Similarity Affects Memory Retrieval in Hopfield Networks*
*Scope: the step-5 three-memory experiment (notebook 06), with step 4 (5 memories, notebook 04) and the group-size pilot (notebook 05) as supporting evidence. No new simulations were run. Every check below uses the saved results in `results/` and the unchanged model in `src/hopfield.py`.*

---

## 1. Summary

1. **Confirmed:** in the three-memory experiment, every trial ends in one of exactly two states: the **target image** (20,601 trials) or the **pixel-wise majority vote** of the three stored images (22,599 trials). No trial ended anywhere else.
2. **New, exact result:** with 3 memories, a stored image $\xi^a$ is stable **if and only if** $m_{ab} + m_{ac} < 1 - 3/N$, where $m_{ab}$ and $m_{ac}$ are its overlaps with the other two images.
   This rule predicts the stability of all 540 stored images correctly (540/540). It explains the clean-stability 50% point ($\bar m_{50} = 0.497$): averaged over a group, $m_{ab}+m_{ac}$ equals $2\bar m$.
3. **Exact deduction:** the majority (mixture) state is a fixed point in every group, including groups whose memories are perfectly stable. Starting from an unstable stored image, the network *must* end exactly in this mixture, whatever the update order.
4. **One correction to my earlier summary:** I wrote that the final overlap of failed trials "follows exactly $(1+\bar m)/2$". That holds only for the **average over the three images**. For a single image the exact value is $(1 + m_{ab} + m_{ac} - m_{bc})/2$, which differs from the group value by up to 0.27. The notebook text states this correctly, and no saved result is affected.
5. **One claim not verified:** I suggested the same mechanism explains the step-4 (5-memory) final-overlap pattern. Step-4 end states were not saved, so this is untested (see §6).

---

## 2. Part 1: Validation

### 2.1 Do all failed trials end exactly in the majority-vote state?

**Method.** All 43,200 saved end states (`step5_final_states.npy`) were unpacked and compared pixel by pixel with (i) the target image and (ii) $M = \operatorname{sign}(\xi^1+\xi^2+\xi^3)$ for that group. The notebook's outcome labels were not used.
`M` is always defined, because a sum of three ±1 values is never 0.

| Check | Result |
|---|---|
| End state = target (computed) vs. label `correct` | 20,601 = 20,601; they agree trial by trial |
| Failed trials ending exactly in $M$ | **22,599 / 22,599** |
| Every trial is either the target or $M$, never both and never neither | true |
| $M$ equal to one of the stored images in any group | never |
| By corruption level: failures at 0 / 5 / 10 / 20% | 5,240 / 5,551 / 5,752 / 6,056, all exactly $M$ |

**Verdict: confirmed.**

### 2.2 Final overlap with the target: the correct relationship

Sort each pixel by how the three images agree. Let $s_a$ be the fraction of pixels where image $a$ is the **minority** (the other two both have the opposite value). Then:

- $M$ differs from $\xi^a$ exactly on those minority pixels, so $m(M,\xi^a) = 1 - 2s_a$.
- From the pairwise overlaps, $m_{ab} + m_{ac} - m_{bc} = 1 - 4 s_a$.

Combining the two:

$$m(M,\xi^a) = \frac{1 + m_{ab} + m_{ac} - m_{bc}}{2} \qquad\text{(one image)}$$

$$\frac{1}{3}\sum_a m(M,\xi^a) = \frac{1+\bar m}{2}\qquad\text{(average over the group's three images)}$$

| Case | Holds? | Evidence |
|---|---|---|
| One image, per-image formula | exactly | 540/540 images |
| One image, $(1+\bar m)/2$ | **no** | mean deviation 0.058, max 0.268 |
| Single failed trial, $(1+\bar m)/2$ | **no** | mean deviation 0.049, max 0.188 |
| Mean over a group's three images | exactly | 180/180 groups |
| Group mean final overlap, fully merged groups | exactly | these form the straight line in `step5_final_overlap_and_attractors.png` |
| Group mean final overlap, partly failing groups | no | it mixes correct trials (overlap 1) with per-image mixture values |

### 2.3 Is the mixture a stable attractor, and why does Hebbian learning favour it?

**The basic equation.** With $W_{ij} = \frac1N\sum_\mu \xi^\mu_i\xi^\mu_j$ and $W_{ii}=0$, the input (local field) to neuron $i$ in state $s$ is exactly

$$h_i(s) = \sum_\mu \xi^\mu_i\, m_\mu(s) \;-\; \frac{k}{N}\, s_i ,$$

where $m_\mu(s)$ is the overlap of the current state with stored image $\mu$ and $k$ is the number of stored images.

**Is the mixture stable?** At $M$, the margin is $M_i h_i$, and the network stays put wherever it is positive:

- On pixels where all three images agree: $M_i h_i = \sum_\mu m_\mu(M) - 3/N > 0$, always.
- On pixels where image $a$ is the minority: $M_i h_i = \tfrac12(1 - m_{ab} - m_{ac} + 3m_{bc}) - 3/N$.

This stays positive for all realistic overlaps; it is positive even for uncorrelated patterns, where it equals ½. That matches the classical result that 3-pattern mixtures are spurious attractors.
**Observed:** $M$ is an exact fixed point in **180/180** groups. The smallest margin anywhere is 0.267.

**Is it also lower in energy?** $E(M) < E(\xi^a)$ (averaged over the three images) in **180/180** groups, including low-overlap groups where every memory is perfectly stable.
So **being lower in energy does not decide retrieval.** What decides it is whether each stored image is *locally* stable.

**Why Hebbian learning favours the mixture.** The weights are a sum of one term per stored image.
- On any pixel where two images agree, the mixture gets support from two of the three terms.
- A stored image, on its own minority pixels, gets support only from its own term and opposition from the other two.

The more the images overlap, the stronger that opposition becomes. Shared structure, such as common background and common ink regions, is counted once per memory, so it adds up into a strong "consensus" component.

### 2.4 The exact stability rule for 3 memories

Set $s = \xi^a$:

$$\xi^a_i h_i = 1 - \tfrac{3}{N} + (\xi^a_i\xi^b_i)\,m_{ab} + (\xi^a_i\xi^c_i)\,m_{ac}.$$

- On pixels where $\xi^a$ agrees with at least one other image, this is $\ge 1 - 3/N - |m_{ab}-m_{ac}| > 0$, so these pixels are always stable.
- On the image's own minority pixels, it is $1 - 3/N - m_{ab} - m_{ac}$.

Every image has minority pixels (at least 37, median 126), so:

$$\boxed{\;\xi^a \text{ is a fixed point} \iff m_{ab} + m_{ac} < 1 - 3/N\;}$$

Overlaps are multiples of $2/N$, so a tie is impossible. **Observed: the rule agrees with the measured stability for 540/540 images.**

> **Correction (step A):** The boxed rule is **not** a general theorem for any three patterns. The general, exact condition is: every pixel type that actually occurs must have $1 - 3/N + \sigma_b m_{ab} + \sigma_c m_{ac} \ge 0$.
> The boxed form follows only if three assumptions hold:
> - **A1:** the image has at least one minority pixel.
> - **A2:** $|m_{ab}-m_{ac}| < 1-3/N$.
> - **A3:** $m_{ab}+m_{ac} > -(1-3/N)$.
>
> Ties are impossible only because N is even. Notebook 06 §12.3 builds synthetic counterexamples in both directions and confirms that A1–A4 hold for all 540 of our images. Every image with $m_{ab}+m_{ac}<1$ is stable, and every image above 1 is unstable.

The third overlap, $m_{bc}$, does not appear. Whether a memory survives depends only on **its own** similarity to the other two.

**Why an unstable image falls exactly into $M$.** All of $\xi^a$'s minority pixels have the same pattern across the three images. So flipping any one of them changes the field at every other pixel by the same amount. As minority pixels flip, each neuron's field therefore changes steadily in one direction, between its value at $\xi^a$ and its value at $M$.

- Minority pixels have the "wrong" sign at both ends, so they all flip.
- All other pixels have the right sign at both ends, so none of them flip.

The dynamics therefore end at exactly $M$ for **any** update order. This proves the 0% result. For corrupted starts it is observed in every trial but not proved.

---

## 3. Part 2: Interpretation

### 3.1 Two memories always work; four or five usually fail

The general worst case for image $a$ is a pixel where *all* other images disagree with it. There, $\xi^a_i h_i = 1 - k/N - \sum_{\mu\neq a} m_{a\mu}$.

| Group size $k$ | Median $\sum_{\mu\ne a} m_{a\mu}$ | Share of images with sum > 1 | Observed clean stability |
|---|---|---|---|
| 2 (pilot) | 0.47 | 0% | 100% |
| 3 (step 5) | about $2\bar m$ | depends on $\bar m$ | 51.5% overall |
| 4 (pilot) | 1.42 | 94% | 6% |
| 5 (step 4) | 1.89 | ~100% | 0.5% |

- **$k=2$:** a stored image is unstable only if $m_{12} > 1 - 2/N \approx 0.998$, i.e. the two images are nearly identical. Distinct digits are therefore always stable, which is why the pilot was at the ceiling.
- **$k=4$ and $k=5$:** a worst-case rule ("such a pixel exists *and* the sum exceeds $1-k/N$") reproduces the measured stability of all 48 pilot $k=4$ images and all 420 step-4 $k=5$ images.

Because typical digits overlap about 0.45 with each other, three or four such overlaps add up to more than the stored image's own contribution of 1.
This is **not** the classical capacity limit: $0.138\,N \approx 141$ uncorrelated patterns. It is a much earlier failure caused by correlation.

### 3.2 Why three memories show a transition

With $k=3$ the stability rule $m_{ab}+m_{ac}<1$ sits exactly in the range the dataset covers. The group mean of $m_{ab}+m_{ac}$ is $2\bar m$, so at $\bar m \approx 0.5$ a typical image is right at the limit.

The transition is gradual rather than a step because images within one group have different pair sums. In the 0.44–0.56 region, individual pair sums range from 0.69 to 1.23. A group with $\bar m = 0.48$ can therefore contain one image above 1 and two below.
The fitted clean-stability midpoint (0.497, 95% CI 0.493–0.502) is close to what the rule implies. That makes it a consequence of the model, not an empirically discovered threshold.

### 3.3 Why corruption moves the 50% point lower

**Observed** (stable images only, from saved trials):

| Margin $1-3/N-(m_{ab}+m_{ac})$ | Images | Accuracy 5% | 10% | 20% |
|---|---|---|---|---|
| 0–0.05 | 60 | 0.74 | 0.59 | 0.43 |
| 0.05–0.10 | 35 | 0.99 | 0.96 | 0.83 |
| 0.10–0.15 | 28 | 1.00 | 1.00 | 0.97 |
| > 0.15 | 155 | 1.00 | 1.00 | ≥0.998 |

**Interpretation (plausible, not derived).**
Corruption shrinks the signal pulling the state back to the target, roughly by a factor of $(1-2c)$ for corruption fraction $c$. It also adds random variation to each neuron's field, because the flipped pixels are scattered randomly.
An image that is only barely stable, with a small margin, can then be pushed over the edge, and it slides to $M$.
Reliable retrieval therefore needs a larger margin as corruption rises: about 0.1 or more at 20%. The images that qualify have lower pair sums, which moves the group-level 50% point down:

| Corruption | 0% | 5% | 10% | 20% |
|---|---|---|---|---|
| 50% point | 0.497 | 0.490 | 0.485 | 0.477 |

### 3.4 Storage instability vs. failure to recover

These are two different failures, and the data separate them cleanly:

- **Storage instability:** the image is not a fixed point, so it cannot be retrieved even from a perfect cue. Unstable images had accuracy 0.00 at every corruption level. They account for 5,240 failures per level, out of 10,800 trials.
- **Recovery failure:** the image is a fixed point, but a corrupted start crosses into the mixture's basin. This adds 311, 512 and 816 extra failures at 5%, 10% and 20%, almost all from images with margin < 0.10.

### 3.5 Why high final overlap is not retrieval

Failed trials end at $M$, whose overlap with the target is between 0.58 and 0.93; 3.9% of failed trials are at or above 0.9.
The group average, $(1+\bar m)/2$, actually **rises** with similarity, while exact retrieval falls to zero.
So a lenient criterion such as "overlap ≥ 0.9" would count some mixtures as successful recall, and the most similar groups would look *best*. The exact criterion avoids this.
Every end state is either exactly the target or exactly $M$, so the exact criterion is not at risk from near misses.

### 3.6 What the shared mixture states say about interference

- Interference never produced **substitution**: no trial ended in another stored image (wrong-memory 0%) or an inverted image (0%). It always produced **merging**: the memories collapse into their consensus.
- Above $\bar m \approx 0.60$, every trial in every group ends in the same $M$ (all-merged fraction 1.0), so the network effectively stores one combined memory. Across the 0.48–0.56 bins, the fraction of image pairs that share an end state at 10% corruption rises from 0.64 to 1.00.
- **Interpretation:** for correlated memories, what Hebbian storage represents most robustly is what the memories have **in common**. The individual details are lost first.

---

## 4. Part 3: Scientific rigour

### 4.1 Observed vs. deduced vs. interpreted

| Type | Statements |
|---|---|
| **Directly observed** (saved data) | All failures end exactly in $M$. $M$ is a fixed point in 180/180 groups. 0 wrong-memory, mirror or non-converged trials. Per-bin stability and accuracy (notebook 06, §6–7). The margin–accuracy table (§3.3). Spearman ρ and logistic fits. Pilot and step-4 stability. |
| **Mathematical deductions** (exact for this model) | The field equation. The $k=3$ stability rule (also confirmed 540/540). The overlap formulas (§2.2). Mixture stability conditions. The exact slide from an unstable image to $M$. $k=2$ always stable for distinct images. The worst-case condition for general $k$. |
| **Plausible interpretations** | The noise explanation for the corruption shift. Hebbian weights "storing the consensus". Shared background and ink regions as the main source of opposition. Any link to biological memory. |

### 4.2 What the Spearman correlations and 50% points show

- **Spearman ρ (−0.93 to −0.96)** shows a strong, consistent **downward association** between group overlap and the outcomes **within this sampled design**.
  Its size depends on the overlap range sampled. Long flat regions at 1.0 and 0.0 inflate ρ, and a different range would give a different value.
  It does not show causation, the shape of the curve, or that the group mean is the right variable. The per-image pair sum is the exact one.
- **The logistic 50% points** summarise where the change happens, assuming a smooth S-shaped curve.
  The confidence intervals (±0.004) treat groups as independent, so they are **too narrow**, since images are reused. They also reflect the stratified design.
  For clean stability, the 50% point follows largely from the rule in §2.4. It is a summary of the model, not evidence of a sharp threshold, and not a parameter of biological memory.

### 4.3 Limitations

- **Typical-image selection:** only the more typical half of each digit (476/946 images). Atypical, thick or blob-like images are excluded, so results apply to this subset.
- **Digit identity:** high-overlap groups contain more 1/3/5/8/9, and low-overlap groups more 4/7/2. Dropping the most unevenly spread digit (4) leaves ρ at −0.91 to −0.93, but digit identity and overlap cannot be fully separated.
- **Image reuse:** 158 images appear in more than one group (89 twice, 69 three times). A subset with no reused images (73 groups) gives ρ −0.92 to −0.95, but every interval is optimistic.
- **Exact pixel scoring:** strict, but harmless here because every end state is exactly target or $M$. It *would* matter for $k=5$, where end states were not characterised.
- **What "similarity" means:** the variable is raw ±1 overlap, which is heavily inflated by shared background. Two random patterns with 30% ink would overlap about 0.15, while different digits average about 0.44. Conclusions are about raw overlap, not perceived shape similarity.
- **Model simplifications:** dense ±1 patterns, symmetric Hebbian weights, no sparse coding or bias correction, and 1,024 binary neurons. The exact thresholds depend on these choices (e.g. the 1 in $m_{ab}+m_{ac}<1$), so they should not be read as biological constants.

### 4.4 Remaining issues that could change the conclusions

1. **Group mean vs. per-image cause.** The analysis is statistically sound at group level, but the variable that actually decides stability is each image's own pair sum. The report should present the group-level findings as a consequence of this per-image rule.
2. **Raw overlap and the background.** If the report says "similar digits interfere", it should make clear that much of the overlap comes from shared blank background, not digit shape.
3. **The step-4 (5-memory) mechanism is unverified.** I claimed the step-4 final-overlap pattern came from the same majority mechanism, but those end states were never saved. With 5 memories there are several possible mixture states (e.g. 3- and 5-image mixtures).
4. **Confidence intervals are optimistic** because of image reuse. This doesn't change the direction or size of the effects, but it matters for any claim about precision.

---

## 5. Figures referenced

- `results/figures/step5_primary_clean_stability.png`: clean stability vs. overlap (primary outcome)
- `results/figures/step5_noisy_retrieval.png`: exact retrieval at 5/10/20%
- `results/figures/step5_logistic_fits.png`: descriptive logistic fits and 50% points
- `results/figures/step5_final_overlap_and_attractors.png`: final overlap (straight line = fully merged groups) and shared-attractor fractions
- `results/figures/step5_outcome_breakdown.png`: outcome categories per bin
- `results/figures/step5_sampling.png`, `step5_digit_composition.png`, `step5_image_reuse.png`: sampling diagnostics

## 6. Proposed targeted validation steps (not run)

| # | Step | Cost | Purpose |
|---|---|---|---|
| A | Add a "mechanism validation" section to notebook 06: rebuild $M$ from saved states, per-image overlap formula, mixture fixed-point check, the $k=3$ stability rule (540/540), and the margin–accuracy table | seconds, saved data only | puts the checks in this document into the reproducible record |
| B | For the 84 step-4 groups, re-run **clean starts only** (420 recalls) with end states saved, and classify them (5-mixture, 3-mixtures, other) | about 10 s | needed before saying anything about how 5-memory networks fail |
| C | Split each pairwise overlap into agreement on shared-background pixels vs. on ink | seconds, analysis only | quantifies how much of "similarity" is background |

The validation script used for this document is in the session scratchpad and can become step A.
