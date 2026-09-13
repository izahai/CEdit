# SPEED: Method and Experiments Notes

**Paper:** *SPEED: Scalable, Precise, and Efficient Concept Erasure for
Diffusion Models*\
**Focus:** Method, implementation, experiments, and ablations.
Mathematical proofs and derivations are intentionally omitted.

## 1. What SPEED Does

SPEED is an editing-based concept-erasure method for text-to-image
diffusion models. Its goal is to remove one or many target concepts
while preserving unrelated concepts and general image-generation
capability.

Instead of fine-tuning the diffusion model, SPEED directly edits model
parameters. The central idea is to restrict parameter changes to a
**null space of retained knowledge**: an editing space intended to
change target concepts without changing representations associated with
non-target concepts.

The main practical problem is that simply putting a very large number of
non-target concepts into the retain set makes accurate null-space
construction difficult. SPEED therefore introduces **Prior Knowledge
Refinement**, consisting of:

1.  **Influence-based Prior Filtering (IPF)**
2.  **Directed Prior Augmentation (DPA)**
3.  **Invariant Equality Constraints (IEC)**

------------------------------------------------------------------------

## 2. Method

### 2.1 Inputs: erasure, anchor, and retain concepts

SPEED works with three groups of concepts:

-   **Erasure set:** concepts to remove.
-   **Anchor set:** replacement/general concepts to which erased
    concepts should be redirected.
-   **Retain set:** non-target concepts whose behavior should remain as
    unchanged as possible.

Examples used in the experiments include:

-   Instance erasure:
    -   `Snoopy -> " "`
    -   `Mickey -> " "`
    -   `SpongeBob -> " "`
-   Artistic-style erasure:
    -   `Van Gogh -> art`
    -   `Picasso -> art`
    -   `Monet -> art`
-   Celebrity erasure:
    -   target celebrity names `-> person`
-   Implicit nudity erasure:
    -   `nudity -> " "`

Here `" "` denotes the blank/null text anchor used by the paper.

### 2.2 Null-space constrained model editing

Conventional editing methods jointly optimize target erasure and
preservation of non-target concepts. The paper argues that preservation
errors can accumulate, especially as more target concepts are erased.

SPEED instead constructs a subspace from retained concepts and restricts
the model update to directions that should not affect those concepts.

Conceptually:

1.  Encode target, anchor, and retain concepts with the text encoder.
2.  Build a representation of retained knowledge.
3.  Estimate a null space associated with that retained knowledge.
4.  Project the erasure update into that null space.
5.  Apply the projected update to the selected model parameters.

This is intended to preserve non-target knowledge while still allowing
target concepts to be erased.

A naive null-space approach has a practical limitation: if the retain
set becomes too large/diverse, its representation approaches high rank,
leaving too little useful null-space freedom. Approximation then causes
semantic degradation. Prior Knowledge Refinement is designed to solve
this.

------------------------------------------------------------------------

## 3. Prior Knowledge Refinement

### 3.1 Influence-based Prior Filtering (IPF)

**Purpose:** Keep the retain set focused on non-target concepts that are
actually at risk of being changed by the erasure.

Not every non-target concept is affected equally by a target-concept
edit. Including weakly affected concepts adds rank to the retain
representation but contributes little useful protection.

SPEED therefore:

1.  Computes an erasure-only model update.
2.  Measures the **prior shift** of each candidate non-target concept
    under that update.
3.  Uses the prior shift as an influence score.
4.  Keeps concepts whose influence is above a threshold.

The default threshold is the **mean prior-shift value** across the
candidate retain set. In the paper's hyperparameter analysis this
corresponds to filtering scale `alpha = 1`.

**Intuition:** protect the non-target concepts most likely to be
damaged, instead of trying to protect every possible concept equally.

### 3.2 Directed Prior Augmentation (DPA)

**Purpose:** Increase retain-set coverage without filling it with
arbitrary or semantically meaningless perturbations.

Simple random perturbations of a concept embedding can produce
embeddings that no longer correspond to coherent versions of the
original concept. DPA instead adds **directed noise** along directions
where the edited model parameters vary least.

Operationally:

1.  Start from each filtered non-target embedding.
2.  Find low-variation directions of the relevant model weight matrix.
3.  Project random noise into those directions.
4.  Add the projected noise to the original embedding.
5.  Filter the augmented concepts again with IPF.
6.  Combine the filtered originals and useful augmented embeddings into
    the refined retain set.

The paper uses:

-   **Augmentation times `N_A = 10`**
-   **Augmentation rank `r = 1`**

The t-SNE visualization in Figure 3 shows the motivation: directed
perturbations can cover a broader embedding neighborhood while remaining
closer to the original semantics after mapping through the model.

### 3.3 Invariant Equality Constraints (IEC)

**Purpose:** Explicitly preserve representations that should stay
constant across prompts.

The paper identifies two important invariants:

-   the CLIP **\[SOT\]** token embedding
-   the **null-text** embedding used for unconditional generation in
    classifier-free guidance

SPEED constrains the edit so that these invariant representations remain
unchanged.

**Intuition:** because these embeddings participate broadly in
generation, preserving them provides an additional safeguard for general
model behavior beyond concept-specific retain examples.

------------------------------------------------------------------------

## 4. Practical SPEED Pipeline

A high-level implementation can be summarized as:

1.  **Choose target concepts.**
2.  **Assign anchors** for those targets.
3.  **Construct an initial retain set** of non-target concepts.
4.  **Encode** target, anchor, and retain concepts.
5.  **Compute erasure influence** on retain concepts.
6.  **Run IPF** to remove weakly affected retain concepts.
7.  **Run DPA** on the filtered retain concepts.
8.  **Run IPF again** on the augmented concepts.
9.  **Combine** the filtered original and augmented concepts into the
    refined retain set.
10. **Construct the null-space projection** from the refined retain set.
11. **Add IEC constraints** for \[SOT\] and null-text embeddings.
12. **Compute the direct parameter edit.**
13. **Apply the edit only to value matrices in cross-attention layers.**
14. **Generate evaluation images** and measure erasure efficacy plus
    prior preservation.

There is no iterative fine-tuning loop in the standard SPEED edit; the
method is based on direct/closed-form parameter editing.

------------------------------------------------------------------------

## 5. Implementation Details

### Base model and generation

Main experiments use **Stable Diffusion v1.4**.

Generated images use:

-   **DPM-Solver**
-   **20 sampling steps**
-   **Classifier-free guidance = 7.5**

### Parameters edited

SPEED edits **cross-attention layers**, but specifically edits only the
**value matrices**.

The paper's motivation is:

-   keys are more associated with layout/compositional structure;
-   values are more associated with content and visual appearance;
-   concept erasure mainly needs to alter semantic/appearance
    information.

The parameter ablation supports this choice:

-   key-only editing does not erase effectively;
-   key + value editing erases but damages prior knowledge more;
-   value-only editing gives the best balance.

### Main hyperparameters

-   IPF filtering scale: **`alpha = 1`**
-   DPA augmentation times: **`N_A = 10`**
-   DPA augmentation rank: **`r = 1`**

### Null-space singular-value thresholds

Because singular values are rarely exactly zero in practice, the
implementation treats sufficiently small singular values as null-space
directions:

-   **few-concept erasure:** `< 1e-1`
-   **implicit concept erasure:** `< 1e-1`
-   **multi-concept erasure:** `< 1e-4`

For implicit concept erasure, the retain set contains only the blank
concept, so the implementation additionally uses an identity
regularization term with **weight `lambda = 0.5`** to ensure the
required matrix is invertible.

------------------------------------------------------------------------

# 6. Experiments

The paper evaluates three main settings:

1.  **Few-concept erasure**
2.  **Multi-concept erasure**
3.  **Implicit concept erasure**

It also tests transfer to other text-to-image model variants and
performs component/hyperparameter ablations.

Main-paper baselines include:

-   ConAbl
-   MACE
-   RECE
-   UCE

Additional appendix comparisons include:

-   ESD
-   RACE
-   Receler
-   SPM
-   SPM without Facilitated Transport

------------------------------------------------------------------------

## 7. Few-Concept Erasure

### 7.1 Tasks

Two types are evaluated:

-   **Instance erasure**
-   **Artistic-style erasure**

### 7.2 Instance setup

Targets include:

-   Snoopy
-   Mickey
-   SpongeBob

Their anchor is blank text.

Non-target concepts highlighted for evaluation include:

-   Pikachu
-   Hello Kitty

The experiment uses **80 instance prompt templates**, including prompts
such as:

-   `a photo of the {Instance}`
-   `a drawing of the {Instance}`
-   `a painting of the {Instance}`

For each concept:

-   80 templates
-   10 generated images per template
-   **800 images per concept**

The retain-set construction crawls Wikipedia fictional-character
category pages and keeps characters with more than **500,000 page
views** over **2020-01-01 to 2023-12-31**, resulting in **1,352
instances** before SPEED's refinement.

### 7.3 Artistic-style setup

Targets include:

-   Van Gogh
-   Picasso
-   Monet

Anchor:

-   `art`

Non-target styles highlighted for evaluation include:

-   Paul Gauguin
-   Caravaggio

The experiment uses **30 artistic-style templates**, with examples such
as:

-   `{Artistic} style painting of the night sky with bold strokes`
-   `{Artistic} style landscape of rolling hills with dramatic brushwork`
-   `Sunrise scene in {Artistic} style, capturing the beauty of dawn`

For each concept:

-   30 templates
-   10 generated images per template
-   **300 images per concept**

The initial retain set contains **1,734 artistic styles** collected from
UCE.

### 7.4 Metrics

The few-concept experiments use:

-   **CLIP Score (CS):** text-image similarity
-   **FID:** distributional difference before vs. after erasure

The main comparison reports:

-   target CS for erasure efficacy;
-   non-target FID for preservation;
-   MS-COCO CS/FID for general prior preservation.

For MS-COCO, the first **1,000 captions** are used to generate **1,000
images**.

### 7.5 Main findings

SPEED generally produces lower FID on non-target concepts, meaning less
change to unrelated concepts.

A key scalability observation is that preservation degrades more slowly
as additional targets are erased. For example:

-   Pikachu FID with SPEED rises from **16.81** for one erased target to
    **21.40** for three targets: `+4.59`.
-   UCE rises from **19.06** to **29.20**: `+10.14`.

For Hello Kitty:

-   SPEED degradation: `+4.48`
-   UCE degradation: `+10.29`

The paper also stresses that the lowest possible target CLIP score is
not necessarily desirable. Very low target CS can correspond to
aggressive over-erasure that damages unrelated knowledge.

------------------------------------------------------------------------

## 8. Multi-Concept Celebrity Erasure

### 8.1 Dataset

The experiment contains **200 celebrities** whose SDv1.4-generated
portraits can be recognized reliably by the GIPHY Celebrity Detector.

They are divided into:

-   up to **100 target celebrities**
-   **100 non-target celebrities**

The experiment erases:

-   10 celebrities
-   50 celebrities
-   100 celebrities

The anchor for celebrity targets is:

-   `person`

The entire 100-celebrity retain set is used for each scale.

### 8.2 Prompt templates

Five templates are used:

-   `a portrait of {Celebrity}`
-   `a sketch of {Celebrity}`
-   `an oil painting of {Celebrity}`
-   `{Celebrity} in an official photo`
-   `an image capturing {Celebrity} at a public event`

### 8.3 Image counts

For non-targets:

-   100 celebrities
-   5 templates
-   1 image/template
-   **500 images total**

For targets, the number of samples per target is adjusted so that the
target side also contains **500 images total**.

### 8.4 Metrics

Using the **GIPHY Celebrity Detector (GCD)**:

-   **Acc_e:** top-1 recognition accuracy for erased targets; lower is
    better.
-   **Acc_r:** top-1 recognition accuracy for retained celebrities;
    higher is better.
-   **H_o:** harmonic-style combined measure balancing erasure and
    retention; higher is better.

The paper also reports:

-   runtime;
-   MS-COCO CS;
-   MS-COCO FID.

### 8.5 Results

For **100-celebrity erasure**, SPEED reports:

-   `Acc_e = 5.87`
-   `Acc_r = 85.54`
-   `H_o = 89.63`
-   runtime = **5.0 s**
-   MS-COCO `CS = 26.22`
-   MS-COCO `FID = 44.97`

For comparison, MACE at 100 celebrities reports:

-   `Acc_e = 4.80`
-   `Acc_r = 80.20`
-   `H_o = 87.06`
-   runtime = **1736 s**
-   MS-COCO `CS = 24.80`
-   MS-COCO `FID = 50.41`

Thus MACE has slightly lower target recognition, but SPEED preserves
non-target celebrities better, obtains a higher overall score, and is
dramatically faster.

The paper summarizes this as roughly a **350x speedup** over MACE for
erasing 100 concepts.

UCE and RECE are faster than training-heavy approaches but their
retained-celebrity accuracy and MS-COCO preservation deteriorate
strongly as the number of erased celebrities increases.

------------------------------------------------------------------------

## 9. Implicit Concept Erasure

### 9.1 Goal

This experiment evaluates concepts that may be present in generated
images even when the target word does not explicitly appear in the
prompt.

The main target is:

-   `nudity -> " "`

### 9.2 Benchmarks

The paper evaluates:

-   **I2P** --- inappropriate prompts involving violence, sexual
    content, and nudity
-   **MMA** --- black-box adversarial benchmark
-   **Ring-A-Bell** --- black-box adversarial benchmark
-   **UnlearnDiff** --- white-box adversarial benchmark

### 9.3 Detection metric

Generated nudity is detected using **NudeNet** with threshold **0.6**.

The reported metric is **Attack Success Rate (ASR)**:

-   lower ASR is better.

The paper also evaluates MS-COCO CS/FID and runtime.

### 9.4 SPEED variants

Two variants are tested:

-   **SPEED without adversarial training/editing (`Ours w/o AT`)**
-   **SPEED with adversarial training/editing (`Ours w/ AT`)**

The adversarially adapted version follows the RECE setting for a fair
comparison.

### 9.5 Results

`Ours w/o AT`:

-   I2P ASR: **0.20**
-   MMA ASR: **0.24**
-   Ring-A-Bell ASR: **0.20**
-   UnlearnDiff ASR: **0.75**
-   runtime: **3.6 s**
-   MS-COCO CS: **26.29**
-   MS-COCO FID: **37.82**
-   supports white-box defense

`Ours w/ AT`:

-   I2P ASR: **0.10**
-   MMA ASR: **0.01**
-   Ring-A-Bell ASR: **0.00**
-   UnlearnDiff ASR: **0.45**
-   runtime: **4.5 s**
-   MS-COCO CS: **26.03**
-   MS-COCO FID: **39.51**
-   supports white-box defense

The adversarial extension substantially improves robustness while
keeping runtime low compared with adversarial-training-heavy baselines.

------------------------------------------------------------------------

## 10. Transfer to Other T2I Models

The paper demonstrates that SPEED is not limited to the main SDv1.4
setup.

### Community Stable Diffusion variants

Tested on:

-   DreamShaper
-   RealisticVision

The experiment performs **composite concept erasure**, including
`Snoopy + Van Gogh`, while attempting to preserve non-target elements in
the same prompt.

### SDXL

SPEED is also used for knowledge editing by selecting arbitrary anchors,
with examples such as:

-   `Wonder Woman -> Woman`
-   `Superman -> Batman`

The qualitative examples indicate that the edited knowledge changes
while the overall image layout and semantics are largely maintained.

### Stable Diffusion 3

The method is adapted to the DiT-based architecture of SDv3. Qualitative
experiments show both target erasure and preservation of non-target
generations.

------------------------------------------------------------------------

# 11. Ablation Studies

## 11.1 Contribution of IEC, IPF, and DPA

The main component ablation erases Van Gogh and evaluates:

-   target CS;
-   average FID over four non-target artistic styles;
-   MS-COCO CS/FID.

Results:

  ---------------------------------------------------------------------------
  Configuration      Target CS ↓ Non-target FID   MS-COCO CS ↑  MS-COCO FID ↓
                                              ↓                
  --------------- -------------- -------------- -------------- --------------
  Base null-space          27.20          50.43          26.42          26.33
  objective                                                    

  \+ IEC                   27.20          48.17          26.44          24.95

  \+ IEC + IPF             26.68          38.02          26.54          20.57

  \+ IEC + IPF +           26.30          32.62          26.52          20.99
  random                                                       
  augmentation                                                 

  **Full SPEED:        **26.29**      **29.35**      **26.55**      **20.36**
  IEC + IPF +                                                  
  DPA**                                                        
  ---------------------------------------------------------------------------

Interpretation:

-   **IEC** gives a modest preservation improvement.
-   **IPF** provides a large reduction in non-target and MS-COCO FID.
-   augmentation further improves coverage.
-   **DPA** outperforms random prior augmentation by keeping augmented
    concepts more semantically meaningful.

## 11.2 Which cross-attention parameters should be edited?

For Van Gogh erasure:

  -------------------------------------------------------------------------------------
  Edited       Target CS ↓ Picasso FID Monet FID ↓        Paul   Caravaggio MS-COCO FID
  parameters                         ↓             Gauguin FID        FID ↓           ↓
                                                             ↓              
  ------------ ----------- ----------- ----------- ----------- ------------ -----------
  Key only           27.67       42.11       26.09       28.08        52.44       18.72

  Key + Value        26.24       48.41       28.65       33.79        57.23       23.20

  **Value        **26.29**   **35.86**   **16.85**   **24.94**    **39.75**   **20.36**
  only**                                                                    
  -------------------------------------------------------------------------------------

Key-only editing barely changes the target compared with the original
Van Gogh CS of 28.75. Editing both keys and values increases erasure but
damages non-target knowledge more. Value-only editing provides the
paper's preferred trade-off.

## 11.3 IPF filtering scale

The IPF threshold is scaled by `alpha`.

Observed behavior:

-   Very small `alpha` retains too many weakly affected concepts.
-   This increases rank, shrinks the useful null space, and hurts both
    erasure and preservation.
-   Very large `alpha` keeps too few concepts, improving editing freedom
    but reducing prior coverage.
-   **`alpha = 1`** gives the best reported balance.

## 11.4 DPA augmentation count

The paper varies `N_A`.

Observed behavior:

-   Increasing augmentation from roughly **1 to 10** improves non-target
    FID because retain coverage becomes more comprehensive.
-   Increasing beyond about **10 toward 20** starts to worsen
    preservation because the enlarged retain representation again
    narrows the useful null space.
-   The chosen value is **`N_A = 10`**.

## 11.5 DPA augmentation rank

Increasing rank `r` gives DPA more perturbation directions, but the
paper observes that non-target FID generally increases with larger `r`.

The selected setting is:

-   **`r = 1`**

This keeps augmentation low-rank while still improving semantic
coverage.

------------------------------------------------------------------------

# 12. Experimental Takeaways

The experiments support four main practical claims.

### 1. Preservation is the main strength

SPEED is not designed simply to minimize similarity to the erased
target. It aims to erase enough of the target while keeping non-target
outputs close to the original model.

### 2. The advantage becomes clearer at scale

Methods that work reasonably for one target can cause severe prior
damage when tens or hundreds of concepts are edited together. SPEED's
null-space approach plus retain-set refinement is specifically designed
for this multi-concept regime.

### 3. Direct editing is extremely fast

The method avoids a conventional fine-tuning loop. The headline
experiment erases **100 celebrity concepts in about 5 seconds on one
A100 GPU**.

### 4. Prior Knowledge Refinement matters

The ablations show that a naive null-space constraint is not sufficient.
The largest gains come from carefully deciding:

-   which priors need protection (IPF);
-   how to expand them without meaningless noise (DPA);
-   which globally important representations should be held invariant
    (IEC).

------------------------------------------------------------------------

## 13. Reproduction-Oriented Checklist

To reproduce the main SPEED setup from the paper:

-   Use **Stable Diffusion v1.4**.
-   Edit **cross-attention value matrices only**.
-   Define target concepts and their anchors.
-   Build the appropriate initial retain set.
-   Compute prior-shift influence and apply **IPF**.
-   Use **`alpha = 1`** for the default IPF threshold scale.
-   Apply **DPA** with:
    -   `N_A = 10`
    -   `r = 1`
-   Re-filter DPA augmentations with IPF.
-   Include **\[SOT\]** and **null-text** as IEC invariants.
-   Construct the null-space projection using singular-value cutoffs:
    -   `1e-1` for few-concept and implicit erasure
    -   `1e-4` for multi-concept erasure
-   For implicit erasure, use identity regularization with
    `lambda = 0.5` in the specified inverse term.
-   Generate with:
    -   DPM-Solver
    -   20 steps
    -   CFG = 7.5
-   Evaluate both:
    -   **target erasure**
    -   **non-target/general prior preservation**
-   For few-concept tasks, use CS/FID and MS-COCO.
-   For celebrity erasure, use GCD `Acc_e`, `Acc_r`, `H_o`, runtime, and
    MS-COCO.
-   For implicit nudity erasure, use NudeNet ASR at threshold 0.6 on
    I2P/adversarial benchmarks.

------------------------------------------------------------------------

## 14. Scope Note

This document intentionally leaves out the paper's mathematical proofs
and detailed closed-form derivations. Equations are replaced with
operational descriptions of what each component computes and why it is
used.
