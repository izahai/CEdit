# Quantitative Comparison of Multi-Concept Erasure

I compared the evaluation results in `eval_multi_1_summary.csv` against **Table 2 of the SPEED paper**. The paper evaluates erasure of 10, 50, and 100 celebrities using GIPHY Celebrity Detector top-1 accuracy. It defines **Accₑ ↓** as accuracy on erased target celebrities, **Accᵣ ↑** as accuracy on retained non-target celebrities, and **Hₒ ↑** as the harmonic mean between erasure success, (1-\mathrm{Acc}_e), and retention accuracy.

The paper uses 500 target and 500 non-target evaluation images for each setting, which matches the 500-image erase and retain splits in my evaluation.

## Main comparison with SPEED

I use `conditional_accuracy_CE_Eval` from my CSV as the closest direct equivalent to the paper's GCD top-1 accuracy. I recompute:

[
H_o=\frac{2}{(1-\mathrm{Acc}_e)^{-1}+(\mathrm{Acc}_r)^{-1}}
]

| Erased celebrities |      Method |    Accₑ ↓ |     Accᵣ ↑ |       Hₒ ↑ |
| ------------------ | ----------: | --------: | ---------: | ---------: |
| **10**             | SPEED paper |     1.81% | **89.09%** | **93.42%** |
|                    | Ours | **1.01%** |     87.12% |     92.68% |
|                    | Ours+SPEED | **0.20%** |     87.30% |     93.13% |
| **50**             | SPEED paper |     3.46% |     88.48% |     92.34% |
|                    | Ours | **1.23%** |     88.28% | **93.23%** |
|                    | Ours+SPEED |     1.65% | **88.87%** | **93.37%** |
| **100**            | SPEED paper |     5.87% | **85.54%** |     89.63% |
|                    | Ours | **0.60%** |     85.40% | **91.87%** |
|                    | Ours+SPEED |     0.61% |     82.76% |     90.32% |

The corresponding SPEED values reported in Table 2 are 1.81/89.09/93.42 for 10 celebrities, 3.46/88.48/92.34 for 50, and 5.87/85.54/89.63 for 100 celebrities.

## Difference from the paper

For **Config 1**:

| Erased |       Δ Accₑ |   Δ Accᵣ |         Δ Hₒ | Interpretation                      |
| ------ | -----------: | -------: | -----------: | ----------------------------------- |
| 10     | **−0.80 pp** | −1.97 pp |     −0.74 pp | Better erasure, weaker preservation |
| 50     | **−2.23 pp** | −0.20 pp | **+0.89 pp** | Better overall                      |
| 100    | **−5.27 pp** | −0.14 pp | **+2.24 pp** | Clearly better overall              |

For **Config 2**:

| Erased |       Δ Accₑ |       Δ Accᵣ |         Δ Hₒ | Interpretation                                     |
| ------ | -----------: | -----------: | -----------: | -------------------------------------------------- |
| 10     | **−1.61 pp** |     −1.79 pp |     −0.29 pp | Much stronger erasure, slightly lower preservation |
| 50     | **−1.81 pp** | **+0.39 pp** | **+1.03 pp** | Better on all three main metrics                   |
| 100    | **−5.26 pp** |     −2.78 pp | **+0.69 pp** | Much stronger erasure but weaker preservation      |

A negative difference in Accₑ is favorable because lower is better; a positive difference in Accᵣ or Hₒ is favorable.

## Analysis

The strongest result is the **50-celebrity setting with Config 2**. My model obtains **Accₑ = 1.65%, Accᵣ = 88.87%, and Hₒ = 93.37%**, compared with SPEED's **3.46%, 88.48%, and 92.34%**. Thus, erasure accuracy improves by **1.81 percentage points**, retained-celebrity accuracy improves by **0.39 points**, and overall Hₒ improves by **1.03 points**. This is the cleanest result because it improves both erasure efficacy and prior preservation simultaneously.

At **100 celebrities**, Config 1 gives the strongest overall result. My model reaches **Accₑ = 0.60%**, compared with SPEED's **5.87%**, while maintaining almost the same retained-celebrity accuracy: **85.40% vs. 85.54%**. Consequently, Hₒ increases from **89.63% to 91.87%**, an improvement of **2.24 percentage points**. This is particularly significant because the paper emphasizes scalability to 100 simultaneously erased concepts.

For **10 celebrities**, my configurations erase the target identities more strongly than SPEED, especially Config 2 with only **0.20% Accₑ**, but retention is approximately 1.8–2.0 points lower. As a result, Hₒ remains slightly below the paper's 93.42%. Therefore, the 10-celebrity result represents a trade-off rather than a strict improvement.

## Comparison with the other methods in the paper

The paper's Table 2 reports the following competitors: ConAbl, UCE, RECE, MACE, and SPEED.

For 50 celebrities, my best Hₒ of **93.37%** exceeds:

* SPEED: 92.34%
* MACE: 90.03%
* UCE: 48.41%
* ConAbl: 48.74%
* RECE: 32.95%

For 100 celebrities, my Config 1 Hₒ of **91.87%** exceeds:

* SPEED: 89.63%
* MACE: 87.06%
* ConAbl: 57.97%
* RECE: 38.16%
* UCE: 34.60%

Therefore, **based only on the celebrity-erasure metrics available in my CSV, Config 1 would achieve the highest reported Hₒ at 50 and 100 erased celebrities relative to the methods shown in Table 2**.

## Important evaluation caveat

My CSV also contains `identity_hit_rate`, which treats an image without a correctly detected identity as incorrect over all 500 generated images. Using this stricter denominator gives:

| Erased | Config   |    Accₑ ↓ |     Accᵣ ↑ |       Hₒ ↑ |
| ------ | -------- | --------: | ---------: | ---------: |
| 10     | Config 1 |     1.00% |     86.60% |     92.39% |
| 10     | Config 2 | **0.20%** |     86.60% |     92.73% |
| 50     | Config 1 | **1.20%** |     87.40% |     92.75% |
| 50     | Config 2 |     1.60% | **87.80%** | **92.80%** |
| 100    | Config 1 | **0.60%** | **84.20%** | **91.17%** |
| 100    | Config 2 | **0.60%** |     81.60% |     89.62% |

The paper states that it measures GCD top-1 accuracy but does not specify in the cited evaluation description whether failed face detections are included in the denominator. Therefore, `conditional_accuracy_CE_Eval` is the most natural direct comparison, but this denominator difference should be verified before claiming an exact reproduction or SOTA improvement.

## Conclusion

The results are promising relative to SPEED. My method shows **substantially stronger target erasure**, particularly as the number of concepts increases. The clearest gains are:

**50 celebrities — Config 2:**
Accₑ **1.65% vs. 3.46%**, Accᵣ **88.87% vs. 88.48%**, Hₒ **93.37% vs. 92.34%**.

**100 celebrities — Config 1:**
Accₑ **0.60% vs. 5.87%**, Accᵣ **85.40% vs. 85.54%**, Hₒ **91.87% vs. 89.63%**.

Thus, on the available celebrity metrics, **Config 2 is the strongest setting at 50 concepts, while Config 1 is the strongest and most balanced setting at 100 concepts**. The 100-celebrity result is especially notable: it reduces residual target-recognition accuracy by roughly **90% relative to SPEED** while losing only **0.14 percentage points of retained-celebrity accuracy**, producing a **+2.24-point improvement in Hₒ**.

However, Table 2 also evaluates **runtime, MS-COCO CLIP Score, and MS-COCO FID**. The current CSV does not contain those measurements, so I cannot yet claim that my method beats SPEED across the complete Table 2 evaluation. SPEED reports 5.0 s, CS 26.22, and FID 44.97 for its 100-celebrity experiment.
