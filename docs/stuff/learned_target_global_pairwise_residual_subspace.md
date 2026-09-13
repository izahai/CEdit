# Learned Target-Global Pairwise Residual Subspace

## Goal

Extend `target_global_pairwise_residual_subspace` so that the final residual for
each target is learned for each edited cross-attention value-projection weight.
The learned residual should rotate the target's output direction by a specified
amount while using the smallest possible residual.

The existing target-global pairwise residual subspace remains the set of
allowed directions. The new method learns the direction and magnitude within
that subspace instead of deriving a fixed direction and copying the legacy
anchor-residual norm.

## Formulation

Let:

- \(t_i\) be the embedding of target \(i\);
- \(B\) be the retained target-global pairwise residual basis;
- \(W_\ell\) be the original frozen `attn2.to_v` weight for layer \(\ell\);
- \(c_{i,\ell}\) be learnable coefficients for target \(i\) and layer \(\ell\).

Constrain the learned residual to the global residual subspace:

\[
r_{i,\ell}=B^\top c_{i,\ell}.
\]

The original and shifted outputs are:

\[
y_{i,\ell}=W_\ell t_i,
\qquad
y'_{i,\ell}=W_\ell(t_i+r_{i,\ell}).
\]

Optimize the coefficients with an angular-margin loss and a residual-norm
penalty:

\[
\mathcal L_{i,\ell}=
\lambda_{\mathrm{angle}}
\operatorname{ReLU}
\left(
\cos(y'_{i,\ell},y_{i,\ell})-\tau
\right)^2
+
\lambda_{\mathrm{norm}}\lVert r_{i,\ell}\rVert_2^2.
\]

Here, \(\tau=\cos(\theta_{\min})\) defines the minimum required output-angle
change. An explicit margin is necessary because merely requiring two
directions to be different admits the limiting solution \(r_{i,\ell}\to0\).

An optional output-magnitude preservation term can prevent the optimizer from
meeting the angular objective primarily by shrinking the shifted output:

\[
\mathcal L_{\mathrm{magnitude}}=
\lambda_{\mathrm{magnitude}}
\left(
\lVert y'_{i,\ell}\rVert_2-
\lVert y_{i,\ell}\rVert_2
\right)^2.
\]

The first prototype should minimize the embedding-space norm
\(\lVert r_{i,\ell}\rVert_2\). A later variant can instead minimize the
output-space change \(\lVert W_\ell r_{i,\ell}\rVert_2\).

## Implementation plan

1. Add a new anchor mode named
   `learned_target_global_pairwise_residual_subspace`. Keep the existing TGPRS
   behavior unchanged for reproducibility.

2. Reuse the existing target-global pairwise residual-matrix construction and
   normalized SVD basis. Do not duplicate basis-building logic.

3. Add a deep module in `src/` whose interface hides coefficient
   initialization, optimization, numerical safeguards, and diagnostics:

   ```python
   residuals, diagnostics = learn_layer_residuals(
       target_embeddings,
       basis,
       layer_weight,
       config,
   )
   ```

4. For each `attn2.to_v` layer, freeze the original \(W_\ell\) and optimize one
   coefficient vector \(c_{i,\ell}\) per target. Gradients must update only the
   coefficients, not the diffusion model or text encoder.

5. Construct a layer-specific SPEED interaction statistic from the optimized
   residuals:

   \[
   D_\ell=\frac{1}{N}\sum_{i=1}^{N}r_{i,\ell}^{\top}t_i.
   \]

6. Pass \(D_\ell\) into the existing SPEED closed-form update. Because the
   residuals depend on \(W_\ell\), the interaction statistic can no longer be
   calculated once and shared by every layer.

7. Add CLI and configuration options for:

   - minimum output angle or target cosine;
   - angle-loss weight;
   - residual-norm weight;
   - optimizer learning rate;
   - optimization step count;
   - optional output-magnitude preservation and its weight.

8. Record per-layer diagnostics, including initial and final cosine, achieved
   angle, residual norm, output residual norm, loss, step count, convergence
   status, and subspace projection error.

## Numerical safeguards

- Clamp cosine denominators with an epsilon.
- Detect targets whose original output norm is near zero and report or skip
  them with deterministic behavior.
- Keep the residual parameterized by basis coefficients so that subspace
  membership holds by construction.
- Optimize against the original frozen layer weight to avoid a circular
  dependency between residual learning and the subsequent closed-form edit.
- Fail clearly on non-finite losses, coefficients, residuals, or outputs.

## Tests

Add focused CPU `unittest` coverage verifying that:

- every learned residual remains in the span of \(B\);
- the requested cosine or angular margin is achieved when feasible;
- residual norm decreases while the angular constraint remains satisfied;
- different layer weights can produce different residuals for the same target;
- gradients do not modify \(W_\ell\);
- the result has the expected target and embedding shapes;
- fixed seeds give deterministic results;
- zero-norm and infeasible cases remain numerically stable and are diagnosed;
- the layer-specific interaction statistic matches its direct definition;
- existing TGPRS tests and behavior remain unchanged.

Run the smallest relevant tests first, followed by:

```bash
python -m unittest discover -s tests
```

## First experiment

Use the 10-celebrity benchmark to compare:

1. legacy SPEED;
2. `target_global_pairwise_residual_subspace`;
3. `learned_target_global_pairwise_residual_subspace`.

Start with:

- a fixed minimum output-angle change of \(10^\circ\);
- embedding-space residual-norm minimization;
- frozen original layer weights;
- the existing TGPRS residual rank;
- output-magnitude preservation disabled, then enabled as an ablation.

Report erase accuracy, retain accuracy, mean achieved angle, mean and maximum
residual norms, convergence rate, and optimization cost. If the angular margin
cannot be reached for many targets, inspect the reachable output subspace
\(W_\ell\operatorname{span}(B)\) before increasing optimizer steps or loss
weights.

## Success criteria

The prototype is successful if it:

- satisfies the requested output-angle margin for most target-layer pairs;
- finds smaller residuals than a fixed-norm construction under the same angle
  requirement;
- preserves or improves erasure without materially reducing retention;
- keeps all optimization isolated behind the layer-residual module's small
  interface.
