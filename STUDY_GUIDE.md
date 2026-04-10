## 1. Data Input

Start with data.py:

- **`AEFDataset`** — Synthetic dataset, simplest entry point
- **`AEFNPZDataset`** — Real data from `.npz` chips
- Each sample produces:
  ```python
  {
      "source_data": {"sentinel2": (T, H, W, C)},  # e.g. (16, 128, 128, 5)
      "timestamps":  {"sentinel2": (T,)},           # ms epoch
      "valid_period": (start_ms, end_ms)
  }
  ```
- Collation pads variable-length sequences via `collate_fn` → final batch shape: `(B, T, H, W, C)`

---

## 2. Model Entry Point

Then read aef_module.py — specifically `AlphaEarthFoundations`:

### Forward pass in order:

| Step                       | Code                             | What happens                                                                                                                    |
| -------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Per-source encoding**    | `_stack_inputs()`                | Each source projected to `per_source_latent=32` dims via `IndividualSourceEncoder`, then concatenated → `(B, T, H, W, C_total)` |
| **Teacher encode**         | `self.encoder(x, ts)`            | `STPEncoder` with 3 pathways (Space/Time/Precision) → `(B, T, H/2, W/2, d_p)`                                                   |
| **Student perturbation**   | `_perturb_inputs()`              | Random source drops, frame drops, or half-year drops                                                                            |
| **Student encode**         | same encoder                     | Perturbed inputs → student features                                                                                             |
| **Temporal summarization** | `self.summarizer(feats, ts, vp)` | `TemporalSummarizer` pools time → **64D unit vectors** on $S^{63}$                                                              |
| **Decoding**               | `self.decoder(...)`              | `VonMisesFisherDecoder` samples vMF distribution → reconstructed source patches                                                 |

---

## 3. Core Architecture Components

Read in this order:

```
encoder_utils.py   →  IndividualSourceEncoder, SinusoidalTimeEncoding, SummaryPeriodEncoder
stp_operators.py   →  SpaceOperator, TimeOperator, PrecisionOperator
STPBlock.py        →  One STP block (space+time+precision + exchanges)
encoder.py         →  STPEncoder (stacks 6 or 15 STPBlocks)
decoder.py         →  VonMisesFisherDecoder
aef_module.py      →  AlphaEarthFoundations (ties everything together)
```

---

## 4. Loss Function

Read src/alphaearth/loss_function.py — implements 4 components from the paper:

$$l = a\sum_i f_i(y_i, y'_i)w_i + b|u_i \cdot u'_i| + c \cdot L_{consistency} + d \cdot L_{text}$$

| Weight    | Loss                   | Purpose                        |
| --------- | ---------------------- | ------------------------------ |
| `a`       | Reconstruction (L1/CE) | Decode back to source pixels   |
| `b=0.05`  | Batch uniformity       | Embeddings uniform on $S^{63}$ |
| `c=0.02`  | Consistency            | Teacher ≈ Student embeddings   |
| `d=0.001` | Text CLIP              | Align text ↔ image embeddings  |

---

## 5. Training Loop

Finally read [src/alphaearth/training.py](src/alphaearth/training.py) → [`Trainer`](src/alphaearth/training.py), then the entry point [src/alphaearth/run_train.py](src/alphaearth/run_train.py).

---

## Quickest Sanity Check

Run the test suite to see everything working end-to-end:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python test_aef.py
```

Key tests to read: [`test_model_forward`](src/alphaearth/test_aef.py), [`test_gradient_flow`](src/alphaearth/test_aef.py), [`test_training_step`](src/alphaearth/test_aef.py).Key tests to read: [`test_model_forward`](src/alphaearth/test_aef.py), [`test_gradient_flow`](src/alphaearth/test_aef.py), [`test_training_step`](src/alphaearth/test_aef.py).
