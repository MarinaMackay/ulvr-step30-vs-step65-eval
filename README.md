# ULVR step_30 vs step_65 evaluation

**Author:** Yitong Ma, UC San Diego · October 2026

## Results

Qwen3.5-4B trained with [Vision-OPD](https://github.com/VisionOPD/Vision-OPD) on 6,240 ULVR `bbox_crop` samples (seed 42, 1 epoch / 65 updates). Two checkpoints, four benchmarks, 2,636 questions each (5,272 graded predictions).

| benchmark | n | step_30 | step_65 | Δ (pp) |
| --- | ---: | ---: | ---: | ---: |
| V* Bench | 191 | 76.44% | 73.30% | −3.14 |
| ZoomBench | 845 | 46.15% | 43.43% | −2.72 |
| HR-Bench 4K | 800 | 79.38% | 76.00% | −3.38 |
| HR-Bench 8K | 800 | 74.75% | 70.75% | −4.00 |
| **macro (equal weight)** | — | **69.18%** | **65.87%** | **−3.31** |
| micro (question-weighted) | 2,636 | 67.11% | 63.77% | −3.34 |

Question-level: 275 regressions, 187 improvements from step_30 to step_65.

## Caveats — read before citing

- **MME-RealWorld and MME-RealWorld-CN were not evaluated.** The four-benchmark average is not comparable to six-benchmark averages.
- **Non-standard judge.** Grading used deterministic rules (retained upstream rules, explicit-option matching, exact-integer matching) plus assistant semantic review for 74 of 5,272 answers. This differs from the Qwen2.5-72B-Instruct judge used by Vision-OPD, so numbers are **not strictly comparable** to officially reported scores.
- **No base-model baseline.** This compares two training steps only; it does not show whether training helps relative to the base model.
- One ambiguous item (HR-Bench 4K, step_30) was graded No; grading it Yes changes the macro by +0.03 pp.

## Contents

| path | |
| --- | --- |
| `results/summary.json` | accuracy by benchmark and category |
| `results/comparison.md` | result tables and conclusions (Chinese) |
| `results/grading-sources.json` | which rule graded each answer |
| `results/validation-and-protocol.json` | completeness checks and grading policy |
| `results/cases_redacted.jsonl` | 61 stratified example cases — **IDs and outcomes only**; benchmark questions, gold answers and model outputs are removed to avoid redistributing benchmark content |
| `scripts/`, `eval-kit/`, `config/` | data preparation, training, evaluation and grading code |

Model weights, training data and benchmark data are not included. Vision-OPD itself is not vendored here; use the upstream repository (Apache-2.0).

## Acknowledgments

Thanks to William for suggesting this experimental direction.

## License

- **Code** (`scripts/`, `eval-kit/`, `config/`, `Makefile`): [PolyForm Noncommercial 1.0.0](LICENSE)
- **Results** (`results/`): [CC BY-NC 4.0](results/LICENSE)

Both allow research and other noncommercial use with attribution and forbid commercial use. Use of these results or code to train, fine-tune or evaluate AI models requires the author's written permission.

## Citation

```bibtex
@misc{ma2026ulvrstepeval,
  author = {Ma, Yitong},
  title  = {ULVR step\_30 vs step\_65 evaluation},
  year   = {2026},
  institution = {UC San Diego},
  howpublished = {\url{https://github.com/MarinaMackay/ulvr-step30-vs-step65-eval}}
}
```
