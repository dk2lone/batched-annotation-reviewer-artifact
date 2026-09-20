# Batched LLM annotation dependence: reviewer artifact

This artifact reproduces the paper's reported analyses of excess within-call error dependence
in batched LLM annotation. It compares joint annotation calls with target-only calls that show
the same documents in the same order, estimates how dependence changes with batch size, and
recomputes repeated-sampling confidence-interval coverage. The included evidence is the final
analysis-ready representation of the model outputs; no model inference is required.

## Repository structure

- `src/batched_annotation/`: prompt construction, strict parsing, statistical analyses, dataset
  verification, and figure generation.
- `evidence/covariance_grid/`: call-group error arrays for the 4-model by 4-task covariance grid,
  its randomization null, and recovered-output sensitivity.
- `evidence/batch_size/`: error arrays and an execution summary for batch sizes 2, 4, 8, 16, and
  32 in four cells.
- `evidence/repeated_sampling/`: original and recovered error arrays for four repeated-sampling
  cells, including repair masks and fixed bootstrap seeds.
- `evidence/dataset_indices/`: source indices, document hashes, reference labels, lengths, and
  strata. The Civil Comments index also includes its public toxicity score for reference-label
  sensitivity analysis. Dataset text is not redistributed.
- `protocol.json`: exact prompts, model and dataset revisions, decoding settings, random seed,
  conditions, and recovery constraints.
- `tests/`: numerical regression, prompt, parser, and protocol checks.

Arrays encode `0` for agreement with the dataset reference, `1` for disagreement, and `NaN` for
an unresolved output. Dataset labels are reference labels, not independently adjudicated truth.

## Install

Python 3.11--3.13 and [uv](https://docs.astral.sh/uv/) are required. From the repository root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev
```

`uv.lock` pins every analysis and test dependency. Installation uses about 300 MB of disk and
normally takes 1--3 minutes on a current laptop with a warm package cache.

## Reproduce the reported results

Run the complete CPU analysis and figure pipeline:

```bash
uv run reproduce all
```

This writes the following ignored files under `outputs/`:

- `covariance_grid.json`: equal-cell raw covariance contrast `0.00961043`, 95% randomization
  interval `[0.00935964, 0.00985711]`, and one-sided `p = 0.0002`. All 16 cells are
  Holm-significant against zero, while 13 point estimates meet the separate prespecified
  practical floor of `0.002`. It also reports the four malformed-output treatments,
  recovered-output sensitivity, assignment checks, and fake-batch controls.
- `batch_size.json`: descriptive equal-cell slope `-0.00909521` per doubling of batch size. The
  three complete-assignment slopes are `-0.00960244`, `-0.00879470`, and `-0.00888848`. The
  prespecified block-bootstrap interval is retained as a labeled sensitivity, with an explicit
  warning that it does not preserve recurring item identities across assignments. The pooled
  standardized contrasts for sizes
  2/4/8/16/32 are `0.079479/0.080658/0.071110/0.059559/0.044553`; the corresponding excess
  design effects rise from `1.07948` to `2.38113`.
- `repeated_sampling.json`: IID Wald / call-aware t / whole-call bootstrap coverage of
  `0.7775/0.9300/0.9175` (Granite/IMDb), `0.9675/0.9650/0.9400`
  (Llama/Civil Comments), `0.9075/0.9525/0.9475` (Mistral/MultiNLI), and
  `0.8800/0.9600/0.9525` (Qwen/Civil Comments). Exact binomial intervals and repair-free
  subsets are included in the same file. Frozen reference-precision and reference-uncertainty
  checks support the item-level undercoverage claim for Granite, Mistral, and Qwen, but not Llama.
- `followups.json`: design-effect and effective-sample-size translations plus the descriptive
  four-cell cross-experiment comparison (`Pearson r = 0.986747`, `Spearman rho = 1.0`).
- `reference_sensitivity.json`: post-hoc Civil Comments results at toxicity thresholds `0.4`,
  `0.5`, and `0.6`, plus exclusion of scores strictly between `0.4` and `0.6`. It recomputes the
  four-cell covariance result, both Civil Comments batch-size trajectories, and both Civil
  Comments repeated-sampling comparisons from stored schedules and outputs.
- `figures/*.pdf` and `figures/source_data.csv`: the covariance-grid, batch-size, and coverage
  figures and their plotted values.

Each result can also be reproduced separately:

```bash
uv run reproduce covariance-grid
uv run reproduce batch-size
uv run reproduce repeated-sampling
uv run reproduce reference-sensitivity
uv run reproduce followups
uv run reproduce figures
```

`figures` expects the four JSON outputs, so run it last. On a modern laptop, the full CPU command
takes about 1 minute and uses less than 1 GB of memory. Numerical output is deterministic because
the schedules, randomization arrays, and bootstrap seeds are included.

The repeated-sampling result is a recovered post-execution sensitivity analysis. It uses 1,095
recovered calls for which the input prompt, model revision, decoding seed, precision, runtime, and
decoding settings were held fixed; only enforced JSON output and the maximum response allowance
changed. The output also reports the repair-free subset instead of treating recovery as invisible.

This repository reproduces the reported statistical analyses. Because raw model responses and
cloud execution records are not included, it does not independently validate model execution or
the transformation from raw responses to the analysis-ready error arrays.

## Tests and GPU protocol check

Run the full test suite and lint check:

```bash
uv run pytest
uv run ruff check .
```

All reported analyses above are CPU-only. No reported result requires a GPU because final model
outputs are included as error arrays. To check the pinned model/runtime configuration and all
evidence paths without importing vLLM, downloading weights, or starting a GPU, run:

```bash
uv run reproduce check-gpu-protocol
```

The expected status is `syntax_and_evidence_check_passed` with `gpu_started: false`. Regenerating
the archived model outputs from text would require the four pinned 7B/8B models in `protocol.json`,
vLLM 0.28.0 on Linux, and approximately one 80 GB GPU; it is not needed for artifact reproduction
and no inference launcher or cloud configuration is included.

## Datasets

Civil Comments, AG News, MultiNLI, and IMDb are downloaded from their public providers rather than
redistributed. The CPU analyses do not download them. To independently download and verify that
the indexed rows still match the included document hashes, normalized references, and lengths:

```bash
uv sync --extra data
uv run python -m batched_annotation.datasets civil_comments
uv run python -m batched_annotation.datasets ag_news
uv run python -m batched_annotation.datasets multi_nli
uv run python -m batched_annotation.datasets imdb
```

These optional checks use the exact dataset revisions and splits in `protocol.json`, write only to
the normal dataset cache, and do not modify the repository. They need network access, no GPU, and
roughly 20--90 minutes in total depending on cache and bandwidth. Model weights are not included.

The code is released under the MIT License. Dataset and model use remains subject to the licenses
of their public providers.
