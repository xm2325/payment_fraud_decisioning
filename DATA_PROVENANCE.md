# Data provenance and external validation

## Local pipeline-validation stream

The default `scripts/run_all.py` experiment uses a transparent synthetic payment stream generated inside this repository. It is used to test temporal leakage controls, policy evaluation, emerging-attack detection, monitoring, label delay and Fraud Ops queue design. It is not Moniepoint data and it is not an estimate of real payment-fraud prevalence.

## PaySim external benchmark

The repository now includes `scripts/run_paysim.py`, which accepts a local PaySim CSV and runs a separate temporal benchmark. PaySim is a synthetic mobile-money dataset generated from aggregate patterns rather than a public dump of customer transactions.

Verified source facts used by this project:

- Dataset: **PaySim: Synthetic Financial Datasets For Fraud Detection**.
- Original reference: Edgar Alonso Lopez-Rojas, Ahmad Elmir and Stefan Axelsson, *PaySim: A Financial Mobile Money Simulator for Fraud Detection*, EMSS 2016.
- Full standard dataset: **6,362,620 transactions**, **8,213 fraud transactions**, 744 hourly steps covering about 30 days.
- Standard columns: `step`, `type`, `amount`, origin/destination account identifiers and balances, `isFraud`, `isFlaggedFraud`.
- License on the current Kaggle/Hugging Face distribution: **CC BY-SA 4.0**.
- Hugging Face mirror: `kohdified/synthetic-financial-data`, currently listed as a 494 MB CSV with 6.36M rows.

A RelationalAI repository also publishes a small PaySim-derived subset under the same license. Its sampling script states that it keeps all fraud rows and samples an equal number of non-fraud rows, producing an artificial ~1:1 class balance. This repo therefore does **not** use that mini subset to claim alert rates, customer-friction rates or fraud prevalence.

## Run the external benchmark

Place the standard PaySim CSV locally and run:

```bash
python scripts/run_paysim.py --csv data/raw/paysim.csv --full
```

For a time-prefix smoke test:

```bash
python scripts/run_paysim.py --csv data/raw/paysim.csv --max-step 240
```

The `--max-step` option keeps an early contiguous time prefix; the loader reads CSV chunks and stops after the requested time boundary rather than materialising all 494 MB first. It does not randomly sample individual transactions, so rolling history remains interpretable. `--full` is explicit and verifies the standard 6,362,620-row / 8,213-fraud / step-1-to-743 counts before modelling.

## External-validation status — verified on GitHub Actions

The full public PaySim Parquet representation was materialised on a GitHub-hosted runner on 2026-08-21. The canonical audit passed with **6,362,620 rows, 8,213 fraud cases and step range 1--743**. The completed relational four-model ablation took about **216 seconds** after download.

The temporal split has strong prevalence drift: train 0.0833% fraud, validation 0.6804%, test 1.3384%. The verified relational run gives balance-free PR-AUC **0.3403** for transaction-only features, **0.3408** with basic history and **0.3530** with relational/pair/counterparty history. The relational model improves precision and reduces future legitimate flags but slightly reduces recall/value recall at its validation-derived operating point. Adding old/new-balance-derived fields lifts PR-AUC to **0.9950**, which remains simulator-mechanics sensitivity rather than a production-like result.

Raw shards and materialised feature files were not committed. Only aggregate audit, split, ablation and operating-point outputs are stored in `results/paysim_full/`.

## v1.3 GitHub Actions full-data path

v1.3 runs full external validation to `.github/workflows/paysim-full.yml`, because the interactive build container cannot resolve `huggingface.co` while GitHub-hosted runners can access the public dataset. The workflow downloads both public parquet shards from `kohdified/synthetic-financial-data`, checks the canonical PaySim counts (**6,362,620 rows, 8,213 fraud, step 1--743**), computes strict prior-step rolling features with DuckDB, trains and calibrates a LightGBM model on a 60/20/20 temporal step split, compares it with the source `isFlaggedFraud` rule, and writes only aggregate outputs to `results/paysim_full/`.

The raw parquet shards are not committed to this repository. The workflow also deletes its materialised feature parquet and DuckDB database before publishing results. The workflow has completed successfully multiple times, including relational run `32476220879`; any PaySim metric used in application material must identify PaySim as synthetic and must not use the simulator-balance model as the headline result.
