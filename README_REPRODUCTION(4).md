# IoT Zero-Day Intrusion Detection: Reproduction Supplement

This supplement contains the corrected analysis implementation and the environment record used for bounded execution validation. The complete dataset is distributed separately as 24 multipart RAR volumes in the public release:

https://github.com/osmanyakubu-maker/iot-zero-day-intrusion-detection/releases/tag/v1.0.0

## Included files

- `Code_V2_Q1_Revised.py`: corrected executable analysis pipeline.
- `requirements-validation.lock.txt`: exact packages in the environment that passed the code self-test and bounded end-to-end validation.
- `validation_environment_lock.json`: machine-readable record of that validation environment.
- `ARTIFACT_STATUS.md`: scope and evidence-status statement.

## Data preparation

1. Download `Data.part01.rar` through `Data.part24.rar` into the same directory.
2. Begin extraction from `Data.part01.rar` with WinRAR or 7-Zip. The extractor will read the remaining volumes automatically.
3. Place the extracted dataset tree in a directory named `data`, or pass its location with `--data-dir`.
4. Preserve the nested dataset paths produced by extraction. The program recognizes the archived CSV and Parquet layout.

## Environment setup

Python 3.10 or later is recommended. Create and activate a virtual environment, then install the recorded packages:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-validation.lock.txt
```

The lock file records the environment used for validation. If a platform cannot install one exact wheel version, install the nearest compatible version and retain the resulting environment report with the experiment outputs.

## Preflight checks

```bash
python Code_V2_Q1_Revised.py --check-environment
python Code_V2_Q1_Revised.py --self-test
```

## Full-data execution

The default seeds are `42 123 456 789 101112`. A value of zero for `--max-rows-per-dataset` means that the complete release is used.

```bash
python Code_V2_Q1_Revised.py \
  --mode all \
  --data-dir data \
  --output-dir output \
  --seeds 42 123 456 789 101112 \
  --max-rows-per-dataset 0 \
  --with-xai
```

On Windows PowerShell, enter the command on one line or replace each trailing backslash with a backtick.

The `all` mode includes the external IoT-23 stress test. The publication manuscript identifies IoT-23 as an unreported stress-test resource. To reproduce only the quantitative analyses reported in the manuscript, run the following modes separately:

```bash
python Code_V2_Q1_Revised.py --mode zero_day --data-dir data --output-dir output_zero_day
python Code_V2_Q1_Revised.py --mode cross_dataset --data-dir data --output-dir output_cross_dataset
python Code_V2_Q1_Revised.py --mode baseline --data-dir data --output-dir output_baselines
python Code_V2_Q1_Revised.py --mode ablation --data-dir data --output-dir output_ablation
```

Add `--with-xai` to the relevant run when reproducing explanation-quality analyses.

## Expected outputs

Each run writes its outputs below the selected output directory, including:

- `final_results.json`;
- `dataset_manifest.json`;
- `environment_lock.json` and `requirements.lock.txt`;
- fold- or seed-level predictions;
- model files and metadata;
- summary tables; and
- figures.

Configuration A is the primary cross-dataset experiment. Configuration B retains only protocol and flow duration and must be interpreted as exploratory. Configurations C-E are blocked because their audited semantic feature intersection is empty.

## Reproducibility note

The dataset archive and corrected code are sufficient to rerun the analysis. Exact independent confirmation of the manuscript's numerical tables is strengthened by publishing the matching authors' full-data output directory alongside this supplement.
