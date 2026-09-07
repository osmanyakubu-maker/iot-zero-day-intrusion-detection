# IoT Zero-Day Intrusion Detection

## Reproducibility Package

This repository provides the reproducibility materials accompanying the study **“Cross-Dataset Generalization and Explainable Machine Learning for Zero-Day Intrusion Detection in IoT Networks.”**

The repository is intended to support transparent inspection and reproduction of the study's computational workflow, including the corrected executable analysis pipeline, validated software environment, reproduction instructions, and the complete data archive distributed through the associated GitHub Release.

## Repository Contents

- `Code_V2_Q1_Revised.py` — corrected executable analysis pipeline.
- `requirements-validation.lock.txt` — validated software-environment specification used for bounded execution validation.
- `README_REPRODUCTION.md` — detailed reproduction and execution instructions.
- `ARTIFACT_STATUS.md` — status and scope of the reproducibility artifacts.
- `LICENSE` — repository license.

## Complete Data Archive

The complete dataset is distributed through the **v1.0.0 GitHub Release** as **24 multipart RAR volumes**:

https://github.com/osmanyakubu-maker/iot-zero-day-intrusion-detection/releases/tag/v1.0.0

Download **all 24 parts** (`Data.part01.rar` through `Data.part24.rar`) into the same directory before extraction.

Using WinRAR, open or right-click `Data.part01.rar` and select **Extract Here**. WinRAR will automatically combine the multipart volumes and reconstruct the complete archive.

Do not extract the individual parts separately.

## Reproduction

Detailed reproduction instructions are provided in:

`README_REPRODUCTION.md`

The primary executable analysis pipeline is:

`Code_V2_Q1_Revised.py`

The validated package environment is recorded in:

`requirements-validation.lock.txt`

Users should reconstruct the complete data archive before executing analyses that require the full datasets.

## Artifact Integrity

The GitHub Release provides the archived data volumes together with integrity information for verification. Users should ensure that all required archive parts are present before extraction and analysis.

See `ARTIFACT_STATUS.md` for additional information about artifact availability and validation status.

## Release

**Version:** v1.0.0  
**Release:** https://github.com/osmanyakubu-maker/iot-zero-day-intrusion-detection/releases/tag/v1.0.0

## License

This repository is distributed under the license provided in the `LICENSE` file.
