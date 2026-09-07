# Reproducibility artifact status

## Materials included in this package

* The corrected executable analysis code.
* Exact software-dependency and environment records from bounded validation.
* Documented commands for environment checking, internal self-testing, bounded validation, and full-data execution.

## Data availability

The complete compressed dataset archive used in this study is publicly available through the GitHub release *IoT Zero-Day Intrusion Detection Reproducibility Archive v1.0.0*: https://github.com/osmanyakubu-maker/iot-zero-day-intrusion-detection/releases/tag/v1.0.0.

The archive is distributed as 24 multipart RAR volumes. All volumes should be downloaded into the same directory and extracted beginning with `Data.part01.rar`.

## Scope of the packaged artifacts

The numerical results reported in the manuscript were obtained from the authors' full-data experiments. This reproduction supplement provides the corrected code, execution instructions, and validated software environment needed to rerun the analyses using the public dataset archive. The bounded-validation files included here confirm executable pipeline behavior but are not presented as substitutes for the authors' full-data results.

For the strongest independent numerical audit, the matching full-run artifacts—including `final\\\_results.json`, fold-level predictions, fitted-model metadata, tables, figures, environment locks, and execution logs—may also be deposited alongside this package. Their absence from the present ZIP should not be interpreted as evidence that the full-data experiments were not conducted.

