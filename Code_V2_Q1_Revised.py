"""
Leakage-controlled, cross-dataset and explainable machine learning for
zero-day IoT intrusion detection.

Corrected implementation corresponding to manuscript Version 7.

Key guarantees
--------------
* Dataset splitting occurs before imputation, encoding, scaling, feature
  selection, resampling, calibration, or threshold selection.
* A zero-day test contains held-out-family attacks AND an untouched benign
  subset, so NADR, NAFPR, ZD-F1, ROC AUC, and PR AUC are identifiable.
* Unsupervised domain adaptation uses only an unlabeled adaptation partition;
  a separate locked test partition is never used for fitting.
* MMD and adversarial losses backpropagate through the encoder.
* DA, SMOTE-ENN, and attention/XAI ablation switches alter the trained model.
* Macro F1, strict held-out-family ROC/PR, and unknown-alert NAFPR are distinct.
* Protocol-matched SECL, AE, LSTM-AE, CNN, LightGBM, and XGBoost baselines run.
* Dataset hashes, semantic mappings, model artifacts, predictions, and locks are archived.
* All reported outputs are generated from run-level predictions. No historical
  manuscript result is embedded in this file.

Example
-------
python Code_V1.py --mode all --data-dir ./data --output-dir ./output
python Code_V1.py --self-test
python Code_V1.py --check-environment
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import math
import os
import pickle
import platform
import random
import re
try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

# Heavy dependencies are optional at import time so --help, --self-test, and
# --check-environment remain usable on a machine that has not yet installed the
# complete experiment environment.
OPTIONAL_IMPORT_ERRORS: dict[str, str] = {}

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as exc:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    optim = None
    OPTIONAL_IMPORT_ERRORS["torch"] = str(exc)

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover - environment dependent
    lgb = None
    OPTIONAL_IMPORT_ERRORS["lightgbm"] = str(exc)

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - environment dependent
    xgb = None
    OPTIONAL_IMPORT_ERRORS["xgboost"] = str(exc)

try:
    import optuna
except Exception as exc:  # pragma: no cover - environment dependent
    optuna = None
    OPTIONAL_IMPORT_ERRORS["optuna"] = str(exc)

try:
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE, SMOTENC
    from imblearn.under_sampling import EditedNearestNeighbours
except Exception as exc:  # pragma: no cover - environment dependent
    SMOTEENN = None
    SMOTE = None
    SMOTENC = None
    EditedNearestNeighbours = None
    OPTIONAL_IMPORT_ERRORS["imbalanced-learn"] = str(exc)

try:
    import shap
except Exception as exc:  # pragma: no cover - environment dependent
    shap = None
    OPTIONAL_IMPORT_ERRORS["shap"] = str(exc)

try:
    import lime.lime_tabular
except Exception as exc:  # pragma: no cover - environment dependent
    lime = None
    OPTIONAL_IMPORT_ERRORS["lime"] = str(exc)

try:
    import psutil
except Exception:  # pragma: no cover - optional measurement helper
    psutil = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover - required only for Parquet releases
    pa = None
    pq = None
    OPTIONAL_IMPORT_ERRORS["pyarrow"] = str(exc)

warnings.filterwarnings("ignore", category=FutureWarning)


class Config:
    """Central experiment configuration. Paths are portable and CLI-overridable."""

    DATA_DIR = Path(os.environ.get("IOT_IDS_DATA_DIR", Path.cwd() / "data"))
    OUTPUT_DIR = Path(os.environ.get("IOT_IDS_OUTPUT_DIR", Path.cwd() / "output"))
    MODEL_DIR = OUTPUT_DIR / "models"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    TABLE_DIR = OUTPUT_DIR / "tables"
    FIGURE_DIR = OUTPUT_DIR / "figures"

    RANDOM_SEEDS = [42, 123, 456, 789, 101112]
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    TARGET_ADAPT_FRACTION = 0.30
    ZERO_DAY_ADAPT_FRACTION = 0.30

    AE_LAYERS = [128, 64, 32]
    AE_DROPOUT = 0.20
    AE_LEARNING_RATE = 0.001
    AE_BATCH_SIZE = 256
    AE_EPOCHS = 100
    AE_EARLY_STOP_PATIENCE = 10
    AE_WEIGHT_DECAY = 0.001
    AE_NOISE_STD = 0.10
    BASELINE_EPOCHS = 50
    BASELINE_PATIENCE = 7
    SECL_TEMPERATURE = 0.10
    SECL_CONTRASTIVE_WEIGHT = 0.20

    DA_MMD_LAMBDA = 0.10
    DA_ADV_GAMMA = 0.05
    DA_DISCRIMINATOR_HIDDEN = 64
    DA_EPOCHS = 30
    DA_EARLY_STOP_PATIENCE = 5

    SMOTE_K_NEIGHBORS = 5
    ENN_K_NEIGHBORS = 5
    FEATURE_COUNT = 28
    CROSS_DATASET_CAUTION_FEATURES = 5
    OPTUNA_TRIALS = 20
    INNER_CV_FOLDS = 3

    LIME_NUM_SAMPLES = 3000
    XAI_AUDIT_MAX = 100
    XAI_FIDELITY_THRESHOLD = 0.85
    XAI_STABILITY_REPEATS = 5
    XAI_TOP_K = 10
    RUN_XAI = False

    CSV_CHUNK_SIZE = 250_000
    MAX_ROWS_PER_DATASET = 0  # 0 means use the complete release.
    SEMANTIC_MANIFEST: dict[str, Any] = {}
    SUPPORTED_DATA_SUFFIXES = {".csv", ".parquet", ".pq"}

    DATASET_FILENAMES = {
        "CICIoT2023": [
            "1 CICIoT2023/CICIOT23/train",
            "1 CICIoT2023/CICIOT23/validation",
            "1 CICIoT2023/CICIOT23/test",
            "CICIoT2023.csv", "CIC_IoT_2023.csv",
        ],
        "Edge-IIoTset": [
            "2 EdgeIIoT-dataset/DNN-EdgeIIoT-dataset.csv",
            "Edge-IIoTset.csv", "EdgeIIoTset.csv",
        ],
        "TON_IoT": ["3 TON_IoT/train_test_network.csv", "TON_IoT.csv", "ToN_IoT.csv"],
        "WUSTL-IIOT-2021": [
            "4 wustl_iiot_2021/wustl_iiot_2021.csv",
            "WUSTL-IIOT-2021.csv", "WUSTL_IIOT_2021.csv",
        ],
        "BoT-IoT": ["5 BoT-IoT/DDoSdata.csv", "BoT-IoT.csv", "Bot_IoT.csv"],
        "UNSW-NB15": [
            "6 UNSW_NB15/UNSW_NB15_training-set.parquet",
            "6 UNSW_NB15/UNSW_NB15_testing-set.parquet",
            "UNSW-NB15.csv", "UNSW_NB15.csv",
        ],
        "MQTT-IoT-IDS2020": [
            "7 MQTT-IoT-IDS2020",
            "MQTT-IoT-IDS2020.csv", "MQTTset.csv",
        ],
        "NF-UQ-NIDS-v2": [
            "8 NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv",
            "NF-UQ-NIDS-v2.csv", "NF_UQ_NIDS_v2.csv",
        ],
        "IoT-23": ["9 IoT-23/iot.parquet", "IoT-23.csv", "IoT23.csv"],
    }

    LABEL_ALIASES = {
        "CICIoT2023": ["label", "class"],
        "Edge-IIoTset": ["attack_label", "label", "class"],
        "TON_IoT": ["label", "type", "attack_type"],
        "WUSTL-IIOT-2021": ["target", "label", "attack", "attack_type"],
        "BoT-IoT": ["attack", "label", "category"],
        "UNSW-NB15": ["label", "attack_cat"],
        "MQTT-IoT-IDS2020": ["is_attack", "label", "attack", "class"],
        "NF-UQ-NIDS-v2": ["label", "attack_type", "class"],
        "IoT-23": ["labels", "label", "detailed_label", "class"],
    }

    FAMILY_ALIASES = {
        "CICIoT2023": ["attack_category", "label", "class"],
        "Edge-IIoTset": ["attack_type"],
        "TON_IoT": ["attack_type", "type"],
        "WUSTL-IIOT-2021": ["traffic", "attack_type", "attack"],
        "BoT-IoT": ["category", "subcategory"],
        "UNSW-NB15": ["attack_cat"],
        "MQTT-IoT-IDS2020": ["attack", "label"],
        "NF-UQ-NIDS-v2": ["attack", "attack_type", "label"],
        "IoT-23": ["detailed_label", "attack", "label"],
    }

    GROUP_ALIASES = [
        "flow_id", "session_id", "scenario", "capture_id", "device_id",
        "src_ip", "source_ip", "connection_id",
    ]
    TIME_ALIASES = ["timestamp", "stime", "start_time", "ts", "date", "time"]

    # Ordered semantic aliases. Only the first available raw field is retained
    # for each canonical feature, preventing duplicate semantic columns.
    CANONICAL_FEATURE_ALIASES = {
        "flow_duration": ["flow_duration", "flow_dur", "dur", "conn_time", "duration"],
        "protocol": ["protocol", "proto", "protocol_type"],
        "src_port": ["src_port", "sport", "source_port", "prt_src", "l4_src_port", "id_orig_p", "tcp.srcport"],
        "dst_port": ["dst_port", "dport", "destination_port", "prt_dst", "l4_dst_port", "id_resp_p", "tcp.dstport"],
        "fwd_pkts": ["fwd_pkts", "tot_fwd_pkts", "spkts", "src_pkts", "srcpkts", "fwd_num_pkts", "in_pkts", "orig_pkts"],
        "bwd_pkts": ["bwd_pkts", "tot_bwd_pkts", "dpkts", "dst_pkts", "dstpkts", "bwd_num_pkts", "out_pkts", "resp_pkts"],
        "pkt_count": ["total_packets", "pkt_count", "pkts", "totpkts"],
        "fwd_bytes": ["fwd_bytes", "totlen_fwd_pkts", "sbytes", "src_bytes", "srcbytes", "fwd_num_bytes", "in_bytes", "orig_bytes"],
        "bwd_bytes": ["bwd_bytes", "totlen_bwd_pkts", "dbytes", "dst_bytes", "dstbytes", "bwd_num_bytes", "out_bytes", "resp_bytes"],
        "byte_count": ["total_bytes", "byte_count", "tot_size", "bytes", "totbytes"],
        "pkt_size_mean": ["avg_pkt_size", "pkt_size_mean", "flow_pkt_len_mean", "avg"],
        "pkt_size_std": ["std_pkt_size", "pkt_size_std", "flow_pkt_len_std", "std", "stddev"],
        "pkt_size_min": ["pkt_len_min", "min_pkt_len", "min"],
        "pkt_size_max": ["pkt_len_max", "max_pkt_len", "max"],
        "fwd_pkt_len_mean": ["fwd_pkt_len_mean", "fwd_mean_pkt_len", "smean"],
        "bwd_pkt_len_mean": ["bwd_pkt_len_mean", "bwd_mean_pkt_len", "dmean"],
        "fwd_pkt_len_std": ["fwd_pkt_len_std", "fwd_std_pkt_len"],
        "bwd_pkt_len_std": ["bwd_pkt_len_std", "bwd_std_pkt_len"],
        "flow_iat_mean": ["flow_iat_mean", "avg_interarrival", "iat_mean", "iat"],
        "flow_iat_std": ["flow_iat_std", "iat_std"],
        "fwd_iat_mean": ["fwd_iat_mean", "fwd_mean_iat", "sinpkt"],
        "bwd_iat_mean": ["bwd_iat_mean", "bwd_mean_iat", "dinpkt"],
        "fwd_iat_std": ["fwd_iat_std", "fwd_std_iat"],
        "bwd_iat_std": ["bwd_iat_std", "bwd_std_iat"],
        "syn_flag_count": ["syn_flag_count", "syn_flag_number", "syn_count"],
        "ack_flag_count": ["ack_flag_count", "ack_flag_number", "ack_count"],
        "rst_flag_count": ["rst_flag_count", "rst_flag_number", "rst_count"],
        "fin_flag_count": ["fin_flag_count", "fin_flag_number", "fin_count"],
        "rate": ["rate", "flow_packets_s", "pkt_rate"],
        "src_rate": ["srate", "src_rate", "srcrate"],
        "dst_rate": ["drate", "dst_rate", "dstrate"],
    }

    CROSS_DATASET_CONFIGS = {
        "A": {"source": ["CICIoT2023"], "target": "WUSTL-IIOT-2021"},
        "B": {"source": ["CICIoT2023", "TON_IoT"], "target": "BoT-IoT"},
    }
    EXCLUDED_CROSS_DATASET_CONFIGS = {
        "C": {
            "source": ["Edge-IIoTset", "CICIoT2023"], "target": "UNSW-NB15",
            "reason": "No semantically equivalent common feature survives the audited intersection.",
        },
        "D": {
            "source": ["CICIoT2023", "Edge-IIoTset"], "target": "MQTT-IoT-IDS2020",
            "reason": "No semantically equivalent common feature survives the audited intersection.",
        },
        "E": {
            "source": ["CICIoT2023", "Edge-IIoTset", "TON_IoT"], "target": "NF-UQ-NIDS-v2",
            "reason": "No semantically equivalent common feature survives the audited intersection.",
        },
    }

    @classmethod
    def configure_paths(cls, data_dir: str | Path, output_dir: str | Path) -> None:
        cls.DATA_DIR = Path(data_dir).expanduser().resolve()
        cls.OUTPUT_DIR = Path(output_dir).expanduser().resolve()
        cls.MODEL_DIR = cls.OUTPUT_DIR / "models"
        cls.PREDICTIONS_DIR = cls.OUTPUT_DIR / "predictions"
        cls.TABLE_DIR = cls.OUTPUT_DIR / "tables"
        cls.FIGURE_DIR = cls.OUTPUT_DIR / "figures"

    @classmethod
    def ensure_output_dirs(cls) -> None:
        for directory in [
            cls.OUTPUT_DIR, cls.MODEL_DIR, cls.PREDICTIONS_DIR,
            cls.TABLE_DIR, cls.FIGURE_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


META_COLUMNS = {"label", "family", "__group__", "__time__", "__dataset__"}
DATASET_AUDIT: dict[str, dict[str, Any]] = {}
FILE_HASH_CACHE: dict[str, str] = {}


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _normalise_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def require_training_dependencies(
    require_xai: bool = False, require_baselines: bool = False
) -> None:
    required = ["torch", "lightgbm", "imbalanced-learn"]
    if Config.OPTUNA_TRIALS > 0:
        required.append("optuna")
    if require_xai:
        required.extend(["shap", "lime"])
    if require_baselines:
        required.append("xgboost")
    missing = [name for name in required if name in OPTIONAL_IMPORT_ERRORS]
    if missing:
        details = "; ".join(f"{name}: {OPTIONAL_IMPORT_ERRORS[name]}" for name in missing)
        raise RuntimeError(
            "Missing experiment dependencies. Install the packages listed in "
            f"--check-environment. Details: {details}"
        )


def environment_report() -> dict[str, Any]:
    import scipy
    import sklearn

    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for name, module in [
        ("torch", torch), ("lightgbm", lgb), ("xgboost", xgb), ("optuna", optuna),
        ("shap", shap), ("psutil", psutil), ("pyarrow", pa),
    ]:
        versions[name] = getattr(module, "__version__", None) if module is not None else None
    versions["missing_or_failed"] = OPTIONAL_IMPORT_ERRORS
    versions["required_install"] = (
        "numpy pandas scipy scikit-learn torch lightgbm xgboost optuna "
        "imbalanced-learn shap lime matplotlib psutil pyarrow"
    )
    return versions


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a dataset into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_environment_lock(output_dir: str | Path) -> dict[str, Any]:
    packages = [
        "numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm", "xgboost",
        "optuna", "imbalanced-learn", "shap", "lime", "matplotlib", "psutil", "pyarrow",
    ]
    resolved: dict[str, str | None] = {}
    for package in packages:
        try:
            resolved[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            resolved[package] = None
    output_dir = Path(output_dir)
    save_json({"python": sys.version, "packages": resolved}, output_dir / "environment_lock.json")
    lock_path = output_dir / "requirements.lock.txt"
    with lock_path.open("w", encoding="utf-8") as handle:
        for package, version in resolved.items():
            if version is not None:
                handle.write(f"{package}=={version}\n")
    return {"environment_lock": str(output_dir / "environment_lock.json"), "requirements_lock": str(lock_path)}


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(data), handle, indent=2, sort_keys=True)


def _resolve_column(columns: Sequence[Any], aliases: Sequence[str]) -> str | None:
    normalised = {_normalise_name(column): str(column) for column in columns}
    for alias in aliases:
        match = normalised.get(_normalise_name(alias))
        if match is not None:
            return match
    return None


def _binary_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        return (numeric > 0).astype(np.int8)
    benign = {"0", "benign", "benigntraffic", "normal", "normaltraffic", "background"}
    return series.map(lambda value: 0 if _normalise_label(value) in benign else 1).astype(np.int8)


def cic_family(value: Any) -> str:
    """Map all official CICIoT2023 attacks to its seven published families."""
    label = _normalise_label(value)
    if label in {"0", "benign", "benigntraffic", "normal", "normaltraffic"}:
        return "benign"
    if label.startswith("ddos"):
        return "DDoS"
    if label.startswith("dos"):
        return "DoS"
    if "mirai" in label or any(token in label for token in ["greip", "greeth", "udpplain"]):
        return "Mirai"
    if "bruteforce" in label or "dictionary" in label:
        return "Brute_Force"
    if "spoof" in label:
        return "Spoofing"
    if any(token in label for token in ["recon", "scan", "hostdiscovery", "pingsweep"]):
        return "Recon"
    if any(token in label for token in [
        "sqlinjection", "commandinjection", "xss", "backdoor",
        "uploading", "browserhijacking", "web",
    ]):
        return "Web_Based"
    return "unknown"


def _generic_family(value: Any, binary: int) -> str:
    if binary == 0:
        return "benign"
    cleaned = _normalise_name(value)
    return cleaned if cleaned and cleaned not in {"0", "1", "attack", "malicious"} else "attack"


def locate_dataset_files(dataset_name: str) -> list[Path]:
    """Resolve supported files or official release shards for one dataset."""
    candidates = Config.DATASET_FILENAMES.get(dataset_name, [f"{dataset_name}.csv"])
    resolved: list[Path] = []
    for candidate in candidates:
        path = Config.DATA_DIR / candidate
        if path.is_file() and path.suffix.lower() in Config.SUPPORTED_DATA_SUFFIXES:
            resolved.append(path)
        elif path.is_dir():
            resolved.extend(
                sorted(
                    item for item in path.rglob("*")
                    if item.is_file() and item.suffix.lower() in Config.SUPPORTED_DATA_SUFFIXES
                )
            )
    if resolved:
        return sorted(set(path.resolve() for path in resolved))

    normalised_target = {
        _normalise_name(Path(name).stem) for name in [dataset_name, *candidates]
    }
    if Config.DATA_DIR.exists():
        for path in Config.DATA_DIR.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in Config.SUPPORTED_DATA_SUFFIXES:
                continue
            normalised_parts = {_normalise_name(part) for part in path.parts}
            if _normalise_name(path.stem) in normalised_target or normalised_parts & normalised_target:
                resolved.append(path)
    if resolved:
        return sorted(set(path.resolve() for path in resolved))
    raise FileNotFoundError(
        f"Dataset {dataset_name!r} not found under {Config.DATA_DIR}. "
        f"Expected one of {', '.join(candidates)} or a same-named directory of CSV/Parquet shards."
    )


def locate_dataset(dataset_name: str) -> Path:
    """Backward-compatible single-path resolver; prefer locate_dataset_files."""
    return locate_dataset_files(dataset_name)[0]


def _canonical_mapping(
    columns: Sequence[Any], requested: set[str] | None = None, dataset_name: str | None = None
) -> dict[str, str]:
    explicit = Config.SEMANTIC_MANIFEST.get(dataset_name or "", {}).get("features", {})
    if explicit:
        column_lookup = {_normalise_name(column): str(column) for column in columns}
        mapping: dict[str, str] = {}
        for canonical, specification in explicit.items():
            if requested is not None and canonical not in requested:
                continue
            raw_name = specification.get("column") if isinstance(specification, dict) else specification
            match = column_lookup.get(_normalise_name(raw_name))
            if match is None:
                raise ValueError(
                    f"Semantic manifest column {raw_name!r} for {dataset_name}/{canonical} was not found"
                )
            mapping[str(canonical)] = match
        return mapping
    normalised_columns: dict[str, list[str]] = {}
    for raw_col in columns:
        normalised_columns.setdefault(_normalise_name(raw_col), []).append(str(raw_col))
    mapping: dict[str, str] = {}
    for canonical, aliases in Config.CANONICAL_FEATURE_ALIASES.items():
        if requested is not None and canonical not in requested:
            continue
        for alias in aliases:
            matches = normalised_columns.get(_normalise_name(alias), [])
            if matches:
                mapping[canonical] = matches[0]
                break
    return mapping


def _columns_for_path(path: Path) -> Sequence[Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=0).columns
    if suffix in {".parquet", ".pq"}:
        if pq is None:
            raise RuntimeError(
                f"Reading {path.name} requires pyarrow. Install it with: pip install pyarrow"
            )
        return pq.ParquetFile(path).schema_arrow.names
    raise ValueError(f"Unsupported dataset format: {path}")


def _iter_raw_chunks(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, low_memory=False, chunksize=Config.CSV_CHUNK_SIZE)
        return
    if suffix in {".parquet", ".pq"}:
        if pq is None:
            raise RuntimeError(
                f"Reading {path.name} requires pyarrow. Install it with: pip install pyarrow"
            )
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=Config.CSV_CHUNK_SIZE):
            yield batch.to_pandas()
        return
    raise ValueError(f"Unsupported dataset format: {path}")


def available_features(dataset_name: str) -> set[str]:
    """Discover semantic features from headers only, avoiding a full data load."""
    discovered: set[str] | None = None
    for path in locate_dataset_files(dataset_name):
        columns = _columns_for_path(path)
        current = set(_canonical_mapping(columns, dataset_name=dataset_name))
        discovered = current if discovered is None else discovered & current
    return discovered or set()


def _harmonise_chunk(
    raw: pd.DataFrame,
    dataset_name: str,
    feature_list: Sequence[str] | None,
    source_path: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if raw.empty:
        return pd.DataFrame(), {}

    manifest = Config.SEMANTIC_MANIFEST.get(dataset_name, {})
    label_col = _resolve_column(
        raw.columns,
        [manifest["label_column"]]
        if manifest.get("label_column")
        else Config.LABEL_ALIASES.get(dataset_name, ["label"]),
    )
    if label_col is None:
        raise ValueError(f"No label column found for {dataset_name}. Columns: {list(raw.columns)[:30]}")
    binary = _binary_label(raw[label_col])

    family_col = _resolve_column(
        raw.columns,
        [manifest["family_column"]]
        if manifest.get("family_column")
        else Config.FAMILY_ALIASES.get(dataset_name, []),
    )
    family_source = raw[family_col] if family_col else raw[label_col]
    if dataset_name == "CICIoT2023":
        family = family_source.map(cic_family)
    else:
        family = pd.Series(
            [_generic_family(value, int(y)) for value, y in zip(family_source, binary)],
            index=raw.index,
        )

    requested = set(feature_list) if feature_list is not None else None
    mapping = _canonical_mapping(raw.columns, requested, dataset_name)
    output: dict[str, Any] = {}
    manifest_features = manifest.get("features", {})
    for canonical, raw_column in mapping.items():
        series = raw[raw_column]
        specification = manifest_features.get(canonical, {})
        if isinstance(specification, dict):
            multiplier = float(specification.get("multiplier", 1.0))
            offset = float(specification.get("offset", 0.0))
            if multiplier != 1.0 or offset != 0.0:
                series = pd.to_numeric(series, errors="coerce") * multiplier + offset
        output[canonical] = series

    if not output:
        raise ValueError(
            f"No semantically recognised features found for {dataset_name} in {source_path}. "
            "Extend Config.CANONICAL_FEATURE_ALIASES after auditing the dataset dictionary."
        )

    frame = pd.DataFrame(output, index=raw.index)
    frame["label"] = binary
    frame["family"] = family
    frame["__dataset__"] = dataset_name

    group_col = _resolve_column(raw.columns, Config.GROUP_ALIASES)
    if group_col:
        frame["__group__"] = dataset_name + ":" + raw[group_col].astype(str)
    time_col = _resolve_column(raw.columns, Config.TIME_ALIASES)
    if time_col:
        frame["__time__"] = raw[time_col]

    return frame, mapping


def load_dataset(dataset_name: str, feature_list: Sequence[str] | None = None) -> pd.DataFrame:
    """Load CSV/Parquet shards in bounded chunks and harmonise their semantics."""
    paths = locate_dataset_files(dataset_name)
    print(f"Loading {dataset_name} from {len(paths)} data file(s)")
    chunks: list[pd.DataFrame] = []
    raw_rows = 0
    mappings: dict[str, dict[str, str]] = {}
    remaining = Config.MAX_ROWS_PER_DATASET or None
    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        for chunk in _iter_raw_chunks(path):
            raw_rows += len(chunk)
            if remaining is not None and len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
            harmonised, mapping = _harmonise_chunk(chunk, dataset_name, feature_list, path)
            chunks.append(harmonised)
            mappings[str(path)] = mapping
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    if not chunks:
        raise ValueError(f"Dataset {dataset_name} is empty: {paths}")

    frame = pd.concat(chunks, ignore_index=True, sort=False)
    required_features = list(feature_list) if feature_list is not None else feature_columns(frame)
    missing = [name for name in required_features if name not in frame]
    if missing:
        raise ValueError(f"Missing harmonised features for {dataset_name}: {missing}")
    subset = [column for column in frame.columns if column not in META_COLUMNS]
    frame = frame.drop_duplicates(subset=subset + ["label", "family"]).reset_index(drop=True)
    hashes = {}
    for path in paths:
        key = str(path)
        if key not in FILE_HASH_CACHE:
            FILE_HASH_CACHE[key] = sha256_file(path)
        hashes[key] = FILE_HASH_CACHE[key]
    DATASET_AUDIT[dataset_name] = {
        "files": hashes,
        "raw_rows_read": raw_rows,
        "retained_unique_rows": len(frame),
        "features": subset,
        "raw_to_canonical": mappings,
        "row_cap": Config.MAX_ROWS_PER_DATASET or None,
        "attack_prevalence": float(frame["label"].mean()),
    }
    print(
        f"  retained {len(frame):,} unique rows, {len(subset)} canonical features, "
        f"attack prevalence={frame['label'].mean():.3f}"
    )
    return frame


def get_available_families(frame: pd.DataFrame) -> list[str]:
    families = sorted(set(frame.loc[frame["label"] == 1, "family"]) - {"unknown", "attack"})
    if len(families) < 2:
        raise ValueError(
            "Fewer than two named attack families were recovered. Audit the family column "
            "and the label-to-family mapping before running leave-family-out evaluation."
        )
    return families


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in META_COLUMNS]


def common_features(source_datasets: Sequence[str], target_dataset: str) -> list[str]:
    sets = [available_features(name) for name in [*source_datasets, target_dataset]]
    common = sorted(set.intersection(*sets))
    if not common:
        raise ValueError(f"No audited common features for {source_datasets} -> {target_dataset}")
    if len(common) < Config.CROSS_DATASET_CAUTION_FEATURES:
        warnings.warn(
            f"Only {len(common)} audited common features remain for "
            f"{source_datasets} -> {target_dataset}; treat this configuration as "
            "an exploratory feasibility analysis, not a broad transfer claim.",
            RuntimeWarning,
        )
    print(f"Common feature set ({source_datasets} -> {target_dataset}): {common}")
    return common


def _safe_stratify(labels: pd.Series | np.ndarray) -> np.ndarray | None:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return np.asarray(labels) if len(values) > 1 and counts.min() >= 2 else None


def split_source_frame(
    frame: pd.DataFrame,
    seed: int,
    train_size: float = Config.TRAIN_SIZE,
    val_size: float = Config.VAL_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-aware, time-aware, or stratified source train/validation/test split."""
    test_size = 1.0 - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Train, validation, and test fractions must all be positive")

    indices = np.arange(len(frame))
    if "__group__" in frame and frame["__group__"].nunique() >= 3:
        groups = frame["__group__"].astype(str).unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        n_groups = len(groups)
        n_train_groups = min(n_groups - 2, max(1, int(round(n_groups * train_size))))
        n_val_groups = min(
            n_groups - n_train_groups - 1,
            max(1, int(round(n_groups * val_size))),
        )
        train_groups = set(groups[:n_train_groups])
        val_groups = set(groups[n_train_groups:n_train_groups + n_val_groups])
        test_groups = set(groups[n_train_groups + n_val_groups:])
        group_values = frame["__group__"].astype(str)
        train_idx = indices[group_values.isin(train_groups)]
        val_idx = indices[group_values.isin(val_groups)]
        test_idx = indices[group_values.isin(test_groups)]
    elif "__time__" in frame:
        ordered = frame.assign(__order=pd.to_datetime(frame["__time__"], errors="coerce"))
        if ordered["__order"].notna().mean() < 0.80:
            ordered["__order"] = pd.to_numeric(frame["__time__"], errors="coerce")
        if ordered["__order"].notna().mean() >= 0.80:
            sorted_idx = ordered.sort_values("__order", kind="mergesort").index.to_numpy()
            n_train = int(len(sorted_idx) * train_size)
            n_val = int(len(sorted_idx) * val_size)
            train_idx = sorted_idx[:n_train]
            val_idx = sorted_idx[n_train:n_train + n_val]
            test_idx = sorted_idx[n_train + n_val:]
        else:
            train_idx, val_idx, test_idx = _random_source_indices(frame, seed, test_size, val_size)
    else:
        train_idx, val_idx, test_idx = _random_source_indices(frame, seed, test_size, val_size)

    partitions = tuple(
        frame.iloc[idx].reset_index(drop=True) for idx in [train_idx, val_idx, test_idx]
    )
    for name, partition in zip(["training", "validation"], partitions[:2]):
        if partition.empty or partition["label"].nunique() < 2:
            raise ValueError(
                f"The group/time-aware {name} partition lacks both benign and attack records. "
                "Revise the prespecified grouping or temporal window; do not fall back silently."
            )
    return partitions


def _random_source_indices(
    frame: pd.DataFrame, seed: int, test_size: float, val_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=test_size + val_size,
        random_state=seed,
        stratify=_safe_stratify(frame["label"]),
    )
    temp_labels = frame.iloc[temp_idx]["label"]
    relative_test = test_size / (test_size + val_size)
    val_rel, test_rel = train_test_split(
        np.arange(len(temp_idx)),
        test_size=relative_test,
        random_state=seed + 1,
        stratify=_safe_stratify(temp_labels),
    )
    return train_idx, temp_idx[val_rel], temp_idx[test_rel]


def split_adaptation_test(
    frame: pd.DataFrame, seed: int, adapt_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split without consulting target labels."""
    if adapt_fraction <= 0:
        return frame.iloc[0:0].copy(), frame.reset_index(drop=True)
    if not 0 < adapt_fraction < 1:
        raise ValueError("adapt_fraction must be in [0, 1)")
    indices = np.arange(len(frame))
    if "__group__" in frame and frame["__group__"].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, train_size=adapt_fraction, random_state=seed)
        adapt_idx, test_idx = next(splitter.split(indices, groups=frame["__group__"]))
    elif "__time__" in frame:
        ordered = pd.to_datetime(frame["__time__"], errors="coerce")
        if ordered.notna().mean() >= 0.80:
            sorted_idx = np.argsort(ordered.to_numpy())
            cut = max(1, int(len(sorted_idx) * adapt_fraction))
            adapt_idx, test_idx = sorted_idx[:cut], sorted_idx[cut:]
        else:
            adapt_idx, test_idx = train_test_split(indices, train_size=adapt_fraction, random_state=seed)
    else:
        adapt_idx, test_idx = train_test_split(indices, train_size=adapt_fraction, random_state=seed)
    return frame.iloc[adapt_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)


class SourceFittedPreprocessor:
    """Median/unknown-category encoding and min-max scaling fitted on source train only."""

    def __init__(self) -> None:
        self.features: list[str] = []
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.medians: dict[str, float] = {}
        self.vocabularies: dict[str, dict[str, int]] = {}
        self.scaler = MinMaxScaler()
        self.fitted = False

    def fit(self, frame: pd.DataFrame, features: Sequence[str]) -> "SourceFittedPreprocessor":
        self.features = list(features)
        self.numeric_features = []
        self.categorical_features = []
        self.medians = {}
        self.vocabularies = {}
        encoded = pd.DataFrame(index=frame.index)
        for feature in self.features:
            series = frame[feature]
            numeric = pd.to_numeric(series, errors="coerce")
            if pd.api.types.is_numeric_dtype(series) or numeric.notna().mean() >= 0.95:
                self.numeric_features.append(feature)
                median = float(numeric.median()) if numeric.notna().any() else 0.0
                self.medians[feature] = median
                encoded[feature] = numeric.fillna(median).astype(float)
            else:
                self.categorical_features.append(feature)
                values = series.fillna("__unknown__").astype(str)
                vocabulary = {value: idx + 1 for idx, value in enumerate(sorted(values.unique()))}
                self.vocabularies[feature] = vocabulary
                encoded[feature] = values.map(vocabulary).fillna(0).astype(float)
        self.scaler.fit(encoded[self.features])
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        encoded = pd.DataFrame(index=frame.index)
        for feature in self.features:
            if feature in self.numeric_features:
                encoded[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(
                    self.medians[feature]
                )
            else:
                values = frame[feature].fillna("__unknown__").astype(str)
                encoded[feature] = values.map(self.vocabularies[feature]).fillna(0)
        scaled = self.scaler.transform(encoded[self.features].astype(float))
        return pd.DataFrame(scaled, columns=self.features, index=frame.index).reset_index(drop=True)

    def fit_transform(self, frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
        return self.fit(frame, features).transform(frame)


@dataclass
class ExperimentData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    family_train: np.ndarray
    family_val: np.ndarray
    family_test: np.ndarray
    novel_test: np.ndarray
    X_adapt: pd.DataFrame
    features: list[str]
    categorical_features: list[str]
    preprocessor: SourceFittedPreprocessor
    evaluation_kind: str


def experiment_data_summary(data: ExperimentData) -> dict[str, Any]:
    def partition_summary(labels: np.ndarray) -> dict[str, Any]:
        labels = np.asarray(labels, dtype=int)
        return {
            "rows": int(len(labels)),
            "benign": int(np.sum(labels == 0)),
            "attack": int(np.sum(labels == 1)),
            "attack_prevalence": float(np.mean(labels)) if len(labels) else np.nan,
        }

    return {
        "training": partition_summary(data.y_train),
        "validation": partition_summary(data.y_val),
        "test": partition_summary(data.y_test),
        "adaptation_rows": int(len(data.X_adapt)),
        "novel_test_rows": int(np.sum(data.novel_test)),
        "features_before_selection": len(data.features),
        "evaluation_kind": data.evaluation_kind,
    }


def prepare_zero_day_data(
    frame: pd.DataFrame,
    holdout_family: str,
    seed: int,
    adapt_fraction: float = Config.ZERO_DAY_ADAPT_FRACTION,
) -> ExperimentData:
    holdout = frame[(frame["family"] == holdout_family) & (frame["label"] == 1)].copy()
    known = frame[frame["family"] != holdout_family].copy()
    if holdout.empty:
        raise ValueError(f"No observations for held-out family {holdout_family}")

    train_raw, val_raw, known_target_raw = split_source_frame(known, seed)
    known_adapt_raw, known_test_raw = split_adaptation_test(
        known_target_raw, seed + 13, adapt_fraction
    )
    novel_adapt_raw, novel_test_raw = split_adaptation_test(
        holdout, seed + 17, adapt_fraction
    )
    adapt_raw = pd.concat([known_adapt_raw, novel_adapt_raw], ignore_index=True)
    if not adapt_raw.empty:
        adapt_raw = adapt_raw.sample(frac=1.0, random_state=seed + 19).reset_index(drop=True)
    test_raw = pd.concat([known_test_raw, novel_test_raw], ignore_index=True)
    novel_mask = np.concatenate(
        [np.zeros(len(known_test_raw), dtype=np.int8), np.ones(len(novel_test_raw), dtype=np.int8)]
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(test_raw))
    test_raw = test_raw.iloc[order].reset_index(drop=True)
    novel_mask = novel_mask[order]

    features = feature_columns(frame)
    preprocessor = SourceFittedPreprocessor().fit(train_raw, features)
    return ExperimentData(
        X_train=preprocessor.transform(train_raw),
        X_val=preprocessor.transform(val_raw),
        X_test=preprocessor.transform(test_raw),
        y_train=train_raw["label"].to_numpy(dtype=np.int8),
        y_val=val_raw["label"].to_numpy(dtype=np.int8),
        y_test=test_raw["label"].to_numpy(dtype=np.int8),
        family_train=train_raw["family"].astype(str).to_numpy(),
        family_val=val_raw["family"].astype(str).to_numpy(),
        family_test=test_raw["family"].astype(str).to_numpy(),
        novel_test=novel_mask,
        X_adapt=preprocessor.transform(adapt_raw),
        features=features,
        categorical_features=preprocessor.categorical_features,
        preprocessor=preprocessor,
        evaluation_kind="zero_day",
    )


def prepare_cross_dataset_data(
    source_datasets: Sequence[str], target_dataset: str, seed: int
) -> ExperimentData:
    features = common_features(source_datasets, target_dataset)
    sources = []
    for name in source_datasets:
        frame = load_dataset(name, feature_list=features)
        sources.append(frame)
    source = pd.concat(sources, ignore_index=True)
    target = load_dataset(target_dataset, feature_list=features)

    train_raw, val_raw, _ = split_source_frame(source, seed)
    adapt_raw, test_raw = split_adaptation_test(target, seed + 31, Config.TARGET_ADAPT_FRACTION)
    preprocessor = SourceFittedPreprocessor().fit(train_raw, features)
    y_test = test_raw["label"].to_numpy(dtype=np.int8)
    return ExperimentData(
        X_train=preprocessor.transform(train_raw),
        X_val=preprocessor.transform(val_raw),
        X_test=preprocessor.transform(test_raw),
        y_train=train_raw["label"].to_numpy(dtype=np.int8),
        y_val=val_raw["label"].to_numpy(dtype=np.int8),
        y_test=y_test,
        family_train=train_raw["family"].astype(str).to_numpy(),
        family_val=val_raw["family"].astype(str).to_numpy(),
        family_test=test_raw["family"].astype(str).to_numpy(),
        novel_test=y_test.copy(),  # target attacks are the cross-domain novelty set
        X_adapt=preprocessor.transform(adapt_raw),
        features=features,
        categorical_features=preprocessor.categorical_features,
        preprocessor=preprocessor,
        evaluation_kind="cross_dataset",
    )


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        attack_probability = y_proba[:, 1]
        metrics["auc_roc"] = roc_auc_score(y_true, attack_probability)
        metrics["auc_pr"] = average_precision_score(y_true, attack_probability)
    else:
        metrics["auc_roc"] = np.nan
        metrics["auc_pr"] = np.nan
    return metrics


def compute_zero_day_metrics(
    y_true: np.ndarray,
    novel_pred: np.ndarray,
    novel_mask: np.ndarray,
    novel_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate held-out-family alerts strictly against benign traffic."""
    y_true = np.asarray(y_true)
    novel_pred = np.asarray(novel_pred)
    novel = np.asarray(novel_mask).astype(bool)
    benign = y_true == 0
    nadr = float(novel_pred[novel].mean()) if novel.any() else np.nan
    nafpr = float(novel_pred[benign].mean()) if benign.any() else np.nan
    evaluation_mask = novel | benign
    zd_f1 = (
        f1_score(novel[evaluation_mask].astype(int), novel_pred[evaluation_mask], zero_division=0)
        if evaluation_mask.any() else np.nan
    )
    zd_auc_roc = np.nan
    zd_auc_pr = np.nan
    if novel_score is not None and evaluation_mask.any():
        target = novel[evaluation_mask].astype(int)
        score = np.asarray(novel_score, dtype=float)[evaluation_mask]
        if len(np.unique(target)) == 2:
            zd_auc_roc = float(roc_auc_score(target, score))
            zd_auc_pr = float(average_precision_score(target, score))
    return {
        "nadr": nadr,
        "nafpr": nafpr,
        "zd_f1": zd_f1,
        "zd_auc_roc": zd_auc_roc,
        "zd_auc_pr": zd_auc_pr,
    }


def hybrid_feature_selection(X: pd.DataFrame, y: np.ndarray, k: int, seed: int) -> list[str]:
    if X.shape[1] <= k:
        return list(X.columns)
    mi = mutual_info_classif(X, y, random_state=seed)
    f_values, _ = f_classif(X, y)
    f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
    mi_norm = (mi - mi.min()) / (np.ptp(mi) + 1e-12)
    f_norm = (f_values - f_values.min()) / (np.ptp(f_values) + 1e-12)
    shortlist_size = min(X.shape[1], max(k, int(math.ceil(1.5 * k))))
    shortlist = np.argsort(0.5 * mi_norm + 0.5 * f_norm)[-shortlist_size:]
    candidates = X.columns[shortlist].tolist()
    forest = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    forest.fit(X[candidates], y)
    order = np.argsort(forest.feature_importances_)[::-1][:k]
    return [candidates[index] for index in order]


def smote_enn_resample(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_features: Sequence[str],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if SMOTEENN is None:
        raise RuntimeError("imbalanced-learn is required for SMOTE-ENN")
    _, counts = np.unique(np.asarray(y), return_counts=True)
    minority = int(counts.min()) if len(counts) else 0
    if minority < 2:
        warnings.warn("SMOTE-ENN skipped because a class has fewer than two samples")
        return X.copy(), np.asarray(y)
    k = min(Config.SMOTE_K_NEIGHBORS, minority - 1)
    categorical_indices = [X.columns.get_loc(name) for name in categorical_features if name in X.columns]
    if categorical_indices:
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy="auto",
            random_state=seed,
            k_neighbors=k,
        )
        X_over, y_over = sampler.fit_resample(X, y)
        cleaner = EditedNearestNeighbours(n_neighbors=Config.ENN_K_NEIGHBORS)
        X_res, y_res = cleaner.fit_resample(X_over, y_over)
    else:
        combined = SMOTEENN(
            smote=SMOTE(random_state=seed, k_neighbors=k),
            enn=EditedNearestNeighbours(n_neighbors=Config.ENN_K_NEIGHBORS),
            random_state=seed,
        )
        X_res, y_res = combined.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), np.asarray(y_res)


class ScoreCalibrator:
    """Platt calibration with a deterministic fallback for one-class validation."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.center = 0.0
        self.scale = 1.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ScoreCalibrator":
        scores = np.asarray(scores, dtype=float).reshape(-1)
        labels = np.asarray(labels, dtype=int)
        self.center = float(np.median(scores))
        self.scale = float(np.std(scores)) or 1.0
        if len(np.unique(labels)) == 2:
            self.model = LogisticRegression(random_state=0, max_iter=1000)
            self.model.fit(scores.reshape(-1, 1), labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if self.model is not None:
            return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        z = np.clip((scores - self.center) / (self.scale + 1e-12), -40, 40)
        return 1.0 / (1.0 + np.exp(-z))


if nn is not None:

    class FeatureAttention(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.Tanh(), nn.Linear(input_dim, input_dim)
            )

        def weights(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.softmax(self.scorer(x), dim=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            weights = self.weights(x)
            return x * (1.0 + weights)


    class DenoisingAutoencoder(nn.Module):
        def __init__(
            self,
            input_dim: int,
            layers: Sequence[int] = Config.AE_LAYERS,
            dropout: float = Config.AE_DROPOUT,
            use_attention: bool = True,
        ) -> None:
            super().__init__()
            if len(layers) != 3:
                raise ValueError("AE_LAYERS must contain [hidden_1, hidden_2, bottleneck]")
            self.input_dim = input_dim
            self.use_attention = use_attention
            hidden_1, hidden_2, bottleneck = [int(width) for width in layers]
            self.encoder_1 = nn.Sequential(
                nn.Linear(input_dim, hidden_1), nn.ReLU(), nn.Dropout(dropout)
            )
            self.encoder_2 = nn.Sequential(
                nn.Linear(hidden_1, hidden_2), nn.ReLU(), nn.Dropout(dropout)
            )
            self.feature_attention = FeatureAttention(hidden_2) if use_attention else None
            self.to_bottleneck = nn.Sequential(
                nn.Linear(hidden_2, bottleneck), nn.ReLU(), nn.Dropout(dropout)
            )
            self.decoder_1 = nn.Sequential(
                nn.Linear(bottleneck, hidden_2), nn.ReLU(), nn.Dropout(dropout)
            )
            self.decoder_2 = nn.Sequential(
                nn.Linear(hidden_2, hidden_1), nn.ReLU(), nn.Dropout(dropout)
            )
            self.output_layer = nn.Sequential(nn.Linear(hidden_1, input_dim), nn.Sigmoid())

        def pre_attention(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.encoder_2(self.encoder_1(x))

        def gated_hidden(self, hidden: "torch.Tensor") -> "torch.Tensor":
            if self.feature_attention is None:
                return hidden
            return self.feature_attention(hidden)

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.to_bottleneck(self.gated_hidden(self.pre_attention(x)))

        def decode(self, latent: "torch.Tensor") -> "torch.Tensor":
            return self.output_layer(self.decoder_2(self.decoder_1(latent)))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.decode(self.encode(x))

        def reconstruction_error(self, x: "torch.Tensor") -> np.ndarray:
            self.eval()
            with torch.no_grad():
                return torch.mean((x - self(x)) ** 2, dim=1).detach().cpu().numpy()

        def reconstruction_residuals(self, x: "torch.Tensor") -> np.ndarray:
            self.eval()
            with torch.no_grad():
                return ((x - self(x)) ** 2).detach().cpu().numpy()

        def attention_weights(self, x: "torch.Tensor") -> np.ndarray | None:
            if self.feature_attention is None:
                return None
            self.eval()
            with torch.no_grad():
                hidden = self.pre_attention(x)
                return self.feature_attention.weights(hidden).detach().cpu().numpy()

        def reconstruction_with_hidden_mask(
            self, x: "torch.Tensor", mask: "torch.Tensor"
        ) -> "torch.Tensor":
            """Reconstruct after masking selected 64-unit attention activations."""
            hidden = self.pre_attention(x)
            gated = self.gated_hidden(hidden) * mask
            return self.decode(self.to_bottleneck(gated))


    class LSTMAutoencoder(nn.Module):
        """Sequence baseline that reconstructs the ordered feature vector."""

        def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
            self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
            self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, (hidden, _) = self.encoder(x.unsqueeze(-1))
            repeated = hidden[-1].unsqueeze(1).repeat(1, self.input_dim, 1)
            decoded, _ = self.decoder(repeated)
            return self.output_layer(decoded).squeeze(-1)

        def reconstruction_error(self, x: "torch.Tensor") -> np.ndarray:
            self.eval()
            with torch.no_grad():
                return torch.mean((x - self(x)) ** 2, dim=1).detach().cpu().numpy()


    class CNNBinaryClassifier(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.network(x.unsqueeze(1)).squeeze(1)


    class SECLBinaryClassifier(nn.Module):
        """Contrastive encoder with a supervised binary detection head."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(),
            )
            self.projection = nn.Linear(64, 32)
            self.classifier = nn.Linear(64, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.encoder(x)).squeeze(1)

        def embedding(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.nn.functional.normalize(self.projection(self.encoder(x)), dim=1)

else:  # pragma: no cover - permits import without PyTorch

    class DenoisingAutoencoder:  # type: ignore[no-redef]
        pass


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def _tensor(array: np.ndarray | pd.DataFrame, model: Any) -> Any:
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=_model_device(model))


def train_autoencoder(
    X_train_benign: np.ndarray,
    X_val_benign: np.ndarray,
    seed: int,
    use_attention: bool,
) -> tuple[DenoisingAutoencoder, float, float, float]:
    if len(X_train_benign) == 0 or len(X_val_benign) == 0:
        raise ValueError("Autoencoder training requires benign samples in train and validation")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder(
        X_train_benign.shape[1], use_attention=use_attention
    ).to(device)
    optimiser = optim.Adam(
        model.parameters(), lr=Config.AE_LEARNING_RATE, weight_decay=Config.AE_WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.as_tensor(X_train_benign, dtype=torch.float32)),
        batch_size=Config.AE_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    val_tensor = torch.as_tensor(X_val_benign, dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 0
    for _ in range(Config.AE_EPOCHS):
        model.train()
        for (batch,) in train_loader:
            clean = batch.to(device)
            noisy = torch.clamp(clean + Config.AE_NOISE_STD * torch.randn_like(clean), 0.0, 1.0)
            loss = criterion(model(noisy), clean)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_tensor), val_tensor).item())
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= Config.AE_EARLY_STOP_PATIENCE:
                break
    model.load_state_dict(best_state)
    errors = model.reconstruction_error(val_tensor)
    return model, float(np.quantile(errors, 0.95)), float(np.mean(errors)), float(np.std(errors) or 1.0)


class LightGBMWrapper:
    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = dict(params)
        self.model: Any = None
        self.calibrator = ScoreCalibrator()
        self.class_names: list[str] = []
        self.benign_index = 0
        self.decision_threshold = 0.5

    def fit(
        self,
        X_train: pd.DataFrame,
        family_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val_binary: np.ndarray,
    ) -> "LightGBMWrapper":
        family_train = np.asarray(family_train).astype(str)
        self.class_names = sorted(set(family_train), key=lambda value: (value != "benign", value))
        if len(self.class_names) < 2 or "benign" not in self.class_names:
            raise ValueError("Known-attack classifier requires benign and at least one attack class")
        class_to_index = {name: index for index, name in enumerate(self.class_names)}
        encoded_train = np.asarray([class_to_index[value] for value in family_train], dtype=int)
        self.benign_index = class_to_index["benign"]
        params = {**self.params, "objective": "multiclass", "num_class": len(self.class_names)}
        num_boost_round = int(params.pop("num_boost_round", 500))
        train = lgb.Dataset(X_train, label=encoded_train)
        self.model = lgb.train(
            params,
            train,
            num_boost_round=num_boost_round,
            callbacks=[lgb.log_evaluation(0)],
        )
        raw = self.raw_probability(X_val)
        self.calibrator.fit(raw, y_val_binary)
        return self

    def raw_class_probability(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        prediction = np.asarray(self.model.predict(X))
        if prediction.ndim != 2 or prediction.shape[1] != len(self.class_names):
            raise RuntimeError("Unexpected LightGBM multiclass probability shape")
        return prediction

    def raw_probability(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        prediction = self.raw_class_probability(X)
        return 1.0 - prediction[:, self.benign_index]

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        attack = self.calibrator.predict(self.raw_probability(X))
        return np.column_stack([1.0 - attack, attack])

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold).astype(np.int8)

    def predict_known_class(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        indices = np.argmax(self.raw_class_probability(X), axis=1)
        return np.asarray([self.class_names[int(index)] for index in indices], dtype=object)


def tune_lightgbm(
    X: pd.DataFrame,
    family: np.ndarray,
    categorical_features: Sequence[str],
    use_smote: bool,
    seed: int,
) -> dict[str, Any]:
    family = np.asarray(family).astype(str)
    class_names = sorted(set(family), key=lambda value: (value != "benign", value))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    base = {
        "objective": "multiclass",
        "num_class": len(class_names),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_child_samples": 20,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "num_threads": max(1, (os.cpu_count() or 2) - 1),
    }
    if optuna is None or Config.OPTUNA_TRIALS <= 0:
        return base
    _, counts = np.unique(family, return_counts=True)
    folds = min(Config.INNER_CV_FOLDS, int(counts.min()))
    if folds < 2:
        return base

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    def objective(trial: Any) -> float:
        params = {
            **base,
            "num_leaves": trial.suggest_int("num_leaves", 15, 96),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        }
        scores = []
        for train_idx, val_idx in splitter.split(X, family):
            fold_X = X.iloc[train_idx].reset_index(drop=True)
            fold_family = family[train_idx]
            if use_smote:
                fold_X, fold_family = smote_enn_resample(
                    fold_X, fold_family, categorical_features, seed + int(train_idx[0])
                )
            encoded_train = np.asarray([class_to_index[value] for value in fold_family], dtype=int)
            encoded_val = np.asarray([class_to_index[value] for value in family[val_idx]], dtype=int)
            train = lgb.Dataset(fold_X, label=encoded_train)
            validation = lgb.Dataset(X.iloc[val_idx], label=encoded_val, reference=train)
            model = lgb.train(
                params,
                train,
                valid_sets=[validation],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            probability = np.asarray(model.predict(X.iloc[val_idx], num_iteration=model.best_iteration))
            scores.append(
                f1_score(encoded_val, np.argmax(probability, axis=1), average="macro", zero_division=0)
            )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=Config.OPTUNA_TRIALS, show_progress_bar=False)
    return {**base, **study.best_params, "num_boost_round": 500}


class AdaptiveEnsemble:
    def __init__(
        self,
        autoencoder: DenoisingAutoencoder,
        classifier: LightGBMWrapper,
        novelty_calibrator: ScoreCalibrator,
        error_center: float,
        error_scale: float,
        novelty_threshold: float,
        decision_threshold: float = 0.5,
    ) -> None:
        self.ae = autoencoder
        self.lgb = classifier
        self.novelty_calibrator = novelty_calibrator
        self.error_center = error_center
        self.error_scale = error_scale
        self.novelty_threshold = novelty_threshold
        self.decision_threshold = decision_threshold

    def component_probabilities(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, ...]:
        array = np.asarray(X, dtype=np.float32)
        errors = self.ae.reconstruction_error(_tensor(array, self.ae))
        novelty = self.novelty_calibrator.predict(errors)
        attack = self.lgb.predict_proba(array)[:, 1]
        z = np.clip((errors - self.error_center) / (self.error_scale + 1e-12), -40, 40)
        alpha = 1.0 / (1.0 + np.exp(-z))
        combined = alpha * novelty + (1.0 - alpha) * attack
        return errors, novelty, attack, alpha, combined

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        combined = self.component_probabilities(X)[-1]
        return np.column_stack([1.0 - combined, combined])

    def predict_details(self, X: np.ndarray | pd.DataFrame) -> dict[str, np.ndarray]:
        errors, novelty, attack, alpha, combined = self.component_probabilities(X)
        malicious = combined >= self.decision_threshold
        novelty_evidence = alpha * novelty
        known_evidence = (1.0 - alpha) * attack
        unknown = malicious & (novelty >= self.novelty_threshold) & (
            novelty_evidence > known_evidence
        )
        below_scale = max(self.decision_threshold, 1e-12)
        above_scale = max(1.0 - self.decision_threshold, 1e-12)
        confidence = np.where(
            malicious,
            (combined - self.decision_threshold) / above_scale,
            (self.decision_threshold - combined) / below_scale,
        )
        confidence = np.clip(confidence, 0.0, 1.0)
        known_class = self.lgb.predict_known_class(X)
        alert_class = np.where(unknown, "unknown", np.where(malicious, known_class, "benign"))
        return {
            "malicious": malicious.astype(np.int8),
            "unknown": unknown.astype(np.int8),
            "confidence": confidence,
            "combined": combined,
            "novelty": novelty,
            "known_attack": attack,
            "novel_score": novelty_evidence,
            "alpha": alpha,
            "reconstruction_error": errors,
            "alert_class": np.asarray(alert_class, dtype=object),
        }

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.predict_details(X)["malicious"]

    def reconstruction_residuals(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.ae.reconstruction_residuals(_tensor(np.asarray(X, dtype=np.float32), self.ae))


class AEOnlyDetector:
    def __init__(
        self, ae: DenoisingAutoencoder, calibrator: ScoreCalibrator, threshold: float
    ) -> None:
        self.ae = ae
        self.calibrator = calibrator
        self.decision_threshold = threshold

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        errors = self.ae.reconstruction_error(_tensor(np.asarray(X, dtype=np.float32), self.ae))
        attack = self.calibrator.predict(errors)
        return np.column_stack([1.0 - attack, attack])

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold).astype(np.int8)


class TorchBinaryDetector:
    def __init__(self, model: Any, threshold: float = 0.5) -> None:
        self.model = model
        self.decision_threshold = threshold

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(_tensor(np.asarray(X, dtype=np.float32), self.model))
            attack = torch.sigmoid(logits).detach().cpu().numpy()
        return np.column_stack([1.0 - attack, attack])

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold).astype(np.int8)


class SklearnBinaryDetector:
    def __init__(self, model: Any, threshold: float = 0.5) -> None:
        self.model = model
        self.decision_threshold = threshold

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=float)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold).astype(np.int8)


def supervised_contrastive_loss(
    embeddings: Any, labels: Any, temperature: float = Config.SECL_TEMPERATURE
) -> Any:
    similarity = embeddings @ embeddings.T / temperature
    identity = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    logits_masked = similarity.masked_fill(identity, float("-inf"))
    log_denominator = torch.logsumexp(logits_masked, dim=1)
    log_probability = similarity - log_denominator[:, None]
    positive_counts = positives.sum(dim=1)
    valid = positive_counts > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    positive_log_probability = torch.where(
        positives, log_probability, torch.zeros_like(log_probability)
    )
    mean_positive = positive_log_probability.sum(dim=1)[valid] / positive_counts[valid]
    return -mean_positive.mean()


def train_torch_binary_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    kind: str,
) -> TorchBinaryDetector:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kind == "CNN":
        model: Any = CNNBinaryClassifier(X_train.shape[1]).to(device)
    elif kind == "SECL":
        model = SECLBinaryClassifier(X_train.shape[1]).to(device)
    else:
        raise ValueError(f"Unknown torch baseline: {kind}")
    optimiser = optim.Adam(model.parameters(), lr=Config.AE_LEARNING_RATE, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=Config.AE_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_X = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 0
    for _ in range(Config.BASELINE_EPOCHS):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X)
            loss = bce(logits, batch_y)
            if kind == "SECL":
                view_1 = torch.clamp(batch_X + 0.02 * torch.randn_like(batch_X), 0.0, 1.0)
                view_2 = torch.clamp(batch_X + 0.02 * torch.randn_like(batch_X), 0.0, 1.0)
                embeddings = torch.cat([model.embedding(view_1), model.embedding(view_2)], dim=0)
                labels = torch.cat([batch_y.long(), batch_y.long()], dim=0)
                loss = loss + Config.SECL_CONTRASTIVE_WEIGHT * supervised_contrastive_loss(
                    embeddings, labels
                )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(bce(model(val_X), val_y).item())
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= Config.BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    detector = TorchBinaryDetector(model)
    detector.decision_threshold = select_threshold(y_val, detector.predict_proba(X_val)[:, 1])
    return detector


def train_lstm_baseline(
    X_train_benign: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> AEOnlyDetector:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(X_train_benign.shape[1]).to(device)
    optimiser = optim.Adam(model.parameters(), lr=Config.AE_LEARNING_RATE, weight_decay=1e-4)
    mse = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.as_tensor(X_train_benign, dtype=torch.float32)),
        batch_size=Config.AE_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    benign_val = torch.as_tensor(X_val[y_val == 0], dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 0
    for _ in range(Config.BASELINE_EPOCHS):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            loss = mse(model(batch), batch)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(mse(model(benign_val), benign_val).item())
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= Config.BASELINE_PATIENCE:
                break
    model.load_state_dict(best_state)
    all_errors = model.reconstruction_error(_tensor(X_val, model))
    calibrator = ScoreCalibrator().fit(all_errors, y_val)
    benign_threshold = float(np.quantile(all_errors[y_val == 0], 0.95))
    detector = AEOnlyDetector(
        model, calibrator, float(calibrator.predict(np.asarray([benign_threshold]))[0])
    )
    return detector


def train_xgboost_baseline(
    X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, seed: int
) -> SklearnBinaryDetector:
    if xgb is None:
        raise RuntimeError("xgboost is required for the XGBoost baseline")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    detector = SklearnBinaryDetector(model)
    detector.decision_threshold = select_threshold(y_val, detector.predict_proba(X_val)[:, 1])
    return detector


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.05, 0.95, 91)))
    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        score = f1_score(labels, probabilities >= threshold, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def train_detector(
    data: ExperimentData,
    seed: int,
    ablation: Mapping[str, bool] | None = None,
) -> tuple[Any, list[str], float]:
    flags = {
        "use_ae": True,
        "use_lgb": True,
        "use_ensemble": True,
        "use_da": True,
        "use_smote": True,
        "use_attention": True,
    }
    if ablation:
        flags.update(ablation)
    set_seed(seed)
    selected = hybrid_feature_selection(
        data.X_train, data.y_train, min(Config.FEATURE_COUNT, data.X_train.shape[1]), seed
    )
    X_train = data.X_train[selected]
    X_val = data.X_val[selected]
    categorical = [name for name in data.categorical_features if name in selected]

    start = time.perf_counter()
    ae = None
    novelty_calibrator = None
    novelty_threshold = 0.5
    error_center = 0.0
    error_scale = 1.0
    if flags["use_ae"]:
        ae, raw_novelty_threshold, error_center, error_scale = train_autoencoder(
            X_train.loc[data.y_train == 0].to_numpy(),
            X_val.loc[data.y_val == 0].to_numpy(),
            seed,
            flags["use_attention"],
        )
        val_errors = ae.reconstruction_error(_tensor(X_val.to_numpy(dtype=np.float32), ae))
        novelty_calibrator = ScoreCalibrator().fit(val_errors, data.y_val)
        novelty_threshold = float(
            novelty_calibrator.predict(np.asarray([raw_novelty_threshold]))[0]
        )

    classifier = None
    if flags["use_lgb"]:
        params = tune_lightgbm(
            X_train, data.family_train, categorical, flags["use_smote"], seed
        )
        X_classifier = X_train.copy()
        family_classifier = data.family_train.copy()
        if flags["use_smote"]:
            X_classifier, family_classifier = smote_enn_resample(
                X_classifier, family_classifier, categorical, seed
            )
        classifier = LightGBMWrapper(params).fit(
            X_classifier, family_classifier, X_val, data.y_val
        )

    if flags["use_ensemble"] and ae is not None and classifier is not None:
        detector: Any = AdaptiveEnsemble(
            ae,
            classifier,
            novelty_calibrator,
            error_center,
            error_scale,
            novelty_threshold,
        )
        detector.decision_threshold = select_threshold(
            data.y_val, detector.predict_proba(X_val)[:, 1]
        )
    elif ae is not None:
        detector = AEOnlyDetector(ae, novelty_calibrator, novelty_threshold)
    elif classifier is not None:
        detector = classifier
        detector.decision_threshold = select_threshold(
            data.y_val, detector.predict_proba(X_val)[:, 1]
        )
    else:
        raise ValueError("A detector must include at least one of AE or LightGBM")
    return detector, selected, time.perf_counter() - start


if torch is not None:

    class GradientReversal(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: "torch.Tensor", coefficient: float) -> "torch.Tensor":
            ctx.coefficient = coefficient
            return x.view_as(x)

        @staticmethod
        def backward(ctx: Any, gradient: "torch.Tensor") -> tuple["torch.Tensor", None]:
            return -ctx.coefficient * gradient, None


def compute_mmd(source: Any, target: Any) -> Any:
    kernels = []
    for sigma in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        gamma = 1.0 / (2.0 * sigma * sigma)
        k_ss = torch.exp(-gamma * torch.cdist(source, source).pow(2))
        k_tt = torch.exp(-gamma * torch.cdist(target, target).pow(2))
        k_st = torch.exp(-gamma * torch.cdist(source, target).pow(2))
        kernels.append(k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean())
    return torch.stack(kernels).mean()


def domain_adaptation_train(
    autoencoder: DenoisingAutoencoder,
    source_reference: np.ndarray,
    source_benign: np.ndarray,
    source_val_benign: np.ndarray,
    target_unlabelled: np.ndarray,
    seed: int,
) -> DenoisingAutoencoder:
    if (
        len(source_reference) == 0
        or len(source_benign) == 0
        or len(source_val_benign) == 0
        or len(target_unlabelled) == 0
    ):
        return autoencoder
    set_seed(seed)
    device = _model_device(autoencoder)
    latent_dim = Config.AE_LAYERS[-1]
    discriminator = nn.Sequential(
        nn.Linear(latent_dim, Config.DA_DISCRIMINATOR_HIDDEN),
        nn.ReLU(),
        nn.Linear(Config.DA_DISCRIMINATOR_HIDDEN, 1),
    ).to(device)
    ae_optimiser = optim.Adam(autoencoder.parameters(), lr=Config.AE_LEARNING_RATE)
    disc_optimiser = optim.Adam(discriminator.parameters(), lr=Config.AE_LEARNING_RATE)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    source_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.as_tensor(source_reference, dtype=torch.float32)),
        batch_size=Config.AE_BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    benign_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.as_tensor(source_benign, dtype=torch.float32)),
        batch_size=Config.AE_BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    target_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.as_tensor(target_unlabelled, dtype=torch.float32)),
        batch_size=Config.AE_BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 2),
    )

    best_state = copy.deepcopy(autoencoder.state_dict())
    best_monitor = float("inf")
    patience = 0
    for _ in range(Config.DA_EPOCHS):
        target_iterator = iter(target_loader)
        benign_iterator = iter(benign_loader)
        for (source_batch,) in source_loader:
            try:
                (target_batch,) = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                (target_batch,) = next(target_iterator)
            try:
                (benign_batch,) = next(benign_iterator)
            except StopIteration:
                benign_iterator = iter(benign_loader)
                (benign_batch,) = next(benign_iterator)
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            benign_batch = benign_batch.to(device)

            # 1) Discriminator update with a detached encoder.
            autoencoder.eval()
            with torch.no_grad():
                h_source_detached = autoencoder.encode(source_batch)
                h_target_detached = autoencoder.encode(target_batch)
            source_logits = discriminator(h_source_detached)
            target_logits = discriminator(h_target_detached)
            disc_loss = bce(source_logits, torch.ones_like(source_logits)) + bce(
                target_logits, torch.zeros_like(target_logits)
            )
            disc_optimiser.zero_grad(set_to_none=True)
            disc_loss.backward()
            disc_optimiser.step()

            # 2) Encoder update. GRL reverses only the domain gradient; MMD and
            # reconstruction gradients flow normally through the encoder.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            autoencoder.train()
            h_source = autoencoder.encode(source_batch)
            h_target = autoencoder.encode(target_batch)
            reconstruction = autoencoder(benign_batch)
            reversed_source = GradientReversal.apply(h_source, Config.DA_ADV_GAMMA)
            reversed_target = GradientReversal.apply(h_target, Config.DA_ADV_GAMMA)
            domain_loss = bce(discriminator(reversed_source), torch.ones((len(source_batch), 1), device=device))
            domain_loss += bce(discriminator(reversed_target), torch.zeros((len(target_batch), 1), device=device))
            encoder_loss = (
                mse(reconstruction, benign_batch)
                + Config.DA_MMD_LAMBDA * compute_mmd(h_source, h_target)
                + domain_loss
            )
            ae_optimiser.zero_grad(set_to_none=True)
            encoder_loss.backward()
            ae_optimiser.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
        autoencoder.eval()
        discriminator.eval()
        with torch.no_grad():
            val_tensor = torch.as_tensor(source_val_benign, dtype=torch.float32, device=device)
            val_reconstruction = float(mse(autoencoder(val_tensor), val_tensor).item())
            source_probe = torch.as_tensor(
                source_reference[: min(512, len(source_reference))], dtype=torch.float32, device=device
            )
            target_probe = torch.as_tensor(
                target_unlabelled[: min(512, len(target_unlabelled))], dtype=torch.float32, device=device
            )
            source_domain = torch.sigmoid(discriminator(autoencoder.encode(source_probe))).mean()
            target_domain = torch.sigmoid(discriminator(autoencoder.encode(target_probe))).mean()
            confusion_penalty = float(
                torch.abs(source_domain - 0.5).item() + torch.abs(target_domain - 0.5).item()
            )
            monitor = val_reconstruction + 0.01 * confusion_penalty
        if monitor < best_monitor - 1e-8:
            best_monitor = monitor
            best_state = copy.deepcopy(autoencoder.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= Config.DA_EARLY_STOP_PATIENCE:
                break
    autoencoder.load_state_dict(best_state)
    autoencoder.eval()
    return autoencoder


def apply_domain_adaptation(
    detector: Any, data: ExperimentData, selected: Sequence[str], seed: int
) -> Any:
    if not hasattr(detector, "ae") or detector.ae is None or data.X_adapt.empty:
        return detector
    start = time.perf_counter()
    source_reference = data.X_train.loc[:, selected].to_numpy(dtype=np.float32)
    source_benign = data.X_train.loc[data.y_train == 0, selected].to_numpy(dtype=np.float32)
    source_val_benign = data.X_val.loc[data.y_val == 0, selected].to_numpy(dtype=np.float32)
    target = data.X_adapt[list(selected)].to_numpy(dtype=np.float32)
    detector.ae = domain_adaptation_train(
        detector.ae,
        source_reference,
        source_benign,
        source_val_benign,
        target,
        seed,
    )

    # Adaptation changes the encoder. Re-estimate calibration and thresholds
    # exclusively on the source validation partition; target labels remain unused.
    X_val = data.X_val[list(selected)]
    val_errors = detector.ae.reconstruction_error(_tensor(X_val.to_numpy(dtype=np.float32), detector.ae))
    benign_errors = val_errors[data.y_val == 0]
    refreshed = ScoreCalibrator().fit(val_errors, data.y_val)
    raw_novelty_threshold = float(np.quantile(benign_errors, 0.95))
    refreshed_novelty_threshold = float(
        refreshed.predict(np.asarray([raw_novelty_threshold]))[0]
    )
    if isinstance(detector, AdaptiveEnsemble):
        detector.novelty_calibrator = refreshed
        detector.novelty_threshold = refreshed_novelty_threshold
        detector.error_center = float(np.mean(benign_errors))
        detector.error_scale = float(np.std(benign_errors) or 1.0)
        detector.decision_threshold = select_threshold(
            data.y_val, detector.predict_proba(X_val)[:, 1]
        )
    elif isinstance(detector, AEOnlyDetector):
        detector.calibrator = refreshed
        detector.decision_threshold = refreshed_novelty_threshold
    detector.adaptation_seconds = time.perf_counter() - start
    return detector


def evaluate_detector(
    detector: Any, data: ExperimentData, selected: Sequence[str]
) -> tuple[dict[str, float], pd.DataFrame]:
    X = data.X_test[list(selected)]
    probabilities = detector.predict_proba(X)
    if hasattr(detector, "predict_details"):
        details = detector.predict_details(X)
        predictions = details["malicious"]
        unknown = details["unknown"]
        confidence = details["confidence"]
        novel_score = details["novel_score"]
        alert_class = details["alert_class"]
    else:
        predictions = detector.predict(X)
        unknown = predictions.astype(np.int8)
        novel_score = probabilities[:, 1]
        confidence = np.clip(2.0 * np.abs(probabilities[:, 1] - 0.5), 0.0, 1.0)
        if isinstance(detector, LightGBMWrapper):
            predicted_class = detector.predict_known_class(X)
            alert_class = np.where(predictions == 1, predicted_class, "benign")
        else:
            alert_class = np.where(predictions == 1, "unknown", "benign")
    metrics = compute_metrics(data.y_test, predictions, probabilities)
    novel_prediction = unknown if data.evaluation_kind == "zero_day" else predictions
    metrics.update(
        compute_zero_day_metrics(
            data.y_test, novel_prediction, data.novel_test, novel_score
        )
    )
    predictions_frame = pd.DataFrame(
        {
            "y_true": data.y_test,
            "true_family": data.family_test,
            "novel_family": data.novel_test,
            "y_pred": predictions,
            "unknown_alert": unknown,
            "alert_class": alert_class,
            "p_benign": probabilities[:, 0],
            "p_malicious": probabilities[:, 1],
            "novel_score": novel_score,
            "confidence": confidence,
        }
    )
    return metrics, predictions_frame


def save_predictions(frame: pd.DataFrame, prefix: str, seed: int, identifier: str) -> Path:
    path = Config.PREDICTIONS_DIR / f"{prefix}_seed{seed}_{_normalise_name(identifier)}.csv"
    frame.to_csv(path, index=False)
    return path


def save_model_artifacts(
    detector: Any,
    data: ExperimentData,
    selected: Sequence[str],
    prefix: str,
    seed: int,
    identifier: str,
) -> dict[str, str]:
    """Persist executable model components plus source-fitted preprocessing metadata."""
    stem = f"{prefix}_seed{seed}_{_normalise_name(identifier)}"
    directory = Config.MODEL_DIR / stem
    directory.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if hasattr(detector, "ae") and detector.ae is not None:
        ae_path = directory / "autoencoder_state.pt"
        torch.save(detector.ae.state_dict(), ae_path)
        artifacts["autoencoder"] = str(ae_path)
    classifier = detector.lgb if hasattr(detector, "lgb") else detector
    if isinstance(classifier, LightGBMWrapper) and classifier.model is not None:
        lgb_path = directory / "lightgbm_model.txt"
        classifier.model.save_model(str(lgb_path))
        artifacts["lightgbm"] = str(lgb_path)
    elif isinstance(detector, TorchBinaryDetector):
        baseline_path = directory / "torch_baseline_state.pt"
        torch.save(detector.model.state_dict(), baseline_path)
        artifacts["torch_baseline"] = str(baseline_path)
    elif isinstance(detector, SklearnBinaryDetector):
        baseline_path = directory / "sklearn_baseline.pkl"
        with baseline_path.open("wb") as handle:
            pickle.dump(detector.model, handle)
        artifacts["sklearn_baseline"] = str(baseline_path)
    pipeline_path = directory / "pipeline.pkl"
    with pipeline_path.open("wb") as handle:
        pickle.dump(
            {
                "preprocessor": data.preprocessor,
                "selected_features": list(selected),
                "novelty_calibrator": getattr(detector, "novelty_calibrator", None),
                "classifier_calibrator": getattr(classifier, "calibrator", None),
            },
            handle,
        )
    artifacts["pipeline"] = str(pipeline_path)
    metadata = {
        "seed": seed,
        "identifier": identifier,
        "evaluation_kind": data.evaluation_kind,
        "selected_features": list(selected),
        "decision_threshold": getattr(detector, "decision_threshold", 0.5),
        "novelty_threshold": getattr(detector, "novelty_threshold", None),
        "error_center": getattr(detector, "error_center", None),
        "error_scale": getattr(detector, "error_scale", None),
        "known_classes": getattr(classifier, "class_names", None),
        "lightgbm_parameters": getattr(classifier, "params", None),
        "model_type": type(detector).__name__,
    }
    metadata_path = directory / "metadata.json"
    save_json(metadata, metadata_path)
    artifacts["metadata"] = str(metadata_path)
    return artifacts


if nn is not None:

    class ReconstructionScoreModel(nn.Module):
        """Differentiable scalar anomaly score used by GradientSHAP."""

        def __init__(self, autoencoder: DenoisingAutoencoder) -> None:
            super().__init__()
            self.autoencoder = autoencoder

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            reconstruction = self.autoencoder(x)
            return torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)


class ExplanationModule:
    """Evaluate decision-specific explanations using masking and perturbation tests."""

    def __init__(
        self, detector: AdaptiveEnsemble, feature_names: Sequence[str], background: np.ndarray
    ) -> None:
        if shap is None or lime is None:
            raise RuntimeError("SHAP and LIME are required for explanation evaluation")
        self.detector = detector
        self.feature_names = list(feature_names)
        background = np.asarray(background, dtype=float)
        if len(background) == 0:
            raise ValueError("Explanation background cannot be empty")
        self.background = background[: min(200, len(background))]
        self.reference = np.median(self.background, axis=0)
        self.shap_explainer = shap.TreeExplainer(detector.lgb.model)
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.background,
            feature_names=self.feature_names,
            class_names=["benign", "malicious"],
            mode="classification",
            discretize_continuous=True,
            random_state=0,
        )
        score_model = ReconstructionScoreModel(detector.ae).to(_model_device(detector.ae))
        score_model.eval()
        background_tensor = torch.as_tensor(
            self.background[: min(100, len(self.background))],
            dtype=torch.float32,
            device=_model_device(detector.ae),
        )
        self.gradient_explainer = shap.GradientExplainer(score_model, background_tensor)

    def _predicted_class_tree_shap(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self.shap_explainer.shap_values(rows)
        classes = len(self.detector.lgb.class_names)
        class_indices = np.argmax(self.detector.lgb.raw_class_probability(rows), axis=1)
        if isinstance(values, list):
            arrays = [np.asarray(value) for value in values]
            selected = np.stack(
                [arrays[int(class_index)][row_index] for row_index, class_index in enumerate(class_indices)]
            )
            return selected, class_indices
        array = np.asarray(values)
        if array.ndim == 2:
            return array, class_indices
        if array.ndim == 3 and array.shape[-1] == classes:
            selected = np.stack(
                [array[row_index, :, int(class_index)] for row_index, class_index in enumerate(class_indices)]
            )
            return selected, class_indices
        if array.ndim == 3 and array.shape[0] == classes:
            selected = np.stack(
                [array[int(class_index), row_index, :] for row_index, class_index in enumerate(class_indices)]
            )
            return selected, class_indices
        raise RuntimeError(f"Unsupported TreeSHAP output shape: {array.shape}")

    @staticmethod
    def _gradient_values(values: Any) -> np.ndarray:
        if isinstance(values, list):
            values = values[0]
        array = np.asarray(values)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[:, :, 0]
        return array

    @staticmethod
    def _weighted_correlation(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        weights = 1.0 / np.arange(1, len(x) + 1, dtype=float)
        weights /= weights.sum()
        x_centered = x - np.sum(weights * x)
        y_centered = y - np.sum(weights * y)
        denominator = math.sqrt(
            np.sum(weights * x_centered ** 2) * np.sum(weights * y_centered ** 2)
        )
        if denominator <= 1e-12:
            return 0.0
        return float(np.clip(np.sum(weights * x_centered * y_centered) / denominator, 0.0, 1.0))

    def _masking_fidelity(
        self,
        rows: np.ndarray,
        attributions: np.ndarray,
        score_function: Any,
        contexts: Sequence[Any] | None = None,
    ) -> np.ndarray:
        fidelities = []
        top_k = min(Config.XAI_TOP_K, rows.shape[1])
        for row_index, (row, attribution) in enumerate(zip(rows, attributions)):
            order = np.argsort(np.abs(attribution))[::-1][:top_k]
            context = contexts[row_index] if contexts is not None else None
            score = lambda values: (
                score_function(values, context) if contexts is not None else score_function(values)
            )
            original = float(np.asarray(score(row[None, :])).reshape(-1)[0])
            predicted_change = []
            observed_change = []
            masked = row.copy()
            cumulative = 0.0
            for feature_index in order:
                cumulative += float(attribution[feature_index])
                masked[feature_index] = self.reference[feature_index]
                changed = float(np.asarray(score(masked[None, :])).reshape(-1)[0])
                predicted_change.append(cumulative)
                observed_change.append(original - changed)
            fidelities.append(self._weighted_correlation(predicted_change, observed_change))
        return np.asarray(fidelities, dtype=float)

    @staticmethod
    def _top_k_overlap(reference: np.ndarray, repeats: Sequence[np.ndarray], k: int) -> float:
        reference_rank = np.argsort(np.abs(reference))[::-1][:k]
        reference_set = set(reference_rank.tolist())
        scores = []
        for repeat in repeats:
            repeat_set = set(np.argsort(np.abs(repeat))[::-1][:k].tolist())
            scores.append(len(reference_set & repeat_set) / max(1, k))
        return float(np.mean(scores)) if scores else np.nan

    def _lime_vectors(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors = []
        surrogate_scores = []
        for row in rows:
            explanation = self.lime_explainer.explain_instance(
                row,
                self.detector.predict_proba,
                num_features=len(self.feature_names),
                num_samples=Config.LIME_NUM_SAMPLES,
            )
            vector = np.zeros(len(self.feature_names), dtype=float)
            for feature_index, weight in explanation.local_exp.get(1, []):
                vector[int(feature_index)] = float(weight)
            vectors.append(vector)
            surrogate_scores.append(float(explanation.score))
        return np.asarray(vectors), np.asarray(surrogate_scores)

    def _stratified_indices(
        self, y_true: np.ndarray, novel_mask: np.ndarray, seed: int, n: int
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        y_true = np.asarray(y_true)
        novel = np.asarray(novel_mask).astype(bool)
        strata = [np.flatnonzero(y_true == 0), np.flatnonzero((y_true == 1) & ~novel), np.flatnonzero(novel)]
        nonempty = [indices for indices in strata if len(indices)]
        selected: list[int] = []
        remaining = n
        for position, indices in enumerate(nonempty):
            allocation = min(len(indices), max(1, remaining // (len(nonempty) - position)))
            selected.extend(rng.choice(indices, size=allocation, replace=False).tolist())
            remaining = n - len(selected)
        if remaining > 0:
            pool = np.setdiff1d(np.arange(len(y_true)), np.asarray(selected, dtype=int))
            selected.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
        rng.shuffle(selected)
        return np.asarray(selected[:n], dtype=int)

    def _attention_fidelity(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = self.detector.ae
        tensor = _tensor(rows, model)
        model.eval()
        with torch.no_grad():
            weights = model.attention_weights(tensor)
            original_error = torch.mean((tensor - model(tensor)) ** 2, dim=1).cpu().numpy()
        if weights is None:
            return np.full(len(rows), np.nan), np.empty((len(rows), 0))
        fidelities = []
        top_k = min(Config.XAI_TOP_K, weights.shape[1])
        for row_index, attention_row in enumerate(weights):
            order = np.argsort(attention_row)[::-1][:top_k]
            predicted_change = []
            observed_change = []
            mask = torch.ones((1, weights.shape[1]), dtype=torch.float32, device=_model_device(model))
            cumulative = 0.0
            for unit_index in order:
                cumulative += float(attention_row[unit_index])
                mask[0, unit_index] = 0.0
                with torch.no_grad():
                    reconstruction = model.reconstruction_with_hidden_mask(tensor[row_index:row_index + 1], mask)
                    changed_error = float(
                        torch.mean((tensor[row_index:row_index + 1] - reconstruction) ** 2).item()
                    )
                predicted_change.append(cumulative)
                observed_change.append(changed_error - float(original_error[row_index]))
            fidelities.append(self._weighted_correlation(predicted_change, observed_change))
        return np.asarray(fidelities), weights

    def evaluate(
        self,
        X: np.ndarray,
        seed: int,
        y_true: np.ndarray | None = None,
        novel_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        X = np.asarray(X, dtype=float)
        n = min(Config.XAI_AUDIT_MAX, len(X))
        if y_true is not None and novel_mask is not None:
            indices = self._stratified_indices(y_true, novel_mask, seed, n)
        else:
            indices = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        audit = X[indices]
        rng = np.random.default_rng(seed + 901)
        audit_decisions = self.detector.predict_details(audit)
        known_eligible = (audit_decisions["malicious"] == 1) & (
            audit_decisions["unknown"] == 0
        )
        novelty_eligible = audit_decisions["unknown"] == 1

        shap_start = time.perf_counter()
        shap_values, shap_classes = self._predicted_class_tree_shap(audit)
        shap_ms = 1000.0 * (time.perf_counter() - shap_start) / max(1, n)
        shap_fidelity = self._masking_fidelity(
            audit,
            shap_values,
            lambda rows, class_index: self.detector.lgb.raw_class_probability(rows)[:, int(class_index)],
            shap_classes,
        )

        lime_start = time.perf_counter()
        lime_values, lime_surrogate = self._lime_vectors(audit)
        lime_ms = 1000.0 * (time.perf_counter() - lime_start) / max(1, n)
        lime_fidelity = self._masking_fidelity(
            audit, lime_values, lambda rows: self.detector.predict_proba(rows)[:, 1]
        )

        residual_start = time.perf_counter()
        residual_values = self.detector.reconstruction_residuals(audit)
        residual_ms = 1000.0 * (time.perf_counter() - residual_start) / max(1, n)
        reconstruction_score = lambda rows: self.detector.ae.reconstruction_error(
            _tensor(np.asarray(rows, dtype=np.float32), self.detector.ae)
        )
        residual_fidelity = self._masking_fidelity(audit, residual_values, reconstruction_score)

        gradient_start = time.perf_counter()
        gradient_values = self._gradient_values(
            self.gradient_explainer.shap_values(_tensor(audit, self.detector.ae))
        )
        gradient_ms = 1000.0 * (time.perf_counter() - gradient_start) / max(1, n)
        gradient_fidelity = self._masking_fidelity(audit, gradient_values, reconstruction_score)

        attention_start = time.perf_counter()
        attention_fidelity, attention_values = self._attention_fidelity(audit)
        attention_ms = 1000.0 * (time.perf_counter() - attention_start) / max(1, n)

        stability: dict[str, list[float]] = {
            "SHAP": [], "LIME": [], "Residual": [], "GradientSHAP": [], "Attention": []
        }
        perturbation_scale = np.maximum(np.std(self.background, axis=0), 1e-6) * 0.01
        repeated_rows = [
            np.clip(audit + rng.normal(0.0, perturbation_scale, size=audit.shape), 0.0, 1.0)
            for _ in range(Config.XAI_STABILITY_REPEATS)
        ]
        repeated_shap = [self._predicted_class_tree_shap(rows)[0] for rows in repeated_rows]
        repeated_lime = [self._lime_vectors(rows)[0] for rows in repeated_rows]
        repeated_residual = [self.detector.reconstruction_residuals(rows) for rows in repeated_rows]
        repeated_gradient = [
            self._gradient_values(self.gradient_explainer.shap_values(_tensor(rows, self.detector.ae)))
            for rows in repeated_rows
        ]
        repeated_attention = [
            self.detector.ae.attention_weights(_tensor(rows, self.detector.ae))
            for rows in repeated_rows
        ]
        for row_index in range(n):
            k_feature = min(Config.XAI_TOP_K, audit.shape[1])
            stability["SHAP"].append(self._top_k_overlap(shap_values[row_index], [v[row_index] for v in repeated_shap], k_feature))
            stability["LIME"].append(self._top_k_overlap(lime_values[row_index], [v[row_index] for v in repeated_lime], k_feature))
            stability["Residual"].append(self._top_k_overlap(residual_values[row_index], [v[row_index] for v in repeated_residual], k_feature))
            stability["GradientSHAP"].append(self._top_k_overlap(gradient_values[row_index], [v[row_index] for v in repeated_gradient], k_feature))
            if attention_values.shape[1] and all(value is not None for value in repeated_attention):
                k_attention = min(Config.XAI_TOP_K, attention_values.shape[1])
                stability["Attention"].append(
                    self._top_k_overlap(
                        attention_values[row_index],
                        [value[row_index] for value in repeated_attention if value is not None],
                        k_attention,
                    )
                )

        def summarise(fidelity: np.ndarray, consistency: Sequence[float], elapsed: float) -> dict[str, float]:
            valid = np.isfinite(fidelity)
            return {
                "local_fidelity": float(np.mean(fidelity[valid])) if valid.any() else np.nan,
                "rank_consistency": float(np.nanmean(consistency)) if len(consistency) else np.nan,
                "coverage": float(np.mean(fidelity[valid] >= Config.XAI_FIDELITY_THRESHOLD)) if valid.any() else 0.0,
                "time_ms": float(elapsed),
            }

        shap_mask = known_eligible
        novelty_mask = novelty_eligible
        output = {
            "SHAP": summarise(
                shap_fidelity[shap_mask], np.asarray(stability["SHAP"])[shap_mask], shap_ms
            ),
            "LIME": summarise(lime_fidelity, stability["LIME"], lime_ms),
            "ReconstructionResidual": summarise(
                residual_fidelity[novelty_mask],
                np.asarray(stability["Residual"])[novelty_mask],
                residual_ms,
            ),
            "GradientSHAP": summarise(
                gradient_fidelity[novelty_mask],
                np.asarray(stability["GradientSHAP"])[novelty_mask],
                gradient_ms,
            ),
            "Attention": summarise(attention_fidelity, stability["Attention"], attention_ms),
            "audit": {
                "indices": indices,
                "lime_surrogate_r2_mean": float(np.mean(lime_surrogate)),
                "fidelity_threshold": Config.XAI_FIDELITY_THRESHOLD,
                "stability_repeats": Config.XAI_STABILITY_REPEATS,
                "known_alerts": int(np.sum(known_eligible)),
                "novelty_alerts": int(np.sum(novelty_eligible)),
            },
        }
        return output


def measure_computational_performance(
    detector: Any, X_test: pd.DataFrame, selected: Sequence[str], training_seconds: float
) -> dict[str, Any]:
    X = X_test[list(selected)]
    n = min(100, len(X))
    if n == 0:
        return {}
    detector.predict_proba(X.iloc[: min(10, n)])
    latencies = []
    for index in range(n):
        start = time.perf_counter()
        detector.predict_proba(X.iloc[index:index + 1])
        latencies.append(1000.0 * (time.perf_counter() - start))
    memory_mb = np.nan
    if psutil is not None:
        try:
            memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        except Exception as exc:  # resource accounting must never invalidate a run
            warnings.warn(f"Process memory measurement unavailable: {exc}", RuntimeWarning)
    if not np.isfinite(memory_mb) and resource is not None:
        try:
            # Linux reports ru_maxrss in KiB; macOS reports bytes.
            peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            memory_mb = peak_rss / (1024 ** 2 if sys.platform == "darwin" else 1024)
        except Exception as exc:  # keep resource metrics optional
            warnings.warn(f"Peak-memory fallback unavailable: {exc}", RuntimeWarning)
    buffer = io.BytesIO()
    try:
        if hasattr(detector, "ae") and detector.ae is not None:
            torch.save(detector.ae.state_dict(), buffer)
        if hasattr(detector, "lgb") and detector.lgb is not None:
            pickle.dump(detector.lgb.model, buffer)
        elif isinstance(detector, LightGBMWrapper):
            pickle.dump(detector.model, buffer)
        elif isinstance(detector, TorchBinaryDetector):
            torch.save(detector.model.state_dict(), buffer)
        elif isinstance(detector, SklearnBinaryDetector):
            pickle.dump(detector.model, buffer)
        model_size = buffer.tell() / (1024 ** 2)
    finally:
        buffer.close()
    total_training_seconds = training_seconds + float(getattr(detector, "adaptation_seconds", 0.0))
    return {
        "training_hours": total_training_seconds / 3600.0,
        "inference_ms_mean": float(np.mean(latencies)),
        "inference_ms_std": float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
        "memory_mb": float(memory_mb),
        "model_size_mb": float(model_size),
        "device": (
            str(_model_device(detector.ae))
            if hasattr(detector, "ae") and detector.ae is not None
            else str(_model_device(detector.model))
            if isinstance(detector, TorchBinaryDetector)
            else "cpu"
        ),
        "batch_size": 1,
        "samples_timed": n,
    }


def aggregate_metric_records(records: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    keys = sorted(set().union(*(record.keys() for record in records))) if records else []
    summary: dict[str, Any] = {}
    for key in keys:
        converted = []
        for record in records:
            try:
                converted.append(float(record.get(key, np.nan)))
            except (TypeError, ValueError):
                continue
        values = np.asarray(converted, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
    return summary


def train_protocol_baseline(
    data: ExperimentData, name: str, seed: int
) -> tuple[Any, list[str], float]:
    if name == "Autoencoder":
        return train_detector(
            data,
            seed,
            {
                "use_ae": True, "use_lgb": False, "use_ensemble": False,
                "use_da": False, "use_smote": False, "use_attention": False,
            },
        )
    if name == "LightGBM":
        return train_detector(
            data,
            seed,
            {
                "use_ae": False, "use_lgb": True, "use_ensemble": False,
                "use_da": False, "use_smote": True, "use_attention": False,
            },
        )
    selected = hybrid_feature_selection(
        data.X_train, data.y_train, min(Config.FEATURE_COUNT, data.X_train.shape[1]), seed
    )
    X_train = data.X_train[selected]
    X_val = data.X_val[selected]
    categorical = [feature for feature in data.categorical_features if feature in selected]
    start = time.perf_counter()
    if name == "LSTM autoencoder":
        detector = train_lstm_baseline(
            X_train.loc[data.y_train == 0].to_numpy(dtype=np.float32),
            X_val.to_numpy(dtype=np.float32),
            data.y_val,
            seed,
        )
    elif name in {"CNN", "SECL"}:
        resampled_X, resampled_y = smote_enn_resample(
            X_train, data.y_train, categorical, seed
        )
        detector = train_torch_binary_baseline(
            resampled_X.to_numpy(dtype=np.float32),
            resampled_y.astype(np.int8),
            X_val.to_numpy(dtype=np.float32),
            data.y_val,
            seed,
            name,
        )
    elif name == "XGBoost":
        resampled_X, resampled_y = smote_enn_resample(
            X_train, data.y_train, categorical, seed
        )
        detector = train_xgboost_baseline(
            resampled_X, resampled_y.astype(np.int8), X_val, data.y_val, seed
        )
    else:
        raise ValueError(f"Unknown protocol baseline: {name}")
    return detector, selected, time.perf_counter() - start


def run_protocol_baselines() -> dict[str, Any]:
    """Run every baseline reported in the performance and efficiency tables."""
    baseline_names = ["SECL", "Autoencoder", "LSTM autoencoder", "CNN", "LightGBM", "XGBoost"]
    frame = load_dataset("CICIoT2023")
    families = get_available_families(frame)
    output: dict[str, Any] = {}
    for name in baseline_names:
        runs = []
        seed_means = []
        for seed in Config.RANDOM_SEEDS:
            family_metrics = []
            for family in families:
                print(f"Baseline={name} seed={seed} held-out family={family}")
                data = prepare_zero_day_data(frame, family, seed)
                detector, selected, training_seconds = train_protocol_baseline(data, name, seed)
                metrics, predictions = evaluate_detector(detector, data, selected)
                save_predictions(predictions, f"baseline_{name}", seed, family)
                efficiency = measure_computational_performance(
                    detector, data.X_test, selected, training_seconds
                )
                model_artifacts = save_model_artifacts(
                    detector, data, selected, f"baseline_{name}", seed, family
                )
                runs.append(
                    {
                        "seed": seed,
                        "family": family,
                        "selected_features": selected,
                        "data_summary": experiment_data_summary(data),
                        "metrics": metrics,
                        "efficiency": efficiency,
                        "model_artifacts": model_artifacts,
                    }
                )
                family_metrics.append(metrics)
            seed_means.append(
                {
                    key: float(np.mean([record[key] for record in family_metrics]))
                    for key in family_metrics[0]
                }
            )
        output[name] = {
            "summary_across_seed_means": aggregate_metric_records(seed_means),
            "efficiency_summary": aggregate_metric_records([run["efficiency"] for run in runs]),
            "runs": runs,
        }
    return output


def run_zero_day_experiment() -> dict[str, Any]:
    frame = load_dataset("CICIoT2023")
    families = get_available_families(frame)
    runs: list[dict[str, Any]] = []
    xai_runs: list[dict[str, Any]] = []
    for seed in Config.RANDOM_SEEDS:
        for family in families:
            print(f"Zero-day seed={seed} held-out family={family}")
            data = prepare_zero_day_data(frame, family, seed)
            detector, selected, training_seconds = train_detector(data, seed)
            detector = apply_domain_adaptation(detector, data, selected, seed)
            metrics, predictions = evaluate_detector(detector, data, selected)
            save_predictions(predictions, "zero_day", seed, family)
            model_artifacts = save_model_artifacts(
                detector, data, selected, "zero_day", seed, family
            )
            efficiency = measure_computational_performance(
                detector, data.X_test, selected, training_seconds
            )
            runs.append(
                {
                    "seed": seed,
                    "family": family,
                    "selected_features": selected,
                    "data_summary": experiment_data_summary(data),
                    "metrics": metrics,
                    "efficiency": efficiency,
                    "model_artifacts": model_artifacts,
                }
            )
            if Config.RUN_XAI and isinstance(detector, AdaptiveEnsemble):
                module = ExplanationModule(
                    detector,
                    selected,
                    data.X_train.loc[data.y_train == 0, selected].to_numpy(),
                )
                xai_runs.append(
                    {
                        "seed": seed,
                        "family": family,
                        "methods": module.evaluate(
                            data.X_test[selected].to_numpy(),
                            seed,
                            data.y_test,
                            data.novel_test,
                        ),
                    }
                )
    seed_summaries = []
    for seed in Config.RANDOM_SEEDS:
        seed_records = [run["metrics"] for run in runs if run["seed"] == seed]
        seed_summaries.append(
            {key: np.mean([record[key] for record in seed_records]) for key in seed_records[0]}
        )
    xai_summary: dict[str, Any] = {}
    if xai_runs:
        method_names = sorted(
            set().union(
                *(set(run["methods"]) - {"audit"} for run in xai_runs)
            )
        )
        for method in method_names:
            metric_records = [
                run["methods"][method] for run in xai_runs if method in run["methods"]
            ]
            xai_summary[method] = aggregate_metric_records(metric_records)
    return {
        "overall_across_seed_means": aggregate_metric_records(seed_summaries),
        "per_family": {
            family: aggregate_metric_records([run["metrics"] for run in runs if run["family"] == family])
            for family in families
        },
        "runs": runs,
        "xai_runs": xai_runs,
        "xai_summary": xai_summary,
        "efficiency_summary": aggregate_metric_records([run["efficiency"] for run in runs]),
    }


def run_cross_dataset_all() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for config_name, specification in Config.CROSS_DATASET_CONFIGS.items():
        runs = []
        for seed in Config.RANDOM_SEEDS:
            print(f"Cross-dataset config={config_name} seed={seed}")
            data = prepare_cross_dataset_data(specification["source"], specification["target"], seed)
            detector, selected, training_seconds = train_detector(data, seed)
            no_da_metrics, no_da_predictions = evaluate_detector(detector, data, selected)
            save_predictions(no_da_predictions, f"cross_{config_name}_no_da", seed, specification["target"])
            no_da_artifacts = save_model_artifacts(
                detector,
                data,
                selected,
                f"cross_{config_name}_no_da",
                seed,
                specification["target"],
            )
            detector = apply_domain_adaptation(detector, data, selected, seed)
            da_metrics, da_predictions = evaluate_detector(detector, data, selected)
            save_predictions(da_predictions, f"cross_{config_name}_da", seed, specification["target"])
            da_artifacts = save_model_artifacts(
                detector,
                data,
                selected,
                f"cross_{config_name}_da",
                seed,
                specification["target"],
            )
            runs.append(
                {
                    "seed": seed,
                    "selected_features": selected,
                    "data_summary": experiment_data_summary(data),
                    "no_da": no_da_metrics,
                    "with_da": da_metrics,
                    "no_da_model_artifacts": no_da_artifacts,
                    "with_da_model_artifacts": da_artifacts,
                    "da_gain_accuracy_pp": 100.0 * (da_metrics["accuracy"] - no_da_metrics["accuracy"]),
                    "efficiency": measure_computational_performance(
                        detector, data.X_test, selected, training_seconds
                    ),
                }
            )
        output[config_name] = {
            "source": specification["source"],
            "target": specification["target"],
            "no_da": aggregate_metric_records([run["no_da"] for run in runs]),
            "with_da": aggregate_metric_records([run["with_da"] for run in runs]),
            "da_gain_accuracy_pp": aggregate_metric_records(
                [{"gain": run["da_gain_accuracy_pp"]} for run in runs]
            ).get("gain", {}),
            "efficiency_summary": aggregate_metric_records(
                [run["efficiency"] for run in runs]
            ),
            "runs": runs,
        }
    return output


def run_ablation_study() -> dict[str, Any]:
    ablations = {
        "A_AE_only": {
            "use_ae": True, "use_lgb": False, "use_ensemble": False,
            "use_da": False, "use_smote": False, "use_attention": False,
        },
        "B_LightGBM_only": {
            "use_ae": False, "use_lgb": True, "use_ensemble": False,
            "use_da": False, "use_smote": True, "use_attention": False,
        },
        "C_Ensemble_no_DA": {"use_da": False},
        "D_No_SMOTE_ENN": {"use_smote": False},
        "E_No_attention": {"use_attention": False},
        "F_Proposed_full": {},
    }
    frame = load_dataset("CICIoT2023")
    families = get_available_families(frame)
    output: dict[str, Any] = {}
    for name, flags in ablations.items():
        seed_means = []
        runs = []
        for seed in Config.RANDOM_SEEDS:
            family_metrics = []
            for family in families:
                print(f"Ablation={name} seed={seed} family={family}")
                # The adaptation/test division is fixed across every ablation;
                # no-DA configurations simply do not consume X_adapt.
                data = prepare_zero_day_data(frame, family, seed)
                detector, selected, training_seconds = train_detector(data, seed, flags)
                if flags.get("use_da", True):
                    detector = apply_domain_adaptation(detector, data, selected, seed)
                metrics, predictions = evaluate_detector(detector, data, selected)
                save_predictions(predictions, f"ablation_{name}", seed, family)
                family_metrics.append(metrics)
                runs.append(
                    {
                        "seed": seed, "family": family, "metrics": metrics,
                        "selected_features": selected,
                        "data_summary": experiment_data_summary(data),
                        "efficiency": measure_computational_performance(
                            detector, data.X_test, selected, training_seconds
                        ),
                    }
                )
            seed_means.append(
                {key: np.mean([metric[key] for metric in family_metrics]) for key in family_metrics[0]}
            )
        output[name] = {
            "summary_across_seed_means": aggregate_metric_records(seed_means),
            "efficiency_summary": aggregate_metric_records(
                [run["efficiency"] for run in runs]
            ),
            "runs": runs,
        }
    return output


def run_iot23_stress_test() -> dict[str, Any]:
    data = prepare_cross_dataset_data(["CICIoT2023"], "IoT-23", Config.RANDOM_SEEDS[0])
    detector, selected, training_seconds = train_detector(data, Config.RANDOM_SEEDS[0])
    detector = apply_domain_adaptation(detector, data, selected, Config.RANDOM_SEEDS[0])
    metrics, predictions = evaluate_detector(detector, data, selected)
    save_predictions(predictions, "stress_iot23", Config.RANDOM_SEEDS[0], "IoT23")
    artifacts = save_model_artifacts(
        detector, data, selected, "stress_iot23", Config.RANDOM_SEEDS[0], "IoT23"
    )
    return {
        "metrics": metrics,
        "selected_features": selected,
        "data_summary": experiment_data_summary(data),
        "model_artifacts": artifacts,
        "efficiency": measure_computational_performance(
            detector, data.X_test, selected, training_seconds
        ),
    }


def export_summary_tables(results: Mapping[str, Any]) -> None:
    zero_day_comparison: list[dict[str, Any]] = []
    if "zero_day" in results:
        overall_row: dict[str, Any] = {"model": "Proposed framework"}
        for metric, estimate in results["zero_day"]["overall_across_seed_means"].items():
            overall_row[f"{metric}_mean"] = estimate["mean"]
            overall_row[f"{metric}_sd"] = estimate["std"]
        pd.DataFrame([overall_row]).to_csv(
            Config.TABLE_DIR / "zero_day_overall.csv", index=False
        )
        zero_day_comparison.append(overall_row)
        rows = []
        for family, metrics in results["zero_day"]["per_family"].items():
            row = {"family": family}
            for metric, estimate in metrics.items():
                row[f"{metric}_mean"] = estimate["mean"]
                row[f"{metric}_sd"] = estimate["std"]
            rows.append(row)
        pd.DataFrame(rows).to_csv(Config.TABLE_DIR / "zero_day_per_family.csv", index=False)
        if results["zero_day"].get("xai_summary"):
            xai_rows = []
            for method, metrics in results["zero_day"]["xai_summary"].items():
                row = {"method": method}
                for metric, estimate in metrics.items():
                    row[f"{metric}_mean"] = estimate["mean"]
                    row[f"{metric}_sd"] = estimate["std"]
                xai_rows.append(row)
            pd.DataFrame(xai_rows).to_csv(
                Config.TABLE_DIR / "explanation_quality.csv", index=False
            )
    if "cross_dataset" in results:
        rows = []
        for config_name, entry in results["cross_dataset"].items():
            row = {"configuration": config_name, "target": entry["target"]}
            for condition in ["no_da", "with_da"]:
                for metric, estimate in entry[condition].items():
                    row[f"{condition}_{metric}_mean"] = estimate["mean"]
                    row[f"{condition}_{metric}_sd"] = estimate["std"]
            row["da_gain_accuracy_pp"] = entry["da_gain_accuracy_pp"].get("mean", np.nan)
            rows.append(row)
        pd.DataFrame(rows).to_csv(Config.TABLE_DIR / "cross_dataset_summary.csv", index=False)
    if "ablation" in results:
        rows = []
        for name, entry in results["ablation"].items():
            row = {"ablation": name}
            for metric, estimate in entry["summary_across_seed_means"].items():
                row[f"{metric}_mean"] = estimate["mean"]
                row[f"{metric}_sd"] = estimate["std"]
            rows.append(row)
        pd.DataFrame(rows).to_csv(Config.TABLE_DIR / "ablation_summary.csv", index=False)
        efficiency_rows = []
        for name, entry in results["ablation"].items():
            row = {"model": name}
            for metric, estimate in entry.get("efficiency_summary", {}).items():
                row[f"{metric}_mean"] = estimate["mean"]
                row[f"{metric}_sd"] = estimate["std"]
            efficiency_rows.append(row)
        pd.DataFrame(efficiency_rows).to_csv(
            Config.TABLE_DIR / "computational_performance.csv", index=False
        )
    if "baselines" in results:
        baseline_rows = []
        efficiency_rows = []
        for name, entry in results["baselines"].items():
            row = {"model": name}
            for metric, estimate in entry["summary_across_seed_means"].items():
                row[f"{metric}_mean"] = estimate["mean"]
                row[f"{metric}_sd"] = estimate["std"]
            baseline_rows.append(row)
            zero_day_comparison.append(row)
            efficiency = {"model": name}
            for metric, estimate in entry.get("efficiency_summary", {}).items():
                efficiency[f"{metric}_mean"] = estimate["mean"]
                efficiency[f"{metric}_sd"] = estimate["std"]
            efficiency_rows.append(efficiency)
        if "zero_day" in results:
            proposed = {"model": "Proposed framework"}
            for metric, estimate in results["zero_day"].get("efficiency_summary", {}).items():
                proposed[f"{metric}_mean"] = estimate["mean"]
                proposed[f"{metric}_sd"] = estimate["std"]
            efficiency_rows.append(proposed)
        pd.DataFrame(baseline_rows).to_csv(
            Config.TABLE_DIR / "protocol_matched_baselines.csv", index=False
        )
        pd.DataFrame(efficiency_rows).to_csv(
            Config.TABLE_DIR / "computational_performance.csv", index=False
        )
    if zero_day_comparison:
        manuscript_rows = []
        for row in zero_day_comparison:
            manuscript_rows.append(
                {
                    "model": row["model"],
                    "nadr_mean": row.get("nadr_mean", np.nan),
                    "nadr_sd": row.get("nadr_sd", np.nan),
                    "nafpr_mean": row.get("nafpr_mean", np.nan),
                    "nafpr_sd": row.get("nafpr_sd", np.nan),
                    "zd_f1_mean": row.get("zd_f1_mean", np.nan),
                    "zd_f1_sd": row.get("zd_f1_sd", np.nan),
                    "roc_auc_mean": row.get("zd_auc_roc_mean", np.nan),
                    "roc_auc_sd": row.get("zd_auc_roc_sd", np.nan),
                    "pr_auc_mean": row.get("zd_auc_pr_mean", np.nan),
                    "pr_auc_sd": row.get("zd_auc_pr_sd", np.nan),
                }
            )
        pd.DataFrame(manuscript_rows).to_csv(
            Config.TABLE_DIR / "zero_day_model_comparison.csv", index=False
        )


def generate_figures(results: Mapping[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(f"Figures skipped because matplotlib is unavailable: {exc}")
        return
    if "zero_day" in results:
        summary = results["zero_day"]["overall_across_seed_means"]
        labels = ["NADR", "NAFPR"]
        values = [100 * summary["nadr"]["mean"], 100 * summary["nafpr"]["mean"]]
        errors = [100 * summary["nadr"]["std"], 100 * summary["nafpr"]["std"]]
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.bar(labels, values, yerr=errors, color=["#254f87", "#d99132"], capsize=4)
        ax.set_ylabel("Percentage")
        ax.set_title("Zero-day detection and benign false-positive rate")
        fig.tight_layout()
        fig.savefig(Config.FIGURE_DIR / "zero_day_performance.png", dpi=600)
        plt.close(fig)
        xai_summary = results["zero_day"].get("xai_summary", {})
        display_methods = [name for name in ["SHAP", "LIME", "Attention"] if name in xai_summary]
        if display_methods:
            metrics = ["local_fidelity", "rank_consistency", "coverage"]
            positions = np.arange(len(display_methods))
            width = 0.24
            fig, ax = plt.subplots(figsize=(7.0, 4.3))
            for offset, metric in enumerate(metrics):
                values = [xai_summary[method][metric]["mean"] for method in display_methods]
                ax.bar(positions + (offset - 1) * width, values, width, label=metric.replace("_", " ").title())
            ax.set_xticks(positions, display_methods)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Score")
            ax.set_title("Explanation quality on the stratified audit sample")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(Config.FIGURE_DIR / "explanation_quality.png", dpi=600)
            plt.close(fig)
    if "cross_dataset" in results:
        names = [name for name in results["cross_dataset"] if name != "B"]
        values = [100 * results["cross_dataset"][name]["with_da"]["accuracy"]["mean"] for name in names]
        errors = [100 * results["cross_dataset"][name]["with_da"]["accuracy"]["std"] for name in names]
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.bar(names, values, yerr=errors, color="#315f91", capsize=4)
        ax.set_ylabel("Target accuracy (%)")
        ax.set_xlabel("Cross-dataset configuration")
        ax.set_title("Locked target performance after adaptation")
        fig.tight_layout()
        fig.savefig(Config.FIGURE_DIR / "cross_dataset_accuracy.png", dpi=600)
        plt.close(fig)
    if "ablation" in results:
        names, values = [], []
        for name, entry in results["ablation"].items():
            names.append(name.split("_", 1)[-1].replace("_", " "))
            values.append(100 * entry["summary_across_seed_means"]["nadr"]["mean"])
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.barh(names, values, color="#4677a9")
        ax.set_xlabel("Novel attack detection rate (%)")
        ax.set_title("Component ablation")
        fig.tight_layout()
        fig.savefig(Config.FIGURE_DIR / "ablation_nadr.png", dpi=600)
        plt.close(fig)


def self_test() -> None:
    """Fast dependency-light tests for the corrected split and metric invariants."""
    rng = np.random.default_rng(7)
    n = 300
    frame = pd.DataFrame(
        {
            "flow_duration": rng.lognormal(size=n),
            "pkt_count": rng.integers(1, 100, size=n),
            "protocol": rng.choice(["tcp", "udp", "icmp"], size=n),
            "label": np.r_[np.zeros(120), np.ones(180)].astype(np.int8),
            "family": np.r_[
                np.repeat("benign", 120), np.repeat("DDoS", 60),
                np.repeat("DoS", 60), np.repeat("Recon", 60),
            ],
            "__dataset__": "synthetic",
        }
    )
    data = prepare_zero_day_data(frame, "Recon", seed=42, adapt_fraction=0.30)
    assert "Recon" not in set(data.family_train), "held-out family leaked into source training"
    assert "Recon" not in set(data.family_val), "held-out family leaked into source validation"
    assert set(np.unique(data.y_test)) == {0, 1}, "zero-day test must include benign and attacks"
    assert data.novel_test.sum() > 0, "zero-day test must include novel attacks"
    assert len(data.X_adapt) > 0, "transductive adaptation partition must be non-empty"
    assert np.all(np.isfinite(data.X_train.to_numpy()))
    assert np.nanmin(data.X_train.to_numpy()) >= -1e-9
    assert np.nanmax(data.X_train.to_numpy()) <= 1.0 + 1e-9
    example_pred = data.y_test.copy()
    metrics = compute_zero_day_metrics(data.y_test, example_pred, data.novel_test)
    assert np.isfinite(metrics["nadr"]) and np.isfinite(metrics["nafpr"])
    metric_check = compute_zero_day_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 1, 1, 1]),
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.8, 0.7, 0.9]),
    )
    assert math.isclose(metric_check["nadr"], 1.0)
    assert math.isclose(metric_check["nafpr"], 0.5)
    assert np.isfinite(metric_check["zd_auc_roc"]) and np.isfinite(metric_check["zd_auc_pr"])
    ordinary = compute_metrics(
        np.asarray([0, 0, 0, 1]),
        np.asarray([0, 0, 1, 1]),
        np.asarray([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9]]),
    )
    assert "macro_f1" in ordinary and ordinary["macro_f1"] != ordinary["f1"]
    assert "mean" not in _canonical_mapping(["mean", "label"]), "ambiguous generic alias retained"
    grouped = frame.copy()
    grouped["__group__"] = np.repeat([f"session_{index}" for index in range(30)], 10)
    grouped_train, grouped_val, grouped_test = split_source_frame(grouped, seed=9)
    train_groups = set(grouped_train["__group__"])
    val_groups = set(grouped_val["__group__"])
    test_groups = set(grouped_test["__group__"])
    assert not (train_groups & val_groups or train_groups & test_groups or val_groups & test_groups)
    official_examples = {
        "DDoS-UDP_Flood": "DDoS",
        "DoS-TCP_Flood": "DoS",
        "Mirai-greip_flood": "Mirai",
        "DictionaryBruteForce": "Brute_Force",
        "DNS_Spoofing": "Spoofing",
        "Recon-PortScan": "Recon",
        "SqlInjection": "Web_Based",
    }
    for label, expected in official_examples.items():
        assert cic_family(label) == expected, (label, cic_family(label), expected)
    print("Self-test passed: taxonomy, split, preprocessing, and zero-day metrics are valid.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled zero-day and cross-dataset IoT IDS experiments"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["zero_day", "cross_dataset", "baseline", "ablation", "stress", "all"],
    )
    parser.add_argument("--data-dir", default=str(Config.DATA_DIR))
    parser.add_argument("--output-dir", default=str(Config.OUTPUT_DIR))
    parser.add_argument("--results-file", default="final_results.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=Config.RANDOM_SEEDS)
    parser.add_argument("--optuna-trials", type=int, default=Config.OPTUNA_TRIALS)
    parser.add_argument("--ae-epochs", type=int, default=Config.AE_EPOCHS)
    parser.add_argument("--da-epochs", type=int, default=Config.DA_EPOCHS)
    parser.add_argument("--max-rows-per-dataset", type=int, default=Config.MAX_ROWS_PER_DATASET)
    parser.add_argument("--csv-chunk-size", type=int, default=Config.CSV_CHUNK_SIZE)
    parser.add_argument("--xai-audit-max", type=int, default=Config.XAI_AUDIT_MAX)
    parser.add_argument("--lime-num-samples", type=int, default=Config.LIME_NUM_SAMPLES)
    parser.add_argument(
        "--semantic-manifest",
        default=None,
        help="Optional JSON file with audited raw columns and unit conversions per dataset",
    )
    parser.add_argument("--with-xai", action="store_true")
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_environment:
        print(json.dumps(environment_report(), indent=2))
        return 0
    if args.self_test:
        self_test()
        return 0

    Config.configure_paths(args.data_dir, args.output_dir)
    Config.RANDOM_SEEDS = list(args.seeds)
    Config.OPTUNA_TRIALS = max(0, args.optuna_trials)
    Config.AE_EPOCHS = max(1, args.ae_epochs)
    Config.DA_EPOCHS = max(1, args.da_epochs)
    Config.MAX_ROWS_PER_DATASET = max(0, args.max_rows_per_dataset)
    Config.CSV_CHUNK_SIZE = max(1_000, args.csv_chunk_size)
    Config.XAI_AUDIT_MAX = max(1, args.xai_audit_max)
    Config.LIME_NUM_SAMPLES = max(100, args.lime_num_samples)
    semantic_manifest_metadata = None
    if args.semantic_manifest:
        semantic_path = Path(args.semantic_manifest).expanduser().resolve()
        with semantic_path.open("r", encoding="utf-8") as handle:
            Config.SEMANTIC_MANIFEST = json.load(handle)
        semantic_manifest_metadata = {
            "path": str(semantic_path),
            "sha256": sha256_file(semantic_path),
        }
    Config.RUN_XAI = bool(args.with_xai)
    Config.ensure_output_dirs()
    require_training_dependencies(
        require_xai=Config.RUN_XAI,
        require_baselines=args.mode in {"baseline", "all"},
    )

    results: dict[str, Any] = {
        "metadata": {
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "environment": environment_report(),
            "seeds": Config.RANDOM_SEEDS,
            "data_dir": str(Config.DATA_DIR),
            "code_sha256": sha256_file(Path(__file__).resolve()),
            "configuration": {
                "ae_epochs": Config.AE_EPOCHS,
                "da_epochs": Config.DA_EPOCHS,
                "optuna_trials": Config.OPTUNA_TRIALS,
                "max_rows_per_dataset": Config.MAX_ROWS_PER_DATASET or None,
                "csv_chunk_size": Config.CSV_CHUNK_SIZE,
                "with_xai": Config.RUN_XAI,
                "canonical_feature_aliases": Config.CANONICAL_FEATURE_ALIASES,
                "cross_dataset_caution_features": Config.CROSS_DATASET_CAUTION_FEATURES,
                "cross_dataset_configurations": Config.CROSS_DATASET_CONFIGS,
                "excluded_cross_dataset_configurations": Config.EXCLUDED_CROSS_DATASET_CONFIGS,
                "semantic_manifest": semantic_manifest_metadata,
            },
        }
    }
    if args.mode in {"zero_day", "all"}:
        results["zero_day"] = run_zero_day_experiment()
    if args.mode in {"cross_dataset", "all"}:
        results["cross_dataset"] = run_cross_dataset_all()
    if args.mode in {"baseline", "all"}:
        results["baselines"] = run_protocol_baselines()
    if args.mode in {"ablation", "all"}:
        results["ablation"] = run_ablation_study()
    if args.mode in {"stress", "all"}:
        results["stress_iot23"] = run_iot23_stress_test()

    dataset_manifest_path = Config.OUTPUT_DIR / "dataset_manifest.json"
    save_json(DATASET_AUDIT, dataset_manifest_path)
    results["metadata"]["dataset_manifest"] = str(dataset_manifest_path)
    results["metadata"]["environment_artifacts"] = write_environment_lock(Config.OUTPUT_DIR)

    results_path = Config.OUTPUT_DIR / args.results_file
    save_json(results, results_path)
    export_summary_tables(results)
    generate_figures(results)
    print(f"Results: {results_path}")
    print(f"Predictions: {Config.PREDICTIONS_DIR}")
    print(f"Tables: {Config.TABLE_DIR}")
    print(f"Figures: {Config.FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
