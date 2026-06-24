from __future__ import annotations

import fcntl
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from fairmix_audit.memory import trim_process_memory


FOLKTABLE_TASKS = {
    "ACSIncome": "ACSIncome",
    "ACSEmployment": "ACSEmployment",
    "ACSPublicCoverage": "ACSPublicCoverage",
    "ACSMobility": "ACSMobility",
    "ACSTravelTime": "ACSTravelTime",
}

SUPPORTED_FOLKTABLE_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY",
]

ContextSpec = tuple[int, str]

TASK_EXTRA_COLUMNS = {
    "ACSIncome": {"PWGTP"},
    "ACSEmployment": {"PWGTP"},
    "ACSPublicCoverage": set(),
    "ACSMobility": set(),
    "ACSTravelTime": {"PWGTP", "ESR"},
}

ACS_DTYPE_OVERRIDES = {
    "PINCP": np.float64,
    "RT": str,
    "SOCP": str,
    "SERIALNO": str,
    "NAICSP": str,
}


def _get_folktables_task(task_name: str):
    try:
        import folktables
    except ImportError as exc:
        raise RuntimeError(
            "folktables is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    if task_name not in FOLKTABLE_TASKS:
        known = ", ".join(sorted(FOLKTABLE_TASKS))
        raise ValueError(f"Unknown Folktables task `{task_name}`. Known tasks: {known}")
    return getattr(folktables, FOLKTABLE_TASKS[task_name])


def task_feature_names(task_name: str) -> list[str]:
    return list(getattr(_get_folktables_task(task_name), "features", []))


def required_acs_columns_for_task(task_name: str) -> list[str]:
    task = _get_folktables_task(task_name)
    return sorted(_required_acs_columns(task, task_name))


def _make_data_source(year: int, horizon: str, survey: str, cache_dir: Path):
    from folktables import ACSDataSource

    try:
        return ACSDataSource(
            survey_year=str(year),
            horizon=horizon,
            survey=survey,
            root_dir=str(cache_dir),
        )
    except TypeError:
        return ACSDataSource(survey_year=str(year), horizon=horizon, survey=survey)


def _task_to_frame(task, acs_data: pd.DataFrame) -> pd.DataFrame:
    acs_data = _normalize_acs_schema(acs_data)
    if hasattr(task, "df_to_pandas"):
        features, labels, _groups = task.df_to_pandas(acs_data)
        frame = features.copy()
        frame["label"] = np.asarray(labels).reshape(-1)
        return frame.reset_index(drop=True)

    features, labels, _groups = task.df_to_numpy(acs_data)
    feature_names = list(getattr(task, "features", [f"x{i}" for i in range(features.shape[1])]))
    frame = pd.DataFrame(features, columns=feature_names)
    frame["label"] = np.asarray(labels).reshape(-1)
    return frame.reset_index(drop=True)


def _normalize_acs_schema(acs_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize ACS column name changes across survey years used by Folktables."""
    if "RELP" not in acs_data.columns and "RELSHIPP" in acs_data.columns:
        acs_data = acs_data.copy()
        acs_data["RELP"] = acs_data["RELSHIPP"]
    return acs_data


def _subsample_context(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if not max_rows or len(frame) <= max_rows:
        return frame
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for label, label_frame in frame.groupby("label", sort=False):
        take = max(1, int(round(max_rows * len(label_frame) / len(frame))))
        take = min(take, len(label_frame))
        parts.append(label_frame.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _optimize_frame_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    optimized = frame.copy()
    for column in optimized.columns:
        series = optimized[column]
        if pd.api.types.is_integer_dtype(series):
            optimized[column] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            optimized[column] = pd.to_numeric(series, downcast="float")
    return optimized


def load_folktables_frames(
    config: dict,
    seed: int,
    *,
    tasks: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    states: Iterable[str] | str | None = None,
    download: bool = True,
) -> pd.DataFrame:
    data_cfg = config["data"]
    tasks = list(tasks or data_cfg["tasks"])
    years = list(years or data_cfg["years"])
    states = resolve_states(states if states is not None else data_cfg["states"])
    cache_dir = Path(data_cfg.get("cache_dir", "data/raw/folktables"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    task_objects = {task_name: _get_folktables_task(task_name) for task_name in tasks}
    for year in years:
        source = _make_data_source(
            year=year,
            horizon=data_cfg.get("horizon", "1-Year"),
            survey=data_cfg.get("survey", "person"),
            cache_dir=cache_dir,
        )
        for state in states:
            for task_name, task in task_objects.items():
                frame = _load_context_frame(
                    config,
                    task=task,
                    task_name=task_name,
                    state=state,
                    year=int(year),
                    download=download,
                    source=source,
                    seed=stable_seed(seed, task_name, year, state),
                )
                frames.append(frame)

    if not frames:
        raise RuntimeError("No data frames were loaded.")
    return pd.concat(frames, ignore_index=True, copy=False)


def load_folktables_contexts(
    config: dict,
    seed: int,
    *,
    task: str,
    contexts: Iterable[ContextSpec],
    download: bool = True,
    max_total_rows: int | None = None,
) -> pd.DataFrame:
    """Load only the requested state/year contexts for a single Folktables task."""
    data_cfg = config["data"]
    contexts = [(int(year), str(state).upper()) for year, state in contexts]
    if not contexts:
        raise RuntimeError(f"No Folktables contexts were requested for task={task}.")
    resolve_states([state for _year, state in contexts])

    cache_dir = Path(data_cfg.get("cache_dir", "data/raw/folktables"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    task_object = _get_folktables_task(task)
    sources = {}
    frames: list[pd.DataFrame] = []
    total_rows = 0
    max_total = int(max_total_rows or 0)
    per_context_cap = int(data_cfg.get("max_rows_per_context") or 0)

    for context_number, (year, state) in enumerate(contexts, start=1):
        if context_number == 1 or context_number == len(contexts) or context_number % 25 == 0:
            print(
                f"[data] task={task} context={context_number}/{len(contexts)} "
                f"year={year} state={state}",
                flush=True,
            )
        source = sources.get(year)
        if source is None:
            source = _make_data_source(
                year=year,
                horizon=data_cfg.get("horizon", "1-Year"),
                survey=data_cfg.get("survey", "person"),
                cache_dir=cache_dir,
            )
            sources[year] = source
        context_seed = stable_seed(seed, task, year, state)
        frame = _load_context_frame(
            config,
            task=task_object,
            task_name=task,
            state=state,
            year=year,
            download=download,
            source=source,
            seed=context_seed,
        )
        frames.append(frame)
        total_rows += len(frame)
        if max_total and total_rows > max_total + per_context_cap:
            frames = [
                _cap_combined_context_rows(
                    frames,
                    max_rows=max_total,
                    seed=stable_seed(seed, task, "stream_cap", context_number),
                    label_column=data_cfg.get("label_column", "label"),
                )
            ]
            total_rows = len(frames[0])
            trim_process_memory()

    if not frames:
        raise RuntimeError(f"No Folktables frames were loaded for task={task}.")
    return pd.concat(frames, ignore_index=True, copy=False)


def _load_context_frame(
    config: dict,
    *,
    task,
    task_name: str,
    state: str,
    year: int,
    download: bool,
    source=None,
    seed: int,
) -> pd.DataFrame:
    data_cfg = config["data"]
    chunksize = int(data_cfg.get("low_memory_reader_chunksize", 50_000) or 0)
    if data_cfg.get("low_memory_reader", True) and chunksize > 0:
        frame = _read_acs_task_frame_in_chunks(
            cache_dir=Path(data_cfg.get("cache_dir", "data/raw/folktables")),
            task=task,
            task_name=task_name,
            state=state,
            year=year,
            horizon=data_cfg.get("horizon", "1-Year"),
            survey=data_cfg.get("survey", "person"),
            download=download,
            max_rows=data_cfg.get("max_rows_per_context"),
            seed=seed,
            chunksize=chunksize,
            label_column=data_cfg.get("label_column", "label"),
        )
    else:
        acs_data = _get_acs_context_data(
            config,
            task=task,
            task_name=task_name,
            state=state,
            year=year,
            download=download,
            source=source,
        )
        frame = _context_data_to_frame(
            config,
            task=task,
            task_name=task_name,
            acs_data=acs_data,
            year=year,
            state=state,
            seed=seed,
        )
        del acs_data
    return frame


def _context_data_to_frame(
    config: dict,
    *,
    task,
    task_name: str,
    acs_data: pd.DataFrame,
    year: int,
    state: str,
    seed: int,
) -> pd.DataFrame:
    data_cfg = config["data"]
    frame = _optimize_frame_dtypes(_task_to_frame(task, acs_data))
    frame["__task"] = task_name
    frame["__year"] = int(year)
    frame["__state"] = state
    return _subsample_context(
        frame,
        max_rows=data_cfg.get("max_rows_per_context"),
        seed=seed,
    )


def _cap_combined_context_rows(
    frames: list[pd.DataFrame],
    *,
    max_rows: int,
    seed: int,
    label_column: str,
) -> pd.DataFrame:
    """Bound in-flight split memory while keeping state/year/label balance."""
    combined = pd.concat(frames, ignore_index=True, copy=False)
    if len(combined) <= int(max_rows):
        return combined

    strata = [column for column in ("__state", "__year", label_column) if column in combined.columns]
    if not strata:
        return combined.sample(n=int(max_rows), random_state=seed).reset_index(drop=True)

    shuffled_index = pd.Series(range(len(combined))).sample(frac=1.0, random_state=seed).to_numpy()
    shuffled = combined.iloc[shuffled_index]
    parts: list[pd.DataFrame] = []
    for _group_key, group in shuffled.groupby(strata, sort=False, observed=True, dropna=False):
        take = max(1, int(round(int(max_rows) * len(group) / len(combined))))
        take = min(take, len(group))
        parts.append(group.head(take))

    capped = pd.concat(parts, ignore_index=True, copy=False)
    if len(capped) > int(max_rows):
        capped = capped.sample(n=int(max_rows), random_state=seed)
    return capped.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prefetch_folktables_data(
    config: dict,
    *,
    years: Iterable[int] | None = None,
    states: Iterable[str] | str | None = None,
) -> list[dict[str, int | str]]:
    data_cfg = config["data"]
    years = list(years or data_cfg["years"])
    states = resolve_states(states if states is not None else data_cfg["states"])
    cache_dir = Path(data_cfg.get("cache_dir", "data/raw/folktables"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloads: list[dict[str, int | str]] = []
    for year in years:
        for state in states:
            file_path = _ensure_acs_file(
                cache_dir=cache_dir,
                state=state,
                year=int(year),
                horizon=data_cfg.get("horizon", "1-Year"),
                survey=data_cfg.get("survey", "person"),
                download=True,
            )
            downloads.append(
                {
                    "year": int(year),
                    "state": state,
                    "rows": _count_csv_rows(file_path),
                }
            )
    return downloads


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "::".join(str(part) for part in (base_seed, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


def resolve_states(states: Iterable[str] | str) -> list[str]:
    if states == "ALL":
        return SUPPORTED_FOLKTABLE_STATES
    states = [str(state).upper() for state in states]
    if len(states) == 1 and states[0] == "ALL":
        return SUPPORTED_FOLKTABLE_STATES

    invalid = sorted(set(states) - set(SUPPORTED_FOLKTABLE_STATES))
    if invalid:
        raise ValueError(
            "Unsupported Folktables states requested: "
            f"{', '.join(invalid)}. Supported values are the 50 U.S. states only."
        )
    return states


def _get_acs_data(
    source,
    state: str,
    *,
    cache_dir: Path,
    year: int,
    horizon: str,
    survey: str,
    download: bool,
) -> pd.DataFrame:
    if not download:
        try:
            return source.get_data(states=[state], download=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Missing Folktables cache for state={state}, year={year}. "
                "Enable downloads (download=True) to fetch this ACS slice."
            ) from exc

    lock_path = _download_lock_path(cache_dir, year=year, horizon=horizon, survey=survey, state=state)
    with _locked_file(lock_path):
        return source.get_data(states=[state], download=True)


def _get_acs_context_data(
    config: dict,
    *,
    task,
    task_name: str,
    state: str,
    year: int,
    download: bool,
    source=None,
) -> pd.DataFrame:
    data_cfg = config["data"]
    cache_dir = Path(data_cfg.get("cache_dir", "data/raw/folktables"))
    horizon = data_cfg.get("horizon", "1-Year")
    survey = data_cfg.get("survey", "person")
    if data_cfg.get("low_memory_reader", True):
        return _read_acs_task_columns(
            cache_dir=cache_dir,
            task=task,
            task_name=task_name,
            state=state,
            year=year,
            horizon=horizon,
            survey=survey,
            download=download,
        )

    if source is None:
        source = _make_data_source(year=year, horizon=horizon, survey=survey, cache_dir=cache_dir)
    return _get_acs_data(
        source,
        state,
        cache_dir=cache_dir,
        year=year,
        horizon=horizon,
        survey=survey,
        download=download,
    )


def _read_acs_task_columns(
    *,
    cache_dir: Path,
    task,
    task_name: str,
    state: str,
    year: int,
    horizon: str,
    survey: str,
    download: bool,
) -> pd.DataFrame:
    file_path = _ensure_acs_file(
        cache_dir=cache_dir,
        state=state,
        year=year,
        horizon=horizon,
        survey=survey,
        download=download,
    )
    required_columns = _required_acs_columns(task, task_name)
    usecols = _resolve_acs_usecols(file_path, required_columns)
    dtypes = {column: ACS_DTYPE_OVERRIDES[column] for column in usecols if column in ACS_DTYPE_OVERRIDES}
    frame = pd.read_csv(file_path, usecols=usecols, dtype=dtypes)
    frame = frame.replace(" ", "")
    return _normalize_acs_schema(frame)


def _read_acs_task_frame_in_chunks(
    *,
    cache_dir: Path,
    task,
    task_name: str,
    state: str,
    year: int,
    horizon: str,
    survey: str,
    download: bool,
    max_rows: int | None,
    seed: int,
    chunksize: int,
    label_column: str,
) -> pd.DataFrame:
    file_path = _ensure_acs_file(
        cache_dir=cache_dir,
        state=state,
        year=year,
        horizon=horizon,
        survey=survey,
        download=download,
    )
    required_columns = _required_acs_columns(task, task_name)
    usecols = _resolve_acs_usecols(file_path, required_columns)
    dtypes = {column: ACS_DTYPE_OVERRIDES[column] for column in usecols if column in ACS_DTYPE_OVERRIDES}
    max_rows_int = int(max_rows or 0)
    chunksize = max(1, int(chunksize))

    frames: list[pd.DataFrame] = []
    total_rows = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(file_path, usecols=usecols, dtype=dtypes, chunksize=chunksize),
        start=1,
    ):
        chunk = chunk.replace(" ", "")
        frame = _optimize_frame_dtypes(_task_to_frame(task, _normalize_acs_schema(chunk)))
        if frame.empty:
            continue
        if max_rows_int and len(frame) > max_rows_int:
            frame = _subsample_context(
                frame,
                max_rows=max_rows_int,
                seed=stable_seed(seed, "csv_chunk", chunk_number),
            )
        frames.append(frame)
        total_rows += len(frame)
        if max_rows_int and total_rows > max_rows_int:
            frames = [
                _cap_combined_context_rows(
                    frames,
                    max_rows=max_rows_int,
                    seed=stable_seed(seed, "csv_stream_cap", chunk_number),
                    label_column=label_column,
                )
            ]
            total_rows = len(frames[0])
            trim_process_memory()

    if not frames:
        columns = list(getattr(task, "features", [])) + [label_column, "__task", "__year", "__state"]
        return pd.DataFrame(columns=columns)

    combined = pd.concat(frames, ignore_index=True, copy=False)
    if max_rows_int and len(combined) > max_rows_int:
        combined = _cap_combined_context_rows(
            [combined],
            max_rows=max_rows_int,
            seed=stable_seed(seed, "csv_final_cap"),
            label_column=label_column,
        )
        trim_process_memory()
    combined["__task"] = task_name
    combined["__year"] = int(year)
    combined["__state"] = state
    return combined.reset_index(drop=True)


def _required_acs_columns(task, task_name: str) -> set[str]:
    columns = set(getattr(task, "features", []) or [])
    target = getattr(task, "target", None)
    if target:
        columns.add(str(target))
    group = getattr(task, "group", None)
    if group:
        columns.add(str(group))
    columns.update(TASK_EXTRA_COLUMNS.get(task_name, set()))
    if not columns:
        raise RuntimeError(f"Could not infer ACS columns for Folktables task={task_name}.")
    return columns


def _ensure_acs_file(
    *,
    cache_dir: Path,
    state: str,
    year: int,
    horizon: str,
    survey: str,
    download: bool,
) -> Path:
    try:
        from folktables.load_acs import initialize_and_download
    except ImportError as exc:
        raise RuntimeError(
            "folktables is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    base_datadir = cache_dir / str(year) / horizon
    base_datadir.mkdir(parents=True, exist_ok=True)
    if not download:
        try:
            return Path(
                initialize_and_download(
                    str(base_datadir),
                    state,
                    year,
                    horizon,
                    survey,
                    download=False,
                )
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Missing Folktables cache for state={state}, year={year}. "
                "Enable downloads (download=True) to fetch this ACS slice."
            ) from exc

    lock_path = _download_lock_path(cache_dir, year=year, horizon=horizon, survey=survey, state=state)
    with _locked_file(lock_path):
        return Path(
            initialize_and_download(
                str(base_datadir),
                state,
                year,
                horizon,
                survey,
                download=True,
            )
        )


def _resolve_acs_usecols(file_path: Path, required_columns: set[str]) -> list[str]:
    available = set(pd.read_csv(file_path, nrows=0).columns)
    usecols: list[str] = []
    missing: list[str] = []
    for column in sorted(required_columns):
        actual_column = column
        if actual_column not in available and column == "RELP" and "RELSHIPP" in available:
            actual_column = "RELSHIPP"
        if actual_column not in available:
            missing.append(column)
        else:
            usecols.append(actual_column)
    if missing:
        raise RuntimeError(
            f"Missing required ACS columns in {file_path}: {', '.join(missing)}"
        )
    return usecols


def _count_csv_rows(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def _download_lock_path(cache_dir: Path, *, year: int, horizon: str, survey: str, state: str) -> Path:
    token = f"{survey}_{horizon}_{year}_{state}".replace("/", "_")
    return cache_dir / ".locks" / f"{token}.lock"


@contextmanager
def _locked_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def combine_columns(frame: pd.DataFrame, columns: list[str], sep: str = "|") -> pd.Series:
    if not columns:
        raise ValueError("At least one column is required.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return frame[columns].astype(str).agg(sep.join, axis=1)


def feature_columns(frame: pd.DataFrame, config: dict) -> list[str]:
    data_cfg = config["data"]
    excluded = {
        data_cfg.get("label_column", "label"),
        "__task",
        *data_cfg.get("protected_attributes", []),
        *data_cfg.get("context_attributes", []),
    }
    return [column for column in frame.columns if column not in excluded]
