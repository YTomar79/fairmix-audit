import pandas as pd

from fairmix_audit.data import _read_acs_task_frame_in_chunks


class TinyTask:
    features = ["AGEP", "COW"]
    target = "PINCP"
    group = None

    @staticmethod
    def df_to_pandas(acs_data):
        features = acs_data[TinyTask.features].copy()
        labels = (acs_data["PINCP"] > 0).astype(int)
        return features, labels, None


def test_chunked_acs_reader_caps_context_before_return(monkeypatch, tmp_path):
    csv_path = tmp_path / "acs.csv"
    rows = [
        {
            "AGEP": 18 + row_id,
            "COW": row_id % 4,
            "PINCP": 1000 if row_id % 2 else 0,
            "PWGTP": 1,
            "UNUSED": "drop-me",
        }
        for row_id in range(100)
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    monkeypatch.setattr("fairmix_audit.data._ensure_acs_file", lambda **_kwargs: csv_path)

    frame = _read_acs_task_frame_in_chunks(
        cache_dir=tmp_path,
        task=TinyTask,
        task_name="ACSIncome",
        state="CA",
        year=2019,
        horizon="1-Year",
        survey="person",
        download=False,
        max_rows=20,
        seed=7,
        chunksize=11,
        label_column="label",
    )

    assert len(frame) == 20
    assert set(frame["label"]) == {0, 1}
    assert set(frame["__task"]) == {"ACSIncome"}
    assert set(frame["__year"]) == {2019}
    assert set(frame["__state"]) == {"CA"}
    assert "UNUSED" not in frame.columns
