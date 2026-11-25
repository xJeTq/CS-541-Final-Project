"""Compare three valuation methods on later Hudson County sales."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

STATE = "https://www.nj.gov/treasury/taxation/"
ARCHIVES = {
    **{f"Sales{year}.zip": f"lpt/statdata/Sales{year}.zip"
       for year in range(2022, 2027)},
    "YTDSR1A2027.zip": "lpt/statdata/YTDSR1A2027.zip",
    "modiv-2023.zip": "pdf/lpt/modiv-2023.zip",
}
NAMES = dict(zip([f"09{i:02d}" for i in range(1, 13)], [
    "Bayonne", "East Newark", "Guttenberg", "Harrison", "Hoboken", "Jersey City",
    "Kearny", "North Bergen", "Secaucus", "Union City", "Weehawken", "West New York",
]))
SNAPSHOT = "modiv-2023.zip"
SNAPSHOT_SHA256 = "6b3d6bc604524e9f47a5711c058ba27735554cb1e94d4ae30f4c5c1769aad83c"
START = pd.Timestamp("2023-07-01")
TEST = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-01-01")
NUMERIC = ["year_built", "acres", "net_value"]
FEATURES = ["district"] + NUMERIC
CANDIDATES = {
    "Ridge": [1, 10, 100, 1000],
    "Random forest": [5, 15, 40],
}


def fetch(folder, name):
    path = folder / name
    if not path.exists():
        temporary = path.with_suffix(".part")
        with urlopen(STATE + ARCHIVES[name], timeout=90) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        with ZipFile(temporary):
            pass
        temporary.replace(path)
    return path


def fields(record, start, end):
    """Read a one-based, inclusive span in the state's fixed-width layouts."""
    return record[start - 1:end].decode("latin-1").strip().upper()


def parcel_number(value):
    value = value.strip()
    if re.fullmatch(r"\d+(\.\d*)?", value):
        whole, _, decimal = value.partition(".")
        decimal = decimal.rstrip("0")
        return str(int(whole)) + ("." + decimal if decimal else "")
    return value


def sr_component(record, start):
    main = fields(record, start, start + 4)
    suffix = fields(record, start + 5, start + 8).lstrip(".")
    return parcel_number(main + "." + suffix if suffix else main)


def load_sales(path):
    rows = []
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"Expected one sales text file in {path.name}")
        with archive.open(members[0]) as source:
            for row in source:
                if row[:2] != b"09":
                    continue
                if len(row.rstrip(b"\r\n")) != 663:
                    raise ValueError(f"Unexpected SR-1A layout in {path.name}")
                rows.append((
                    fields(row, 1, 4), fields(row, 99, 105), fields(row, 34, 34),
                    fields(row, 627, 629), fields(row, 339, 344), fields(row, 345, 350),
                    fields(row, 47, 55), fields(row, 38, 46), sr_component(row, 351),
                    sr_component(row, 360), fields(row, 620, 624),
                    any(sr_component(row, i) not in ("", "0")
                        for i in range(370, 571, 50)),
                    fields(row, 329, 333), fields(row, 334, 338), path.name,
                ))
    data = pd.DataFrame(rows, columns=[
        "district", "serial", "usable", "class", "deed", "recorded", "price",
        "reported", "block", "lot", "qualifier", "additional", "book", "page", "archive",
    ])
    for column in ["deed", "recorded"]:
        data[column] = pd.to_datetime(data[column], format="%y%m%d", errors="coerce")
    for column in ["price", "reported"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def load_assessments(path):
    rows = []
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist()
                   if "hudson" in name.lower() and name.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"Expected one Hudson assessment file in {path.name}")
    # New Jersey uses Deflate64 in this ZIP. Python's zipfile only lists it.
    with subprocess.Popen(["unzip", "-p", str(path), members[0]], stdout=subprocess.PIPE) as extract:
        for row in extract.stdout:
            if row[:2] != b"09":
                continue
            if len(row.rstrip(b"\r\n")) != 700:
                raise ValueError(f"Unexpected MOD-IV layout in {path.name}")
            rows.append((
                fields(row, 1, 4), parcel_number(fields(row, 5, 13)),
                parcel_number(fields(row, 14, 22)), fields(row, 23, 33),
                fields(row, 34, 35), fields(row, 56, 58), fields(row, 416, 419),
                fields(row, 119, 127), fields(row, 439, 447),
            ))
        if extract.wait() != 0:
            raise ValueError(f"Could not extract assessment records from {path.name}")
    keys = ["district", "block", "lot", "qualifier"]
    assessments = pd.DataFrame(rows, columns=keys + [
        "record_id", "class", "year_built", "acres", "net_value",
    ])
    assessments = assessments.loc[assessments["class"].eq("2") & assessments["record_id"].eq("20")].drop(
        columns=["class", "record_id"]
    ).drop_duplicates()
    ambiguous = int(assessments.duplicated(keys, keep=False).sum())
    assessments = assessments.drop_duplicates(keys, keep=False)
    assessments = assessments.loc[~assessments["block"].isin(["", "0"]) & ~assessments["lot"].isin(["", "0"])]
    for column in NUMERIC:
        assessments[column] = pd.to_numeric(assessments[column], errors="coerce")
    assessments["year_built"] = assessments["year_built"].where(
        assessments["year_built"].between(1700, 2023)
    )
    assessments["acres"] = assessments["acres"].div(10000).where(assessments["acres"].gt(0))
    assessments["net_value"] = assessments["net_value"].where(assessments["net_value"].gt(0))
    return assessments, ambiguous


def prepare(folder, output):
    paths = {name: fetch(folder, name) for name in ARCHIVES}
    provenance = []
    for name, path in paths.items():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if name == SNAPSHOT and digest != SNAPSHOT_SHA256:
            raise ValueError("MOD-IV archive changed. Use the documented snapshot or verify its new vintage.")
        provenance.append({"archive": name, "url": STATE + ARCHIVES[name], "sha256": digest})
    pd.DataFrame(provenance).to_csv(output / "sources.csv", index=False)

    sales = pd.concat([load_sales(paths[name])
                       for name in ARCHIVES
                       if name != SNAPSHOT], ignore_index=True)
    usable = sales["usable"].eq("U") & sales["class"].eq("2")
    by_source = sales.loc[usable].assign(month=lambda x: x["deed"].dt.strftime("%Y-%m"))
    by_source.groupby(["archive", "month"]).size().unstack(fill_value=0).T.to_csv(
        output / "archive_coverage.csv"
    )
    # Serial is state-assigned. Prefer later archive copies before applying filters.
    dedup_keys = ["district", "serial", "deed", "block", "lot", "qualifier"]
    serial_ok = sales["serial"].str.fullmatch(r"[0-9]{7}") & sales["serial"].ne("0000000")
    with_serial = sales.loc[serial_ok].drop_duplicates(dedup_keys, keep="last")
    unique = pd.concat([with_serial, sales.loc[~serial_ok]]).sort_index()
    # Deed book/page catches duplicate matched transfers with different serials.
    deed_keys = ["district", "book", "page", "deed", "block", "lot", "qualifier", "price"]
    book_ok = ~unique["book"].str.fullmatch(r"0*") & ~unique["page"].str.fullmatch(r"0*")
    with_book = unique.loc[book_ok].drop_duplicates(deed_keys, keep="last")
    deed_unique = pd.concat([with_book, unique.loc[~book_ok]]).sort_index()
    scope = deed_unique.loc[deed_unique["deed"].ge(START) & deed_unique["deed"].lt(END)].copy()
    usable = scope.loc[scope["usable"].eq("U") & scope["class"].eq("2")].copy()
    eligible = usable.loc[usable["price"].gt(0)].copy()
    keys = ["district", "block", "lot", "qualifier"]
    eligible["known_key"] = ~eligible["block"].isin(["", "0"]) & ~eligible["lot"].isin(["", "0"])
    eligible["parcel"] = eligible[keys].agg("/".join, axis=1)
    eligible.loc[~eligible["known_key"], "parcel"] = [
        f"unknown/{i}" for i in eligible.index[~eligible["known_key"]]
    ]
    assessments, ambiguous = load_assessments(paths[SNAPSHOT])
    data = eligible.merge(assessments, on=keys, how="left", validate="many_to_one", indicator=True)
    data["matched"] = data.pop("_merge").eq("both")
    data["month"] = data["deed"].dt.strftime("%Y-%m")
    data["year"] = data["deed"].dt.year
    metrics = data.groupby("year").agg(
        n=("price", "size"), matched=("matched", "sum"), low=("price", "min"),
        high=("price", "max"), additional_lot=("additional", "sum"),
        known_key=("known_key", "sum"),
    )
    for field in NUMERIC:
        metrics[field + "_missing"] = data[field].isna().groupby(data["year"]).sum()
    metrics["reported_differs"] = data["reported"].ne(data["price"]).groupby(data["year"]).sum()
    metrics.to_csv(output / "cohort.csv")
    data.groupby("month").size().rename("n").to_csv(output / "monthly.csv")
    data.groupby("year")["price"].quantile([0, .01, .5, .99, 1]).unstack().to_csv(
        output / "price_audit.csv"
    )
    pd.Series({
        "hudson_raw_rows": len(sales),
        "serial_copies_removed": len(sales) - len(unique),
        "additional_deed_copies_removed": len(unique) - len(deed_unique),
        "unparseable_deed_hudson": int(unique["deed"].isna().sum()),
        "date_window_before_filters": len(scope),
        "usable_class_2_in_window": len(usable),
        "nonpositive_or_missing_verified_price": len(usable) - len(eligible),
        "assessment_ambiguous_rows_removed": ambiguous,
    }, name="value").to_csv(output / "filter_audit.csv")
    assert set(data.loc[data["year"].eq(2025), "month"]) == {
        f"2025-{month:02d}" for month in range(1, 13)
    }, "Missing 2025 calendar month"
    assert set(data["district"]) == set(NAMES)
    return data


def training_before(data, cutoff):
    return data.loc[data["deed"].lt(cutoff) & data["recorded"].lt(cutoff)
                    & data["recorded"].ge(data["deed"])]


def median_predictions(train, later):
    medians = train.groupby("district")["price"].median()
    return later["district"].map(medians).fillna(train["price"].median()).to_numpy()


def fit_predict(name, setting, train, later, features=FEATURES):
    if name == "Median":
        return median_predictions(train, later)
    numeric = [field for field in NUMERIC if field in features]
    transform = [("municipality", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False), ["district"])]
    if numeric:
        steps = [SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)]
        if name == "Ridge":
            steps.append(StandardScaler())
        transform.append(("numeric", make_pipeline(*steps), numeric))
    if name == "Ridge":
        estimator = Ridge(alpha=setting)
    else:
        estimator = RandomForestRegressor(n_estimators=300, min_samples_leaf=setting,
                                          random_state=541, n_jobs=1)
    model = make_pipeline(ColumnTransformer(transform), estimator)
    model.fit(train[features], train["price"])
    return model.predict(later[features]).clip(min=0)


def score(actual, predicted):
    residual = np.asarray(predicted) - np.asarray(actual)
    return {"mae": np.abs(residual).mean(), "rmse": np.sqrt((residual ** 2).mean()),
            "bias": residual.mean()}


def choose_settings(data, output):
    rows = []
    for start, end in [("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01")]:
        train = training_before(data, pd.Timestamp(start))
        valid = data.loc[data["deed"].ge(start) & data["deed"].lt(end)
                         & data["recorded"].ge(data["deed"]) & data["recorded"].lt(TEST)]
        for name, settings in {"Median": [0], **CANDIDATES}.items():
            for setting in settings:
                predicted = fit_predict(name, setting, train, valid)
                rows.append({"start": start, "end": end, "model": name, "setting": setting,
                             "n_train": len(train), "n_validation": len(valid),
                             **score(valid["price"], predicted)})
    validation = pd.DataFrame(rows)
    validation.to_csv(output / "validation.csv", index=False)
    averages = validation.groupby(["model", "setting"], as_index=False)["mae"].mean()
    best = averages.sort_values(["mae", "setting"]).drop_duplicates("model")
    best.to_csv(output / "selected.csv", index=False)
    return dict(zip(best["model"], best["setting"]))


def grouped_scores(test, column, minimum=30):
    rows = []
    for group, part in test.groupby(column, observed=True, sort=False):
        for name in ["Median", "Ridge", "Random forest"]:
            row = {"group": group, "model": name, "n": len(part)}
            if len(part) >= minimum:
                row.update(score(part["price"], part[name]))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_intervals(test):
    # Resample parcels, retaining all sales for a sampled parcel in each draw.
    groups = pd.factorize(test["parcel"])[0]
    counts = np.bincount(groups)
    rng = np.random.default_rng(541)
    errors = {}
    for name in ["Median", "Ridge", "Random forest"]:
        residual = test[name] - test["price"]
        errors[name] = (np.bincount(groups, weights=residual.abs()),
                        np.bincount(groups, weights=residual ** 2))
    differences = {"Ridge": [], "Random forest": []}
    for _ in range(2000):
        draw = rng.integers(len(counts), size=len(counts))
        n = counts[draw].sum()
        base_abs, base_sq = errors["Median"]
        for name, values in differences.items():
            abs_error, sq_error = errors[name]
            values.append(((abs_error[draw].sum() - base_abs[draw].sum()) / n,
                           np.sqrt(sq_error[draw].sum() / n) - np.sqrt(base_sq[draw].sum() / n)))
    rows = []
    for name, values in differences.items():
        low, high = np.percentile(values, [2.5, 97.5], axis=0)
        rows.append({"model": name, "mae_low": low[0], "mae_high": high[0],
                     "rmse_low": low[1], "rmse_high": high[1]})
    return pd.DataFrame(rows)


def evaluate(data, output):
    settings = choose_settings(data.loc[data["deed"].lt(TEST)], output)
    train = training_before(data, TEST)
    test = data.loc[data["deed"].ge(TEST)].copy()
    assert train["deed"].max() < test["deed"].min()
    assert train["recorded"].max() < TEST
    test["municipality"] = test["district"].map(NAMES)
    test["overlap"] = np.where(test["parcel"].isin(train["parcel"]), "Seen in training",
                               np.where(test["known_key"], "Unseen parcel", "Unknown key"))
    test["assessment"] = np.where(test["matched"], "Matched", "Unmatched")
    test["parcel_scope"] = np.where(test["additional"], "Additional lot", "Primary lot only")
    cutoffs = train["price"].quantile([.25, .75]).tolist()
    test["price_band"] = pd.cut(test["price"], [-np.inf, *cutoffs, np.inf],
                               labels=["Low", "Middle", "High"])
    pd.Series({"train_n": len(train), "test_n": len(test),
               "train_deed_rows_excluded_by_recorded_cutoff": int(data["deed"].lt(TEST).sum()) - len(train),
               "lower_training_quartile": cutoffs[0], "upper_training_quartile": cutoffs[1],
               "test_distinct_known_parcels": test.loc[test["known_key"], "parcel"].nunique(),
               "test_distinct_seen_parcels": test.loc[test["overlap"].eq("Seen in training"), "parcel"].nunique(),
               }, name="value").to_csv(output / "split.csv")
    rows = []
    for name in ["Median", "Ridge", "Random forest"]:
        test[name] = fit_predict(name, settings[name], train, test)
        rows.append({"model": name, "n": len(test), **score(test["price"], test[name])})
    pd.DataFrame(rows).to_csv(output / "overall.csv", index=False)
    for column in ["municipality", "price_band", "overlap", "assessment", "parcel_scope"]:
        grouped_scores(test, column).to_csv(output / (column + ".csv"), index=False)
    paired_intervals(test).to_csv(output / "intervals.csv", index=False)
    sensitivity = []
    for name in ["Ridge", "Random forest"]:
        locality_only = fit_predict(name, settings[name], train, test, ["district"])
        sensitivity.append({"model": name, "scenario": "Municipality only",
                            **score(test["price"], locality_only)})
    unseen_train = train.loc[~train["parcel"].isin(test["parcel"])]
    for name in ["Median", "Ridge", "Random forest"]:
        prediction = fit_predict(name, settings[name], unseen_train, test)
        sensitivity.append({"model": name, "scenario": "Training overlap removed",
                            **score(test["price"], prediction)})
    pd.DataFrame(sensitivity).to_csv(output / "sensitivity.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    data = prepare(args.data, args.output)
    evaluate(data, args.output)
    print(pd.read_csv(args.output / "overall.csv").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
