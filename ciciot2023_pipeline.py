"""
CIC-IoT2023 — third-dataset fetch/convert pipeline (Section V-I, Table XV, Fig. 11)
===================================================================================
Gets CIC-IoT2023 into the exact CSV shape the EXISTING phase-1 loader already reads, then
runs `phase1_autoencoder_ids.py` unmodified on it. No phase/exp computation lives here.

Source: HuggingFace `lacg030175/CIC-IoT-2023-neto-subsample`, config `random` — a stratified
subsample of the canonical Neto 46.7M re-extraction of CIC-IoT2023 (Neto et al., 2023), with
the canonical **46-flow-feature** NetFlow schema (flow_duration, Header_Length, Rate, Srate,
TCP-flag counts, protocol one-hots, IAT, Magnitue, Covariance, Weight, ...) and a binary
benign/attack label. NOT the full 13-GB / 169-CSV raw release — chosen so the run is tractable
while preserving the canonical feature schema and label semantics.

Expected snapshot (asserted; the script STOPS if the numbers differ):
    train  1,143,802 rows  (160,000 benign / 983,802 attack)
    test     285,951 rows  ( 40,000 benign / 245,951 attack)

Conversion: keep the 46 numeric feature columns verbatim, drop the dataset's own
`Label` / `Label_orig` / `attack_class` / `label` columns, and append a single `class` column
holding the literal strings `normal` / `attack`. That is precisely what
`phase1_autoencoder_ids._find_label_col` + `_to_binary_labels` expect, so the loader,
preprocessor and AE are used UNCHANGED (46 numeric features -> input_dim 46, no one-hot
expansion since the schema carries no categoricals).

RUN:
    python ciciot2023_pipeline.py --dataset-dir /content/drive/MyDrive/dataset --epochs 100
    python ciciot2023_pipeline.py --convert-only          # just write the two CSVs

Outputs:
    $DATASET_DIR/CICIoT2023_Train.csv , $DATASET_DIR/CICIoT2023_Test.csv
    results/ciciot2023_phase1_metrics.json , results/ciciot2023_phase1_recon_hist.png
"""
import argparse, importlib, os, shutil, subprocess, sys, tempfile


def _ensure(mod, pip_name=None):
    try: importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ..."); subprocess.run([sys.executable,"-m","pip","install","-q",pip_name or mod], check=True)


HF_REPO = "lacg030175/CIC-IoT-2023-neto-subsample"
HF_FILES = {"train": "random/train-00000-of-00001.parquet",
            "test":  "random/test-00000-of-00001.parquet"}

# The dataset's own label/metadata columns; everything else is a numeric flow feature.
LABEL_COLS = ["Label", "Label_orig", "attack_class", "label"]

# Snapshot fingerprint. A different subsample would silently change every number downstream,
# so we refuse to continue rather than quietly reproduce something else.
EXPECTED = {"train": dict(rows=1143802, benign=160000, attack=983802),
            "test":  dict(rows=285951,  benign=40000,  attack=245951)}
N_FEATURES = 46


def fetch_split(split: str):
    """Download one split's parquet from the Hub and return it as a DataFrame."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download(HF_REPO, HF_FILES[split], repo_type="dataset")
    print(f"  [{split}] {path}")
    return pd.read_parquet(path)


def check_and_convert(df, split: str):
    """Assert the expected snapshot, then return the 46 features + a `class` column."""
    import numpy as np
    exp = EXPECTED[split]
    feats = [c for c in df.columns if c not in LABEL_COLS]
    n_benign = int((df["label"].astype(int) == 0).sum())
    n_attack = int((df["label"].astype(int) == 1).sum())
    print(f"  [{split}] {len(df)} rows  benign={n_benign}  attack={n_attack}  features={len(feats)}")

    problems = []
    if len(df) != exp["rows"]:      problems.append(f"rows {len(df)} != {exp['rows']}")
    if n_benign != exp["benign"]:   problems.append(f"benign {n_benign} != {exp['benign']}")
    if n_attack != exp["attack"]:   problems.append(f"attack {n_attack} != {exp['attack']}")
    if len(feats) != N_FEATURES:    problems.append(f"features {len(feats)} != {N_FEATURES}")
    if problems:
        raise SystemExit(
            f"\nSTOP — the {split} split does not match the snapshot this paper was run on:\n"
            + "".join(f"    - {p}\n" for p in problems)
            + f"  repo={HF_REPO} file={HF_FILES[split]}\n"
              "  The published CIC-IoT2023 numbers are tied to that snapshot. Re-running on a\n"
              "  different one is legitimate, but the results will NOT be comparable — say so\n"
              "  explicitly rather than proceeding silently.")

    out = df[feats].copy()
    # `normal` / `attack` are the literal strings phase1._to_binary_labels keys on.
    out["class"] = np.where(df["label"].astype(int) == 1, "attack", "normal")
    return out


def main():
    default_dir = os.environ.get("DATASET_DIR", "/content/drive/MyDrive/dataset")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=default_dir,
                    help="where CICIoT2023_Train.csv / _Test.csv are written (default $DATASET_DIR)")
    ap.add_argument("--epochs", type=int, default=100, help="phase-1 autoencoder epochs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--convert-only", action="store_true", help="write the CSVs, skip phase 1")
    ap.add_argument("--force", action="store_true", help="re-download/re-write existing CSVs")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("huggingface_hub"); _ensure("pyarrow")

    os.makedirs(a.dataset_dir, exist_ok=True)
    paths = {s: os.path.join(a.dataset_dir, f"CICIoT2023_{s.capitalize()}.csv") for s in ("train", "test")}

    print("[1] Fetching + converting CIC-IoT2023 ...")
    for split, out_csv in paths.items():
        if os.path.exists(out_csv) and not a.force:
            print(f"  [{split}] {out_csv} already exists (use --force to rebuild)")
            continue
        df = check_and_convert(fetch_split(split), split)
        tmp = out_csv + ".part"
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_csv)
        print(f"  [{split}] wrote {out_csv}  ({df.shape[0]} x {df.shape[1]})")

    if a.convert_only:
        print("\nConvert-only: CSVs ready. Feed them to the existing scripts, e.g.")
        print(f"  python phase1_autoencoder_ids.py --train {paths['train']} --test {paths['test']}")
        return

    # ---- Run the EXISTING phase-1 script, unmodified, on the converted CSVs ----
    here = os.path.dirname(os.path.abspath(__file__))
    results = os.path.join(a.outdir, "results"); os.makedirs(results, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [sys.executable, os.path.join(here, "phase1_autoencoder_ids.py"),
               "--train", paths["train"], "--test", paths["test"],
               "--epochs", str(a.epochs), "--seed", str(a.seed), "--outdir", tmpdir]
        print("\n[2] " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        # phase 1 always writes phase1_metrics.json; keep the NSL-KDD one intact.
        for src, dst in (("phase1_metrics.json", "ciciot2023_phase1_metrics.json"),
                         ("phase1_recon_hist.png", "ciciot2023_phase1_recon_hist.png")):
            s = os.path.join(tmpdir, src)
            if os.path.exists(s):
                shutil.move(s, os.path.join(results, dst))
                print(f"    -> results/{dst}")

    print("\nDone. Table B / Table A rows for CIC-IoT2023 come from the existing scripts:")
    print(f"  python exp1_trivial_learners.py --ciciot-train {paths['train']} --ciciot-test {paths['test']} --epochs 40")
    print(f"  python exp2_static_seeds.py     --ciciot-train {paths['train']} --ciciot-test {paths['test']} --epochs 40 --timesteps 80000")


if __name__ == "__main__":
    main()
