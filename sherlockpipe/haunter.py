import io
import os
import zipfile
from argparse import ArgumentParser

import pandas as pd
import requests

# TODO: Replace with actual Dropbox URL once model zip is uploaded
HAUNTER_MODEL_ZIP_URL = "https://www.dropbox.com/scl/fi/5jc6jubcqnw7dlo5h43y7/Haunter_model_0.0.1.zip?rlkey=l249rvgrqvmpbfhrrn3y8gpx6&st=x0h43hil&dl=1"
HAUNTER_MODEL_VERSION = "1.0.0"
HAUNTER_CACHE_DIR = f"{os.path.expanduser('~')}/.sherlockpipe/haunter"
HAUNTER_MODEL_DIR = f"{HAUNTER_CACHE_DIR}/{HAUNTER_MODEL_VERSION}"
HAUNTER_CV_FOLDS = 10
HAUNTER_EXPECTED_FILES = HAUNTER_CV_FOLDS  # model + chk per fold


def download_model():
    """Download and extract the Haunter CV model zip into ~/.sherlockpipe/haunter/"""
    if os.path.exists(HAUNTER_MODEL_DIR) and len(os.listdir(HAUNTER_MODEL_DIR)) >= HAUNTER_EXPECTED_FILES:
        print(f"Haunter model v{HAUNTER_MODEL_VERSION} already cached at {HAUNTER_MODEL_DIR}")
        return
    os.makedirs(HAUNTER_CACHE_DIR, exist_ok=True)
    print(f"Downloading Haunter model v{HAUNTER_MODEL_VERSION} from Dropbox...")
    r = requests.get(HAUNTER_MODEL_ZIP_URL)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(HAUNTER_MODEL_DIR)
    print(f"Extracted {len(z.namelist())} files to {HAUNTER_MODEL_DIR}")


def predict(input_csv: str, output_csv: str = None, batch_size: int = 512, model_dir: str = None):
    """
    Run Haunter CV predictions on an input CSV.

    Parameters
    ----------
    input_csv : str    Path to CSV with required input columns
    output_csv : str   Path to save predictions (default: input path + '_haunter.csv')
    batch_size : int   Batch size for prediction
    model_dir : str    Path to CV model directory (skips download if provided)
    """
    if model_dir is None:
        download_model()
        model_dir = HAUNTER_MODEL_DIR

    from exoml.haunter.haunter import Haunter
    from exoml.ml.model.base_model import HyperParams

    df = pd.read_csv(input_csv)
    input_cols = ['input_mass_Mearth', 'input_CMF', 'input_Zenv',
                  'input_Zwater_core', 'input_Tsurf_K', 'input_Psurf_bar']
    missing = [c for c in input_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    hp = HyperParams(batch_size=batch_size, epochs=1, initial_learning_rate=0.001,
                     train_percent=0.9, dry_run=True, dropout_rate=0.1)
    haunter = Haunter(hp).build()

    df = haunter.predict_cv(df[input_cols].copy(), model_dir,
                             folds=HAUNTER_CV_FOLDS, batch_size=batch_size)

    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_haunter{ext}"
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    return df


if __name__ == '__main__':
    ap = ArgumentParser(description='Haunter: planetary interior prediction from structure parameters')
    ap.add_argument('--input_csv', required=True, help='Path to input CSV with required columns')
    ap.add_argument('--output_csv', help='Path to save predictions (default: input_haunter.csv)')
    ap.add_argument('--batch_size', type=int, default=512, help='Batch size for prediction')
    ap.add_argument('--download_only', action='store_true', help='Only download the model, skip prediction')
    ap.add_argument('--model_dir', help='Path to local CV model directory (skips Dropbox download)')
    args = ap.parse_args()

    if args.download_only:
        download_model()
    else:
        predict(args.input_csv, args.output_csv, args.batch_size, args.model_dir)
