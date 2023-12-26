# Import BIDMC PPG dataset

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal

root_path = "./dataset"


# Import data
def read_bidmc_dataset(path=root_path, num_patients=53):
    csv_path = os.path.join(path, "bidmc_csv")
    csv_template = "bidmc_{:0>2}"
    # Read the csv files
    dataset = []
    for i in range(num_patients):
        pat_id = i + 1

        sig_path = os.path.join(csv_path, csv_template.format(pat_id) + "_Signals.csv")

        # Read the csv file
        df = pd.read_csv(sig_path)
        # Strip column names
        df.columns = df.columns.str.strip()

        # Get the signals
        ppg = df["PLETH"].values
        time = df["Time [s]"].values
        ecg = df["II"].values
        signal_hz = 125

        numerics_path = os.path.join(
            csv_path, csv_template.format(pat_id) + "_Numerics.csv"
        )
        df = pd.read_csv(numerics_path)
        df.columns = df.columns.str.strip()

        # Get the heart rate
        hr = df["HR"].values
        rr = df["RESP"].values
        metrics_hz = 1

        metadata_path = os.path.join(csv_path, csv_template.format(pat_id) + "_Fix.txt")

        # Read the metadata
        metadata = open(metadata_path, "r").readlines()

        metadata = [x.strip() for x in metadata if ":" in x]
        metadata = {x.strip(): y.strip() for x, y in [x.split(":") for x in metadata]}

        age = metadata["Age"]
        gender = metadata["Gender"]

        patient_data = {
            "id": id,
            "ppg": ppg,
            "ecg": ecg,
            "time": time,
            "signal_hz": signal_hz,
            "hr": hr,
            "rr": rr,
            "metrics_hz": metrics_hz,
            "age": age,
            "gender": gender,
        }

        dataset.append(patient_data)

    return dataset
