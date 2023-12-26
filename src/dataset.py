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
    # Returns a list of dictionaries:
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
        metadata = {
            x.strip(): y.strip()
            for x, y in [x.split(":", maxsplit=1) for x in metadata]
        }

        age = metadata["Age"].replace("+", "")
        if age == "NaN":
            age = np.nan
        else:
            age = int(age)

        gender = metadata["Gender"]

        patient_data = {
            "id": pat_id,
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


def split_dataset_by_age(dataset, ages=[0, 50, 60, 70, 80, 100], return_names=True):
    # Split the dataset
    # Returns a list of datasets
    datasets = []
    for i in range(len(ages) - 1):
        datasets.append([])

    for patient in dataset:
        age = patient["age"]
        for i in range(len(ages) - 1):
            if age >= ages[i] and age < ages[i + 1]:
                datasets[i].append(patient)
                break

    if not return_names:
        return datasets

    return datasets, [
        "Ages {}-{}".format(ages[i], ages[i + 1]) for i in range(len(ages) - 1)
    ]


def split_dataset_by_gender(dataset, genders=["Male", "Female"], return_names=True):
    # Split the dataset
    # Returns a list of datasets
    datasets = []
    for i in range(len(genders)):
        datasets.append([])

    for patient in dataset:
        gender = patient["gender"]
        if gender == genders[0]:
            datasets[0].append(patient)
        else:
            datasets[1].append(patient)

    if not return_names:
        return datasets

    return datasets, genders


def split_datasets_to_windows(
    datasets, metric_agg="mean", window_size=5, window_stride=1
):
    # Split the datasets into windows
    # Returns a list of datasets
    window_datasets = []
    for dataset in datasets:
        window_dataset = []
        for patient in dataset:
            window_dataset += split_patient_to_windows(
                patient, metric_agg, window_size, window_stride
            )
        window_datasets.append(window_dataset)

    return window_datasets


def split_patient_to_windows(
    patient, metric_agg="mean", window_size=30, window_stride=30, ignore_nans=True
):
    # Split the patient into windows
    # Returns a list of windows
    window_dataset = []
    ppg = patient["ppg"]
    ecg = patient["ecg"]
    hr = patient["hr"]
    rr = patient["rr"]
    time = patient["time"]

    window_len_signal = window_size * patient["signal_hz"]
    window_len_metrics = window_size * patient["metrics_hz"]

    num_samples = (
        int((len(ppg) / patient["signal_hz"] - window_size) // window_stride) + 1
    )

    num_samples = int(num_samples)

    # Split the signals into windows
    for i in range(0, num_samples):
        signal_start = i * window_stride * patient["signal_hz"]
        signal_end = signal_start + window_len_signal
        metrics_start = i * window_stride * patient["metrics_hz"]
        metrics_end = metrics_start + window_len_metrics
        window = {
            "ppg": ppg[signal_start:signal_end],
            "ecg": ecg[signal_start:signal_end],
            "hr": hr[metrics_start:metrics_end],
            "rr": rr[metrics_start:metrics_end],
            "time": time[signal_start:signal_end],
        }

        # Fixed characteristics
        window["signal_hz"] = patient["signal_hz"]
        window["metrics_hz"] = patient["metrics_hz"]
        window["id"] = patient["id"]
        window["age"] = patient["age"]
        window["gender"] = patient["gender"]

        if metric_agg == "mean":
            window["hr"] = np.mean(window["hr"])
            window["rr"] = np.mean(window["rr"])

        if ignore_nans:
            if np.isnan(window["hr"]) or np.isnan(window["rr"]):
                continue

        window_dataset.append(window)

    return window_dataset
