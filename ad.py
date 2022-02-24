import argparse
from algorithms.simple import *
from algorithms.k_means import k_means_clustering
from algorithms.fuzzy_c_means import FuzzyCMeans
import pandas
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect anomaly within a single feature")

    parser.add_argument("--dataset", metavar="-D", type=str, help="Path to dataset", required=True)
    parser.add_argument("--operation", metavar="-OP", type=str, help="The operation. Choose from: \n"
                                                                     "1- Simple-Distance\n"
                                                                     "2- Distance-to-NN\n"
                                                                     "3- Distance-to-KNN\n"
                                                                     "4- K-Means"
                                                                     "5- Fuzzy-C-Means",
                        default="K-Means",
                        choices=[
                            "Simple-Distance",
                            "Distance-to-NN",
                            "Distance-to-KNN",
                            "K-Means",
                            "Fuzzy-C-Means"
                        ],
                        required=True)
    parser.add_argument("--metric", metavar="-M", type=str, help="The metric to choose from the dataset",
                        required=True)
    parser.add_argument("--distfunc", metavar="-F", type=str, help="Distance function, default Euclidean",
                        default="euclidean")
    parser.add_argument("--clusters", metavar="-KC", type=int,
                        help="K (non-fuzzy) or C (fuzzy) for clusters, default 3 clusters",
                        default=3)

    parser.add_argument("--minowski-norm", metavar="-MN", type=int, help="Minowski norm, by default 2 aka Euclidean",
                        default=2)

    parser.add_argument("--dt-fmt", metavar="-DTMFT", type=str, help="Datetime format for save file,"
                                                                     " defaults to %d-%HH-%MM-%s",
                        default="%d-%HH-%MM-%s")

    parser.add_argument("--save-dir", metavar="-SVDIR", type=str, help="Save dir,"
                                                                     " defaults to SAME (same place as df)",
                        default="SAME")

    args = parser.parse_args()

    if not os.path.exists(args.dataset.strip()):
        raise Exception("Dataset does not exist!")

    suffix = args.dataset.split(".")[-1].lower()

    match suffix:
        case ("xlsx" | "xls"):
            df = pd.read_excel(args.dataset.strip())
        case "csv":
            df = pd.read_csv(args.dataset.strip())
        case "tsv":
            df = pd.read_csv(args.dataset.strip(), sep="\t")
        case _:
            raise Exception("Format not supported, must be CSV, TSV or Excel")

    if args.metric.strip() not in df.columns:
        raise Exception("Dataset does not contain metric!")

    series = df.loc[:, args.metric.strip()].values

    distance_name = args.distfunc.strip()
    minowski_norm = args.minowski_norm
    k = args.clusters

    path_ = Path(args.dataset.strip())

    parent = path_.parents[0]
    name = path_.stem

    time = datetime.datetime.today()
    fmt_time = time.strftime(args.dt_fmt.strip())

    if args.save_dir.strip() != "SAME":
        parent = args.save_dir.strip()
        if not os.path.exists(parent):
            os.makedirs(parent)

    save_path_csv = os.path.join(parent, f"{name}_{fmt_time}.csv")
    save_path_png = os.path.join(parent, f"{name}_{fmt_time}.png")

    match args.operation.strip():
        case "Simple-Distance":
