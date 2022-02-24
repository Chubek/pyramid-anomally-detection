import argparse
from algorithms.simple import *
from algorithms.k_means import k_means_clustering
from algorithms.fuzzy_c_means import FuzzyCMeans
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime
import json
import threading
from time import time as t
import time as tt
import sys


class PrintWorking(threading.Thread):
    def __init__(self):
        super().__init__()
        self.ev = threading.Event()
        self.daemon = True

    def run(self):
        t0 = t()
        num_dots = 1
        print("Began...")
        while True:
            t1 = t()

            if (t1 - t0) % 4 == 0:
                msg = "Wait" + "." * num_dots
                sys.stdout.write("\r {:<70}".format(msg))
                sys.stdout.flush()
                tt.sleep(.1)

                if num_dots < 4:
                    num_dots += 1
                else:
                    num_dots = 1

            if self.ev.is_set():
                print("Done!")
                break

    def done(self):
        self.ev.set()


if __name__ == "__main__":
    printer = PrintWorking()
    printer.start()

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

    parser.add_argument("--knn-op", metavar="-KNNOP", type=str, help="Aggfunc for KNN"
                                                                     " defaults to mean, can be median, sum, mean",
                        choices=["sum", "mean", "median"],
                        default="mean")

    parser.add_argument("--knn-lambda", metavar="-KNLAM", type=str, help="Custom aggfunc for KNN"
                                                                         " defaults to None (will be eval'd)",
                        default="None")

    parser.add_argument("--cluster-maxiter", metavar="-CI", type=int, help="Max iter for K-Means and C-Means,"
                                                                           " defaults to 1000",
                        default=1000)

    parser.add_argument("--cluster-tol", metavar="-CT", type=float, help="Tolerance for K-Means and C-Means,"
                                                                         " defaults to 1e-3",
                        default=1e-3)

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

    save_path_csv = os.path.join(parent, f"{name}_{args.operation.strip()}_{fmt_time}.csv")
    save_path_json = os.path.join(parent, f"{name}_{args.operation.strip()}_{fmt_time}.json")
    save_path_png = os.path.join(parent, f"{name}_{args.operation.strip()}_{fmt_time}.png")

    num_iter = args.cluster_maxiter
    tol = args.cluster_tol

    try:
        knn_lam = eval(args.knn_lambda.strip())
    except:
        raise Exception("Invalid KNN lambda")

    if knn_lam is None:
        knn_lam = args.knn_op.strip()

    match args.operation.strip():
        case "Simple-Distance":
            res, max_ = distance_to_all_points(series, distance_name, minowski_norm=minowski_norm)

            print(max_)

            res.to_csv(save_path_csv)

            print(f"Saved to {save_path_csv}")

        case "Distance-to-NN":
            res, max_ = distance_to_nearest_neighbor(series, distance_name, minowski_norm=minowski_norm)

            print(max_)

            res.to_csv(save_path_csv)

            print(f"Saved to {save_path_csv}")

        case "Distance-to-KNN":
            res, max_ = distance_to_k_nearest_neighbor(series, distance_name, op=knn_lam, k=k, minowski_norm=minowski_norm)

            print(max_)

            res.to_csv(save_path_csv)

            print(f"Saved to {save_path_csv}")

        case "K-Means":
            cents, classes = k_means_clustering(series, k=k, distance_name=distance_name, num_iter=num_iter, tol=tol,
                                                minowski_norm=minowski_norm)

            res = []

            for cent, cls in zip(cents, classes):
                res.append({"centroid": cent.tolist(), classes: cls.tolist()})

            with open(save_path_json, "w") as jw:
                json.dump(res, jw)

            print(f"Saved to {save_path_json}")

        case "Fuzzy-C-Means":
            fuzzyc = FuzzyCMeans(series, k, distance_name, minowski_norm=minowski_norm, tol=tol)

            res = fuzzyc(num_iter=num_iter)

            for j in range(len(res)):
                np.random.seed(int("".join([str(int(c)) for c in os.urandom(2)])))
                color = np.random.uniform(0, 1, 3).tolist()
                plt.fill(series, res[j, :], c=color)

            plt.savefig(save_path_png)

            print(f"Saved to {save_path_png}")

        case _:
            raise ValueError("Wrong choice for operation!")

    printer.done()
