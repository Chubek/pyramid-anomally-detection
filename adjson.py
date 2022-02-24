import json
import sys
import os
import re

if __name__ == "__main__":
    params_file = "./params.json"

    if len(sys.argv) > 1:
        params_file = sys.argv[1]

    with open(params_file, "r") as jr:
        params_json = json.load(jr)

    fin_str_ls = [params_json["python_executable"], "ad.py"]

    del params_json['python_executable']

    if any([v.lower() == "default" for v in [params_json["operation"], params_json["dataset"], params_json["metric"]]]):
        raise ValueError("operation, dataset and metric can't be set to DEFAULT!")

    params_json['metric'] = re.escape(params_json['metric'])

    for k, v in params_json.items():
        if str(v).lower() == "default":
            continue

        fin_str_ls.append(f"--{k.replace('_', '-')} {v}")

    joined = " ".join(fin_str_ls)

    os.system(joined)

