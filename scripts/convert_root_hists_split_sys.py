#!/usr/bin/env python3

import os
import sys
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import uproot
import pepper
from tqdm import tqdm


def save_histogram(key, histcol, hist, sysname, output):
    fname = "_".join(key).replace("/", "_") + ".root"
    with uproot.recreate(os.path.join(output, fname)) as f:
        for catkey, histsplit in histcol.hist_split_strcat(hist).items():
            catkey = "_".join(catkey).replace("/", "_")
            f[catkey] = histsplit
    return {key: fname}


def process_histogram(histcol, key, output):
    ret = {}
    hist = histcol.load(key)
    if "sys" not in [ax.name for ax in hist.axes]:
        ret = save_histogram(key, histcol, hist, "nominal", args.output)
    else:
        for sysname in hist.axes["sys"]:
            histsys = hist[{"sys": sysname}]
            ret.update(save_histogram(
                key + (sysname,), histcol, histsys, sysname, args.output))
    return ret


parser = argparse.ArgumentParser(
    description="Convert histograms generated by Pepper into Pepper's old "
    "ROOT layout, with one file per systematic variation. This script is for "
    "backwards compatibility. If you merely want Root histograms, please use "
    "'histogram_format': 'root' in your Pepper config.")
parser.add_argument(
    "hists", help="JSON file inside the histogram directory. Normally named "
    "'hists.json'")
parser.add_argument(
    "--processes", type=int, default=10,
    help="Number of processes to use. Default: 10")
parser.add_argument("output", help="Output directory")
args = parser.parse_args()

output_json = os.path.join(args.output, "hists.json")
if os.path.exists(output_json):
    answer = input(f"{output_json} already exists. Overwrite? [y/n]")
    if answer != "y":
        sys.exit(1)

with open(args.hists) as f:
    histcol = pepper.HistCollection.from_json(f)

os.makedirs(args.output, exist_ok=True)

hist_names = {}
with ProcessPoolExecutor(max_workers=args.processes) as executor:
    futures = []
    for key in histcol.keys():
        fut = executor.submit(process_histogram, histcol, key, args.output)
        futures.append(fut)
    for future in tqdm(as_completed(futures),
                       desc="Converting histograms",
                       total=len(futures)):
        hist_names.update(future.result())

with open(output_json, "w") as f:
    json.dump([[tuple(k) for k in hist_names.keys()],
               list(hist_names.values())], f, indent=4)
