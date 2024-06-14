import numpy as np
import uproot
import argparse
import hjson
import os
import hist as hhist
import matplotlib.pyplot as plt

from plot_output import plot_full


parser = argparse.ArgumentParser(description="Makes pre- and post-fit plots given the output from combine's FitDiagnostics option.")
parser.add_argument("json", type=str, help="JSON config file containing the fit setup")
parser.add_argument("combine_output", type=str, help="Path to the fitDiagnosticsTest.root file that combine outputs")
parser.add_argument("outdir", type=str, help="Output directory for pre- and post-fit plots")
args = parser.parse_args()

with open(args.json) as f:
    config = hjson.load(f)

os.makedirs(args.outdir, exist_ok=True)

all_procs = [*config["signal_procs"], *config["bg_procs"]]
if "np_background" in config:
    all_procs.append(config["np_background"]["proc"])

with uproot.open(args.combine_output) as rfile:
    for label, key in zip(["Pre-fit", "Post-fit"], ["prefit", "fit_s"]):
        for cat in config["categories"]:

            print(f"--- {cat} {label} ---")

            if "vars" in config and cat in config["vars"]:
                cat_var = config["vars"][cat]
            else:
                cat_var = config["var"]

            binning = config["rebin"][cat]
            if "hi" in binning:
                axis = hhist.axis.Regular(bins=binning["n_or_arr"], start=binning["lo"], stop=binning["hi"], name=cat_var, label=config["ax_label"])
                nbins = binning["n_or_arr"]
            else:
                axis = hhist.axis.Variable(binning["n_or_arr"], name=cat_var, label=config["ax_label"])
                nbins = len(binning["n_or_arr"]) - 1 

            data_path = f"shapes_{key}/{cat}/data"

            if not data_path in rfile:
                continue

            data_json = rfile[data_path].tojson()
            data_vals = np.array(data_json["fY"])
            data_vals = data_vals[:nbins]

            nom_vals_dict = {}

            for proc in all_procs:
                proc_key = f"shapes_{key}/{cat}/{proc}"
                if proc_key in rfile:
                    h = rfile[proc_key].to_hist()
                    vals = h.values()
                    nom_vals_dict[proc] = vals[:nbins]

            total_hist = rfile[f"shapes_{key}/{cat}/total"].to_hist()

            lo_tot = total_hist.variances()[:nbins]
            hi_tot = lo_tot

            outname = os.path.join(args.outdir, f"{cat}_{key}.pdf")


            fig, (ax1, ax2) = plot_full(config, nom_vals_dict, data_vals, np.zeros_like(data_vals), hi_tot, lo_tot, axis, use_stat_var=False)

            ax1.annotate(cat + "\n" + label, (0.02, 0.98), xycoords="axes fraction", va="top")
            plt.savefig(outname, bbox_inches="tight")
            plt.close()

