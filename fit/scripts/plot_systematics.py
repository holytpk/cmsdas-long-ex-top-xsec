from collections import defaultdict
import numpy as np
import uproot
import argparse
import os
import coffea.hist
import coffea.util
import hjson
from glob import glob
import matplotlib.pyplot as plt
import mplhep as hep

from plot_output import plot_full

parser = argparse.ArgumentParser(
    description="Makes plots with full systematic variations of the final shapes, as well as of the variations themselves for all processes."
        "(Note: this takes a long time for all categories and all processes.)")
parser.add_argument("json", type=str, help="JSON config file containing the fit setup")
parser.add_argument("shapes_folder", type=str, help="Folder containing the shapes used for the combine fit (i.e. the output of export_for_combine.py)")
parser.add_argument("outdir", type=str, help="Plot output directory")
parser.add_argument("--nominal", action="store_true", help="Only plot the nominal in the categories, no variations")
parser.add_argument("--only_sys", type=str, default=None, help="Only plot a specific systematic")
parser.add_argument("--only_proc", type=str, default=None, help="Only plot a specific process")
parser.add_argument("--only_cat", type=str, default=None, help="Only plot a specific category")
parser.add_argument("--ext", type=str, default="png", help="File extension of the plots")
args = parser.parse_args()

def should_plot(proc, sys):
    return not args.nominal and (args.only_sys == None or args.only_sys == sys) \
        and (args.only_proc == None or proc == args.only_proc)

with open(args.json) as f:
    config = hjson.load(f)

os.makedirs(args.outdir, exist_ok=True)

lumi = config["lumi_scale"] if "lumi_scale" in config else 1.
lumi= lumi*1.21

all_procs = [*config["signal_procs"], *config["bg_procs"]]
if "np_background" in config:
    all_procs.append(config["np_background"]["proc"])

for cat in config["categories"]:
    if args.only_cat is not None and cat != args.only_cat:
        continue
    histfile = os.path.join(args.shapes_folder, cat + ".root")
    with uproot.open(histfile) as rfile:

        if not args.nominal:
            cat_out_folder = os.path.join(args.outdir, cat)
            os.makedirs(cat_out_folder, exist_ok=True)

        data_hist = rfile["data_obs"].to_hist()
        data_vals = data_hist.values()
        ax_len = len(data_vals)

        nom_vals_dict = {}
        stat_var = np.zeros(ax_len)
        hi_dict = defaultdict(lambda: np.zeros(ax_len))
        lo_dict = defaultdict(lambda: np.zeros(ax_len))

        for proc in all_procs:

            if not proc in rfile:
                continue

            if not args.nominal:
                proc_out_folder = os.path.join(cat_out_folder, proc)
                os.makedirs(proc_out_folder, exist_ok=True)

            nom_hist = rfile[proc].to_hist()
            nom_vals = nom_hist.values()
            nom_vals_dict[proc] = nom_vals

            nom_vars = nom_hist.variances()
            stat_var += nom_vars

            for sys in config["systematics"]:

                
                if should_plot(proc, sys):
                    fig, (ax1, ax2) = plt.subplots(
                        nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]}, dpi=200)

                    nom_hist.plot(color="black", label="nominal", ax=ax1)
                    ax2.axhline(1, color="black", linestyle="dashed")

                plotlim = 0.
                if proc + "_" + sys + "Up" in rfile:

                    sys_diffs_dirs = []

                    for dir, color in zip(["Up", "Down"], ["orangered", "royalblue"]):
                        sys_hist = rfile[proc + "_" + sys + dir].to_hist()
                        sys_vals = sys_hist.values()

                        sys_diff = sys_vals - nom_vals
                        sys_diffs_dirs.append(sys_diff)

                        if should_plot(proc, sys):
                            sys_hist.plot(color=color, label=sys+dir, ax=ax1)

                            ratio = sys_hist / nom_vals
                            ratio.plot(color=color, label=sys+dir, ax=ax2)
                            ratio_vals = abs(ratio.values() - 1)

                            if np.nanmax(ratio_vals) != np.inf:
                                plotlim = max(plotlim, np.amax(np.nan_to_num(abs(ratio.values()-1))) * 1.1)

                    sys_hi = np.amax(sys_diffs_dirs, axis=0)
                    sys_lo = np.amin(sys_diffs_dirs, axis=0)


                    hi_dict[sys] += sys_hi
                    lo_dict[sys] += sys_lo
                    
                else:
                    print(f"{cat} {proc} {sys} missing in root file")
                    if should_plot(proc, sys):
                        for dir, color in zip(["Up", "Down"], ["orangered", "royalblue"]):
                            nom_hist.plot(color=color, label=sys+dir, ax=ax1)

                if should_plot(proc, sys):

                    ax1.legend(title=cat + " - " + proc, fontsize="large")
                    ax1.annotate(sys, (.5, .98), xycoords="axes fraction", va="top", ha="center", fontsize="large")
                    if plotlim == 0.:
                        plotlim = 0.1
                    ax2.set_ylim(1-plotlim, 1+plotlim)

                    plt.subplots_adjust(hspace=0.2)
                    plt.savefig(os.path.join(proc_out_folder, sys + "." + args.ext), bbox_inches="tight")
                    plt.close()

        #breakpoint()

        hi_tot = np.array(list(hi_dict.values()))
        hi_tot = np.sum(hi_tot**2, axis=0)
        lo_tot = np.array(list(lo_dict.values()))
        lo_tot = np.sum(lo_tot**2, axis=0)


        for sys, sys_conf in config["lnN_uncs"].items():
            if sys != "lumi":
                lnn_var = np.zeros(ax_len)

                for proc, vals in nom_vals_dict.items():
                    if ("procs" not in sys_conf or proc in sys_conf["procs"]) \
                        and ("cats" not in sys_conf or cat in sys_conf["cats"]):
                        lnn_var += vals * sys_conf["unc"]

                hi_tot += lnn_var**2
                lo_tot += lnn_var**2
                        
        plot_outfile = os.path.join(args.outdir, cat + "." + args.ext)

        fig, (ax1, ax2) = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, dpi=200, figsize=(4.8, 4.0))

        plot_full(config, nom_vals_dict, data_vals, stat_var, hi_tot, lo_tot, data_hist.axes[0], lumi=lumi)
        plt.savefig(plot_outfile, bbox_inches="tight")
        plt.close()




