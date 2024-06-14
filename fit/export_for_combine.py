import numpy as np
import hjson
import coffea.util
import argparse
import uproot
import os

# Datacard creation is in its own file
from make_datacard import make_datacard

# Hist unfortunately does not have a flexible enough rebining method.
# I took this function from Kenneth Long (https://github.com/kdlong)
from variable_rebin import rebin_hist

# Suppress annoying warnings from hist
import warnings
warnings.filterwarnings("ignore", "List indexing selection is experimental")

# Command line arguments
parser = argparse.ArgumentParser(description="Exports pepper output to root files for use in combine and creates datacards.")
parser.add_argument("json", type=str, help="JSON config file containing the fit setup")
parser.add_argument("histogram", type=str, help="Path to the histogram file")
parser.add_argument("outdir", type=str, help="Output directory for the root files and datacards")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# Load the json config
with open(args.json) as f:
    config = hjson.load(f)

all_procs = {**config["bg_procs"], **config["signal_procs"], "data_obs": config["data"]}

h = coffea.util.load(args.histogram)


def get_updown_syst(h, sys_conf, h_nom):
    # Handle a (possibly asymmetric) up/down type systematic
    sys_key = sys_conf["hist"]
    return (h[{"sys": sys_key + "_" + direction}] for direction in ("up", "down"))

def get_single_syst(h, sys_conf, h_nom):
    # Handle one-sided uncertainties
    # Can either be symmetrized or kept one-sided
    sys_key = sys_conf["hist"] 
    if "index" in sys_conf:
        sys_key += "_" + str(sys_conf["index"]) 

    onesided = "onesided" in sys_conf and sys_conf["onesided"]

    h_var = h[{"sys": sys_key}]

    if onesided:
        return (h_var, h_nom)
    else:
        # Symmetrize around the nominal values
        diff = h_var.values() - h_nom.values()
        h_mirror = h_var.copy()
        h_mirror[:] = np.stack([h_nom.values() - diff, h_var.variances()]).T
        return (h_var, h_mirror)

def get_envelope_syst(h, sys_conf, h_nom):
    # Handle a systematic where the envelope should be taken from many variations.
    # e.g. PDF uncertainty
    sys_key = sys_conf["hist"]
    # Get all possible variations stored in the histogram
    variations = [var for var in h.axes["sys"] if var.startswith(sys_key + "_")]

    var_sum = np.zeros_like(h_nom.values())
    for var in variations:
        h_var = h[{"sys": var}]
        var_sum += (h_var.values() - h_nom.values())**2
    envelope = np.sqrt(var_sum)

    h_up = h_nom.copy()
    h_down = h_nom.copy()

    h_up[:] = np.stack([h_nom.values() + envelope, h_nom.variances()]).T
    h_down[:] = np.stack([h_nom.values() - envelope, h_nom.variances()]).T

    return (h_up, h_down)
       

# Bins with no events - used for writing the datacard later
excluded_bins = []

# Loop over all categories
for cat_out_name, cat_conf in config["categories"].items():

    print(f"------ {cat_out_name} -----")
    outpath = os.path.join(args.outdir, cat_out_name + ".root")

    # Get the correct categories from the histogram, as specified
    # in the config
    cat_conf = {k: sum if v == "all" else v for k,v in cat_conf.items()}
    h_cat = h[cat_conf]

     # Create the output ROOT file for the category
    with uproot.recreate(outpath) as outf:

        for proc_out_name, proc_conf in all_procs.items():

            # Single out the correct datasets
            h_proc_cat = h_cat[{"dataset": proc_conf}][{"dataset": sum}]

            dense_axes = [ax for ax in h_proc_cat.axes if not ax.name == "sys"]
            assert len(dense_axes) == 1
            dense_ax = dense_axes[0]

            if cat_out_name in config["rebin"]:
                new_binning = config["rebin"][cat_out_name]
                h_proc_cat = rebin_hist(h_proc_cat, dense_ax.name, new_binning)

            # Nominal histogram
            h_nom = h_proc_cat[{"sys": "nominal"}]

            # Exclude bin for process if nominal has no events
            if np.all(h_nom.values() == 0.):
                excluded_bins.append((cat_out_name,proc_out_name))
                print(f"Bin {cat_out_name} {proc_out_name} has no events, excluding")
                continue


            outf[proc_out_name] = h_nom

            if proc_out_name != "data_obs":
                # Handle systematics
                for sys_out_name, sys_conf in config["systematics"].items():

                    # Option to consider specific systs only for specific procs
                    if "procs" in sys_conf and proc_out_name not in sys_conf["procs"]:
                        continue

                    if "cats" in sys_conf and cat_out_name not in sys_conf["cats"]:
                        continue

                    if not("type" in sys_conf) or sys_conf["type"] == "updown":
                        var_up, var_down = get_updown_syst(h_proc_cat, sys_conf, h_nom)

                    elif sys_conf["type"] == "single":
                        var_up, var_down = get_single_syst(h_proc_cat, sys_conf, h_nom)

                    elif sys_conf["type"] == "envelope":
                        var_up, var_down = get_envelope_syst(h_proc_cat, sys_conf, h_nom)

                    else:
                        raise ValueError(f"Unsupported systematic type for {sys_out_name}: {sys_conf['type']}")
                    
                    # Write the systematic templates to the root files
                    outf[proc_out_name + "_" + sys_out_name + "Up"] = var_up
                    outf[proc_out_name + "_" + sys_out_name + "Down"] = var_down
                        
# Make a datacard with all categories combined                       
make_datacard(config, os.path.join(args.outdir, "all.txt"), exclude=excluded_bins)
