import numpy as np
import uproot
import argparse
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser(description="Makes pre- and post-fit plots given the output from combine's FitDiagnostics option.")
parser.add_argument("combine_output", type=str, help="Path to the fitDiagnosticsTest.root file that combine outputs")
parser.add_argument("--include_mcstats", action="store_true")
args = parser.parse_args()

with uproot.open(args.combine_output) as f:
    matrix = f["covariance_fit_s"]
    mvals = matrix.values()
    labels = matrix.axes[0].labels()

    if not args.include_mcstats:
        keep = ["prop" not in l for l in labels]
        mvals = mvals[keep, :][:, list(reversed(keep))]
        labels = [l for l in labels if "prop" not in l]

    figsize = len(labels) / 5

    plt.figure(dpi=100, figsize=(figsize,figsize))

    plt.imshow(mvals)


    tick_range = np.arange(0, len(labels))
    plt.xticks(tick_range, labels[::-1], fontsize="x-small", rotation=90)
    plt.yticks(tick_range, labels, fontsize="x-small")

    plt.colorbar()

    outname = os.path.join(os.path.dirname(args.combine_output), "covariance.pdf")
    plt.savefig(outname, bbox_inches="tight")
    plt.close()