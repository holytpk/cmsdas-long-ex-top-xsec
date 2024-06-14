#!/usr/bin/env python3

import os
from argparse import ArgumentParser
import json
from collections import defaultdict
from functools import partial
import awkward as ak
import uproot
from tqdm import tqdm

import pepper
import pepper.htcondor


def get_counts(lfn, config, geskey, lhesskey, lhepdfskey):
    paths = config.get_paths_for_lfn(lfn)
    for path in paths:
        try:
            f = uproot.open(path, timeout=pepper.misc.XROOTDTIMEOUT)
        except OSError:
            pass
        else:
            break
    else:
        raise OSError(f"Could not open any files for path {lfn}")
    with f:
        if process_name.startswith("WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8"):
            genweight = f["Events/genWeight"].array()
            newweight = np.sign(genweight)
            gen_event_sumw = ak.sum(newweight)

            scaleweights = f["Events/LHEScaleWeight"].array()
            lhe_scale_sumw = ak.sum(newweight * scaleweights, axis=0)

            if lhepdfskey is not None:
                pdfweights = f["Events/LHEPdfWeight"].array()
                lhe_pdf_sumw = ak.sum(newweight * pdfweights, axis=0)
            else:
                lhe_pdf_sumw = ak.Array([])

        else:
            runs = f["Runs"]
            gen_event_sumw = runs[geskey].array()[0]
            lhe_scale_sumw = runs[lhesskey].array()[0]
            lhe_pdf_sumw = ak.Array([])
            has_lhe = len(lhe_scale_sumw) != 0
            if has_lhe:
                lhe_scale_sumw = lhe_scale_sumw * gen_event_sumw
                if lhepdfskey is not None:
                    lhe_pdf_sumw = runs[lhepdfskey].array()[0] * gen_event_sumw
    return lfn, gen_event_sumw, lhe_scale_sumw, lhe_pdf_sumw


def add_counts(counts, datasets, lhepdfskey, result):
    path, gen_event_sumw, lhe_scale_sumw, lhe_pdf_sumw = result
    process_name = datasets[path]
    has_lhe = len(lhe_scale_sumw) != 0
    counts[process_name] += gen_event_sumw
    if has_lhe:
        counts[process_name + "_LHEScaleSumw"] =\
            counts[process_name + "_LHEScaleSumw"] + lhe_scale_sumw
        if lhepdfskey is not None:
            counts[process_name + "_LHEPdfSumw"] =\
                counts[process_name + "_LHEPdfSumw"] + lhe_pdf_sumw


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute factors from luminosity and cross sections to "
                    "scale MC")
    parser.add_argument(
        "config", help="Path to the JSON config file containing the MC "
        "dataset names, luminosity and cross sections. Latter two should be "
        "in 1/fb and fb respectively")
    parser.add_argument("out", help="Path to the output file")
    parser.add_argument(
        "-s", "--skip", action="store_true",
        help="If out exists, skip the data sets that are already in the "
             "output file")
    parser.add_argument(
        "-p", "--pdfsumw", action="store_true", help="Add PdfSumw to the "
        "output file")
    parser.add_argument(
        "-c", "--condor", type=int, metavar="simul_jobs",
        help="Number of HTCondor jobs to launch")
    parser.add_argument(
        "-r", "--retries", type=int, help="Number of times to retry if there "
        "is exception in an HTCondor job. If not given, retry infinitely."
    )
    parser.add_argument(
        "-i", "--condorinit",
        help="Shell script that will be sourced by an HTCondor job after "
        "starting. This can be used to setup environment variables, if using "
        "for example CMSSW. If not provided, the local content of the "
        "environment variable PEPPER_CONDOR_ENV will be used as path to the "
        "script instead.")
    parser.add_argument(
        "--condorsubmit",
        help="Text file containing additional parameters to put into the "
        "HTCondor job submission file that is used for condor_submit"
    )
    parser.add_argument(
        "--condorlogdir", help="Directory to store stdout and stderr logs "
        "running on HTCondor. Default is pepper_logs", default="pepper_logs")
    args = parser.parse_args()

    if args.skip and os.path.exists(args.out):
        with open(args.out) as f:
            factors = json.load(f)
    else:
        factors = {}

    lhepdfskey = "LHEPdfSumw" if args.pdfsumw else None

    config = pepper.ConfigBasicPhysics(args.config)
    lumi = config["luminosity"]
    crosssections = config["crosssections"]
    procs, datasets = config.get_datasets(
        dstype="mc", return_inverse=True, exclude=factors.keys())
    if "dataset_for_systematics" in config:
        dsforsys = config["dataset_for_systematics"]
    else:
        dsforsys = {}
    for process_name in procs.keys():
        if process_name not in crosssections and process_name not in dsforsys:
            raise ValueError(f"Could not find crosssection for {process_name}")

    counts = defaultdict(int)
    get_counts = partial(
        get_counts,
        config=config,
        geskey="genEventSumw",
        lhesskey="LHEScaleSumw",
        lhepdfskey=lhepdfskey
    )
    with pepper.htcondor.Cluster(
        args.condor,
        condorsubmitfile=args.condorsubmit,
        condorinit=args.condorinit,
        retries=args.retries,
        logdir=args.condorlogdir
    ) as cluster:
        cluster.set_global_config()
        for result in tqdm(cluster.process(
                get_counts, datasets.keys()), total=len(datasets)):
            add_counts(counts, datasets, lhepdfskey, result)

    for key in counts.keys():
        if key.endswith("_LHEScaleSumw") or key.endswith("_LHEPdfSumw"):
            dsname = key.rsplit("_", 1)[0]
        else:
            dsname = key
        if dsname in dsforsys:
            xs = crosssections[dsforsys[dsname][0]]
        else:
            xs = crosssections[dsname]
        factor = xs * lumi / counts[key]
        if key.endswith("_LHEScaleSumw") or key.endswith("_LHEPdfSumw"):
            factor = counts[dsname] / counts[key]
        if isinstance(factor, ak.Array):
            factor = list(factor)
        factors[key] = factor
        if key == dsname:
            print(f"{key}: {xs} fb, {counts[key]} events, factor of "
                  f"{factors[key]:.3e}")

    with open(args.out, "w") as f:
        json.dump(factors, f, indent=4)
