import argparse,os
from multiprocessing.sharedctypes import Value
import json
import numpy as np
import subprocess
import re

parser = argparse.ArgumentParser()

def read_parameters(output):
    token = "Floating Parameter"
    if token in output:
        lines = output.split(token)[-1].split("\n")[2:]
        params = {}
        for l in lines:
            s = l.strip()
            s = re.sub(r"\s+", " ", s)
            if s == "":
                break
            else:
                split = s.split(" ")
                param = split[0]
                val = float(split[2])
                params[param] = val
        return params
    else:
        raise ValueError("Could not extract parameters!")

def read_fit_result(output):
    token = "Best fit r:"
    if token in output:
        str = output.split(token)[1].split("(68% CL)")[0]
        vals = re.findall(r"\d+\.?\d*", str)
        return [float(v) for v in vals]
    else:
        print("OUTPUT---",output)
        raise ValueError("No best fit value for r in output!")

parser.add_argument("cardDir",type=str,action="store", default="all")
parser.add_argument("cardName",type=str,action="store", default="all")
parser.add_argument("-o","--outName",type=str,action="store", default="")
parser.add_argument("-d", "--data", action="store_true")

args = parser.parse_args()

#options = "--cminDefaultMinimizerStrategy 0 --cminPreFit 1 --cminDefaultMinimizerTolerance 0.005 --stepSize 0.001 --robustFit 1 --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 --X-rtd MINIMIZER_no_analytic --cminPreScan --skipBOnlyFit --rMin=-2 --rMax=2"

options = "--cminDefaultMinimizerStrategy 1 --cminPreFit 1 --cminDefaultMinimizerTolerance 0.005 --stepSize 0.005 --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 --cminPreScan --skipBOnlyFit --rMin=-2 --rMax=2"

if not args.data:
    options = "-t -1 --expectSignal=1 " + options

os.chdir(args.cardDir)
logdir = "systtable_{}{}_out".format(args.cardName,args.outName)

if not os.path.isdir(logdir):
    os.mkdir(logdir)

os.chdir(logdir)

cmd_t2w = "combineTool.py -M T2W -i ../{0}.txt -o {2}/workspace.root".format(args.cardName,args.outName,logdir)

os.system(cmd_t2w)


cmd_initial = "combine -M FitDiagnostics {0} -v 3 -n initial workspace.root".format(options)

print("Running command: {}".format(cmd_initial))
result_initial = subprocess.check_output(cmd_initial.split(" "))

with open("log_initial.txt", "w") as f:
    f.write(result_initial)

r_init, rdown_init, rup_init = read_fit_result(result_initial)
ravg_init = (rdown_init + rup_init) / 2.


print("Initial result: r = {} -{} +{}".format(r_init, rdown_init, rup_init))

all_params = read_parameters(result_initial)

print(all_params)

errors = {}
systematics = {}


def run_fit(param, params_list):
    print("--- Running fit with frozen parameter {} ---".format(param))

    if len(params_list) > 0:
        set_params = []
        if args.data:
            set_params = ["{0}={1}".format(p, all_params[p]) for p in params_list]
        else:
            set_params = ["{0}=0.0".format(p) for p in params_list]
        cmd_param = "combine -M FitDiagnostics {0} -n freeze_{1} -d workspace.root --setParameters {3} --freezeParameters {2}".format(options, param, ",".join(params_list), ",".join(set_params))
    else:
        raise ValueError()

    print("Running command: {}".format(cmd_param))
    result_param = subprocess.check_output(cmd_param.split(" "))

    with open("log_freeze{}.txt".format(param), "w") as f:
        f.write(result_param)

    r_param, rdown_param, rup_param = read_fit_result(result_param)
    ravg_param = (rdown_param + rup_param) / 2.

    #print(vals_param, errs_param)

    errors[param] = [ravg_param, rdown_param, rup_param]
    if param == "stat" or param == "fullstat":
        diff_unc_down = rdown_param
        diff_unc_up = rup_param
        diff_unc_avg = ravg_param
    else:
        diff_unc_down = np.sqrt(max(rdown_init**2 - rdown_param**2,0))
        diff_unc_up = np.sqrt(max(rup_init**2 - rup_param**2,0))
        diff_unc_avg = (diff_unc_down + diff_unc_up) / 2.
    systematics[param] = [diff_unc_avg*100, diff_unc_down*100, diff_unc_up*100]

    print("--- Result for param {}: avg {}, down {}, up {} percent uncertainty ---".format(param, *systematics[param]))

param_groups = {
    # "none": [],
    "lepton": ["muonsf", "muonsf_reco","electronsf"],
    # "lepton": ["electronsf0", "muonsf0"],
    "JES": ["JuncAbsoluteMPFBias", "JuncAbsoluteSample", "JuncAbsoluteScale", "JuncAbsoluteStat", "JuncFlavorQCD", "JuncFragmentation", "JuncPileUpDataMC", "JuncPileUpPtEC1", "JuncPileUpPtRef", "JuncRelativeBal", "JuncRelativeFSR", "JuncRelativeJEREC1", "JuncRelativePtEC1", "JuncRelativeSample", "JuncRelativeStatEC", "JuncSinglePionECAL", "JuncSinglePionHCAL"],
    # "JES": ["JuncFlavorQCD", "JuncSubTotalPileUp", "JuncSubTotalRelative", "JuncSubTotalPt", "JuncSubTotalScale"],
    "btags": ["btagsf_id", "btagsf_misid"],
    "pileup": ["pileup_nvtx"],
    "trigger": ["triggersfEleMC", "triggersfEleData", "triggersfMuMC", "triggersfMuData"],
    "ME_tt": ["MEfac_tt", "MEren_tt"],
    #"ME_BG": ["MEfac_ST", "MEren_ST", "MEren_DY", "MEren_WJets"],
    "ME_BG": ["MEfac_TW", "MEren_TW", "MEfac_TQ", "MEren_TQ", "MEren_DY", "MEren_WJets"],
    #"ME_DY": ["MEfac_DY", "MEren_DY"],
    #"ME_WJets": ["MEfac_WJets", "MEren_WJets"],
    "PDF": ["PDF", "PDFalphas"],
    "PSscale": ["PSfsr", "PSisr"],
    "PSisr": ["PSisr"],
    "PSfsr": ["PSfsr"],
    "hdamp": ["hdamp"],
    "DrellYanXS": ["DrellYanXS"],
    #"SingleTopXS": ["SingleTopXS"],
    "tWXS": ["tWXS"],
    "SingleToptChXS": ["SingleToptChXS"],
    "top_pt": ["top_pt"],
    "WJetXS": ["WJetXS"],
    "DibosonXS": ["DibosonXS"],
    "NPScale": ["NPScaleEle", "NPScaleMu"],

    "mcstat": [p for p in all_params.keys() if "prop" in p and "NP" not in p],
    #"npstats": [p for p in all_params.keys() if "prop" in p and "NP" in p],
    #"npfull": [p for p in all_params.keys() if "NP" in p],
    "stat": [p for p in all_params.keys() if p != "r"],
    "fullstat": [p for p in all_params.keys() if p != "r" and "prop" not in p]
}

print(param_groups)

for param, params_list in param_groups.items():
    run_fit(param, params_list)

print("----- Results: ------")

for k,v in sorted(systematics.items()):
    print("{:12} : {:.4f} v {:.4f} ^ {:.4f}".format(k,v[0],v[1],v[2]))

with open("systtable.json", "w") as f:
    json.dump({
        "initial_err": [ravg_init, rdown_init, rup_init],
        "errors": errors,
        "systematics": systematics
        }, f, indent=4)

tot_sys = 0.
for sys in systematics.values():
    tot_sys += sys[0]**2

print("Added systematics: {}".format(np.sqrt(tot_sys)))


print(all_params)

