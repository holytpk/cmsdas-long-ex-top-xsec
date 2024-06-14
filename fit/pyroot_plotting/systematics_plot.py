from plot_utils import *
import argparse

gROOT.SetBatch(True)

parser = argparse.ArgumentParser(description='prefit, postfit plots')

parser.add_argument("config",type=str,action="store",help="combine config")
parser.add_argument("cardDir",type=str,action="store",help="directory with datacards")
parser.add_argument("--outDir",type=str,action="store",default="systematic_plots/",help="directory to save outputs")

args = parser.parse_args()

cardDir = args.cardDir
outDir = args.outDir

with open(args.config) as jsonfile:
    config = hjson.load(jsonfile)
processes = [k for k in config["signal_procs"]]+[k for k in config["bg_procs"]]+["data","total"]
categories = [k for k in config["categories"]]

files = {}
for cat in categories:
    files[cat]=TFile(f"{cardDir}/{cat}.root","READ")

sysDict, histDict, lineDict = combineSysts(files,processes)

sysPlots(sysDict, histDict, outDir, lineDict)

