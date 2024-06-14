from plot_utils import *
import argparse

gROOT.SetBatch(True)

parser = argparse.ArgumentParser(description='prefit, postfit plots')

parser.add_argument("config",type=str,action="store",help="combine config")
parser.add_argument("cardDir",type=str,action="store",help="directory with datacards")
parser.add_argument("fitFile",type=str,action="store",help="fitdiagnostics output file with shapes saved")

parser.add_argument("--out",type=str,action="store",default="",help="output file, default is <cardDir>/paperPlot_<fittext>.pdf ")
parser.add_argument("--postfit",action="store_true",help="postfit plot, must have run fit on data")
parser.add_argument("--lumiunc",type=float,action="store",default=0.0,help="additional luminosity unc to draw in shaded bands")
parser.add_argument("--addtext",type=str,action="store",default="Internal Exercise",help="add text like 'preliminary'")
parser.add_argument("--mindraw",type=float,action="store",default=2.0,help="how low to draw on log y axis")


args = parser.parse_args()

addtext = args.addtext 
outfile = args.out
lumiunc = args.lumiunc
cardDir = args.cardDir
fitFile = args.fitFile
mindraw = args.mindraw


fittext = "Prefit"
if args.postfit:
    fittext="Postfit"

fitname = "shapes_prefit"
if args.postfit:
    fitname = "shapes_fit_s"


with open(args.config) as jsonfile:
    config = hjson.load(jsonfile)
processes = [k for k in config["signal_procs"]]+[k for k in config["bg_procs"]]+["data","total"]
categories = [k for k in config["categories"]]

fitDir = os.path.dirname(fitFile)
print(fitDir)
tfile=TFile(fitFile,"READ")

files = {}
for cat in categories:
    files[cat]=TFile(f"{cardDir}/{cat}.root","READ")

if outfile == "":
    outfile = f"{cardDir}/paperPlot_{fittext}.pdf"

histDict, lineDict = combineHists(files,processes,tfile,fitname)
lg = stackLegend(histDict, xmin=0.385, ymin=0.695, xmax=0.69, ymax=0.895, names=procNames, ncol=2)
stackPlot(histDict, lumiunc=lumiunc, dividers=lineDict, legend=lg,cmsadd=addtext, fittext=fittext,  out=outfile,min_log_draw=mindraw)

