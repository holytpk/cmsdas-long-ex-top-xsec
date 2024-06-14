import argparse,os
import sys

parser = argparse.ArgumentParser()

parser.add_argument("cardDir",type=str,action="store",help ="directory with datacard")
parser.add_argument("cardName",type=str,action="store", default="all",help ="datacard name")
parser.add_argument("-o","--outName",type=str,action="store", default="",help ="extra string to add to name, to distinguish different fits in the same card directory")

parser.add_argument("-d","--fitData",action="store_true")

parser.add_argument("-strat","--cminDefaultMinimizerStrategy",type=str, default="1" )
parser.add_argument("-step","--stepSize",type=str,action="store",default="0.01")
parser.add_argument("-tol","--cminDefaultMinimizerTolerance",type=str,action="store",default="0.01")

parser.add_argument("-freeze","--freezeParameters",action="store", type=str, default = "")
parser.add_argument("-set","--setParameters",action="store", type=str, default = "")
parser.add_argument("-ranges","--setParameterRanges",action="store", type=str, default = "r=-2.,2.")
parser.add_argument("-c", "--condor", action="store_true")
parser.add_argument("-v", "--verbosity", default=0)

parser.add_argument("--nosave",action="store_true",help="don't save correlation matrix, plots, etc in combine output (faster fit)")

parser.add_argument("--opts",type=str,default = "--robustFit 1 --cminPreScan --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 --X-rtd MINIMIZER_no_analytic --cminPreFit 1",help="other numerous options piped to combine")

args = parser.parse_args()

# Go to desired output directory
os.chdir(args.cardDir)
logdir = "{}{}_out".format(args.cardName,args.outName)

if not os.path.isdir(logdir):
    os.mkdir(logdir)

os.chdir(logdir)

#various hardcoded options that hopefully never need to change
optsBasic = "--saveNLL -m 120 --skipBOnlyFit "
optsCondor = "--sub-opts='getenv = true' --task-name {}{} --job-mode condor ".format(args.cardName,args.outName)
optsSave = "--plots --saveNormalizations --saveShapes --saveWithUncertainties "

#build commands
cmd_t2w = "combineTool.py -M T2W -i ../{0}.txt -o {2}/workspace_{0}{1}.root".format(args.cardName,args.outName,logdir)
cmd_fit = "combine -M FitDiagnostics -d workspace_{}{}.root ".format(args.cardName,args.outName)

optionString = ""

if not args.fitData:
    optionString += "-t -1 --expectSignal=1 " 

if args.freezeParameters != "":
    optionString += "--freezeParameters {} ".format(args.freezeParameters)
if args.setParameters != "":
    optionString += "--setParameters {} ".format(args.setParameters)
optionString += "--setParameterRanges {} ".format(args.setParameterRanges)

optionString += optsBasic
optionString += "{} ".format(args.opts)
optionString += "--cminDefaultMinimizerStrategy {} ".format(args.cminDefaultMinimizerStrategy)
optionString += "--cminDefaultMinimizerTolerance {} ".format(args.cminDefaultMinimizerTolerance)
optionString += "--stepSize {} ".format(args.stepSize)
optionString += "-v {} ".format(args.verbosity)
if not args.nosave:
    optionString+=optsSave

cmd_fit += optionString

if not os.path.exists("workspace_{0}{1}.root".format(args.cardName,args.outName)):
    os.system(cmd_t2w)


print(cmd_fit)
os.system(cmd_fit)






