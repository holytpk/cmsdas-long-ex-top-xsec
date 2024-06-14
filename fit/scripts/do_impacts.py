import argparse,os
import sys

parser = argparse.ArgumentParser()

parser.add_argument("cardDir",type=str,action="store",help ="directory with datacard")
parser.add_argument("cardName",type=str,action="store", default="all",help ="datacard name")
parser.add_argument("-o","--outName",type=str,action="store", default="",help ="extra string to add to name, to distinguish different fits in the same card directory")

parser.add_argument("-a","--all",action="store_true",help="do the three main steps: initial fit, impacts, plot")
parser.add_argument("-i","--initialFit",action="store_true",help="do just the initial fit")
parser.add_argument("-f","--fits",action="store_true",help="do just the impacts fits")
parser.add_argument("-p","--plotImpacts",action="store_true",help="do just the plot compilation")
parser.add_argument("-r","--retry",action="store", default="",help="retry a specific parameter fit with different options, pray the fit converges")

parser.add_argument("-d","--fitData",action="store_true")

parser.add_argument("-strat","--cminDefaultMinimizerStrategy",type=str, default="1" )
parser.add_argument("-step","--stepSize",type=str,action="store",default="0.01")
parser.add_argument("-tol","--cminDefaultMinimizerTolerance",type=str,action="store",default="0.01")

parser.add_argument("-freeze","--freezeParameters",action="store", type=str, default = "")
parser.add_argument("-set","--setParameters",action="store", type=str, default = "")
parser.add_argument("-ranges","--setParameterRanges",action="store", type=str, default = "r=-2.,2.")
parser.add_argument("-c", "--condor", action="store_true")
parser.add_argument("-v", "--verbosity", default=0)

parser.add_argument("--opts",type=str,default = "--robustFit 1 --cminPreScan --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 --X-rtd MINIMIZER_no_analytic --cminPreFit 1",help="other numerous options piped to combine")

args = parser.parse_args()

# Go to desired output directory
os.chdir(args.cardDir)
logdir = "{}{}_out".format(args.cardName,args.outName)

if not os.path.isdir(logdir):
    os.mkdir(logdir)

os.chdir(logdir)

#various hardcoded options that hopefully never need to change
optsBasic = "--saveNLL --saveFitResult --saveWorkspace -m 120 "
optsCondor = "--sub-opts='getenv = true' --task-name {}{} --job-mode condor ".format(args.cardName,args.outName)

#build commands
cmd_t2w = "combineTool.py -M T2W -i ../{0}.txt -o {2}/workspace_{0}{1}.root".format(args.cardName,args.outName,logdir)
cmd_initial = "combineTool.py -M Impacts -d workspace_{}{}.root --doInitialFit ".format(args.cardName,args.outName)
cmd_impacts = "combineTool.py -M Impacts -d workspace_{}{}.root --doFits ".format(args.cardName,args.outName)
cmd_retry = "combine -M MultiDimFit -n _paramFit_Test_{0} --algo impact --redefineSignalPOIs r -P {0} --floatOtherPOIs 1 --saveInactivePOI 1 -d workspace_{1}{2}.root ".format(args.retry,args.cardName,args.outName)
cmd_merge = "combineTool.py -m 120 -M Impacts -o impacts_{0}{1}.json -d workspace_{0}{1}.root".format(args.cardName,args.outName)
cmd_plot = "plotImpacts.py -i impacts_{0}{1}.json -o impacts_{0}{1}".format(args.cardName,args.outName)
cmd_mv = "mv impacts_{0}{1}.pdf ../".format(args.cardName,args.outName)

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

cmd_initial += optionString
cmd_retry += optionString
cmd_impacts += optionString
if args.condor:
    cmd_impacts += optsCondor
else:
    cmd_impacts += "--parallel 20 "

#only log actual impacts step, The rest sent to terminal
logpipe = "2>&1 | tee log_impacts.log"

cmd_impacts += logpipe

#log all commands separately
fi = open("cmds.log","w")

retry_mode = args.retry != ""

#run cmds
if retry_mode:
    print(cmd_retry)
    os.system(cmd_retry)
    fi.writelines([cmd_retry,"\n\n"])
if args.initialFit or args.all:
    os.system(cmd_t2w)
    fi.writelines([cmd_t2w,"\n"])
    os.system(cmd_initial)
    fi.writelines([cmd_initial,"\n\n"])
if args.all or not (args.initialFit or args.plotImpacts or retry_mode):
    os.system("pwd")
    os.system(cmd_impacts)
    fi.writelines([cmd_impacts,"\n\n"])
    os.system(cmd_merge)
if args.all or args.plotImpacts:
    os.system(cmd_merge)
    fi.writelines([cmd_merge,"\n"])
    os.system(cmd_plot)
    fi.writelines([cmd_plot,"\n\n"])
    os.system(cmd_mv)


fi.close()





