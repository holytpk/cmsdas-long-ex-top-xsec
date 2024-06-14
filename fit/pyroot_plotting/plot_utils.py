import ROOT as r
from ROOT import TH1D, THStack, TTree, gStyle, TFile, gROOT, TCanvas, TPad, TLine, TLegend, gPad, TLatex
import sys
import os
import numpy as np
import hjson
import ctypes as canvas
import sys
import math
from pdb import set_trace
from collections import defaultdict

#define some things for plotting
procNames = { 
    "tt":"t#bar{t}",
    "ST":"Single t",
    "TQ":"t+Q",
    "TW":"t+W",
    "DY":"Z+jets",
    "WJets":"W+jets",
    "VV":"Diboson",
    "NP":"QCD",
    "DYlo":"Drell Yan (low mass)",
    "data":"Data"
}

colors = {}

colors["tt"] = r.kRed-3
colors["TW"] =r.kRed-9
colors["WJets"] = r.kOrange-2
colors["NP"] =r.kCyan-6
colors["DY"] = r.kTeal-7
colors["DYlo"] = r.kTeal-7
colors["VV"] = r.kAzure+4
colors["total"] = r.kBlack
colors["Up"] = r.kRed+1
colors["Down"] = r.kBlue+1

chanNames = { 
    "ee_btag1": "ee (1b)",
    "ee_btag2": "ee (2b)",
    "mm_btag1": "#mu#mu (1b)",
    "mm_btag2": "#mu#mu (2b)",
    "em_btag0": "e#mu (0b)",
    "em_btag1": "e#mu (1b)",
    "em_btag2": "e#mu (2b)",
    "ej_btag1": "e+jets (1b)", 
    "ej_btag2": "e+jets (2b)",
    "mj_btag1": "#mu+jets (1b)",
    "mj_btag2": "#mu+jets (2b)"
}


def combineHists(files,processes,fitfile,fitname):
    # loops over datacard root files from different channels/categories
    # jams them together into one histogram per process
    # gets total uncertainty from the combine fit file
    # returns a dict of TH1s and a dict of channel dividers

    outHistDict = {}

    categories = [k for k in files]

    NBINS = 0

    # Figure out how many bins there are
    for cat in categories:
        testhist = files[cat].Get(processes[0])
        NBINS += testhist.GetNbinsX()

    starterhist = TH1D("dummy","",NBINS,0.,1.*NBINS)
    for i in range(1,NBINS+1):
        starterhist.SetBinContent(i,0.)
        starterhist.SetBinError(i,0.)

    firstbin = 0
    binlabels=[]
    lineDict={}

    for channel in processes:
        outHistDict[channel]= starterhist.Clone(channel)

    for cat,fi in files.items():

        keylist = fi.GetListOfKeys()
        namelist = []

        for tkey in keylist:
            name = tkey.GetName()
            namelist.append(name)

        testChan = processes[0]
        testerhist = files[cat].Get(testChan)
        nx = testerhist.GetNbinsX()

        # bin labels
        for i in range(1,nx+1):
            binstring = ""
            lo_edge = testerhist.GetXaxis().GetBinLowEdge(i)
            hi_edge = testerhist.GetXaxis().GetBinUpEdge(i)-1
            if hi_edge*1.0-lo_edge*1.0 < 0.1:
                binstring = "{} ".format(int(lo_edge))
                if binstring.startswith("1"): binstring=binstring.replace("jets","jet ")
            else:
                binstring = "{}+".format(int(lo_edge),int(hi_edge))
            binlabels.append(binstring)
        lineDict[cat]= firstbin if firstbin!=0 else 0.62

        for proc in processes:

            if proc=="data":
                datathing = fitfile.Get(f"{fitname}/{cat}/{proc}")
                for i in range(1,nx+1):
                    i_nonhist = i-1
                    valx =canvas.c_double(0.)
                    valy =canvas.c_double(0.)
                    datathing.GetPoint(i_nonhist, valx, valy)
                    binc = valy.value
                    bine = binc**0.5
                    totalbin = firstbin+i
                    outHistDict[proc].SetBinContent(totalbin,binc)
                    outHistDict[proc].SetBinError(totalbin,bine)
                    outHistDict[proc].GetXaxis().SetBinLabel(totalbin,binlabels[totalbin-1])
                continue

            if proc not in namelist and proc != "total":
                nomhist = fitfile.Get(f"{fitname}/{cat}/{testChan}")
                for i in range(1,nx+1):
                    totalbin = firstbin+i
                    outHistDict[proc].SetBinContent(totalbin,0.)
                    outHistDict[proc].GetXaxis().SetBinLabel(totalbin,binlabels[totalbin-1])
                continue

            nomhist = fitfile.Get(f"{fitname}/{cat}/{proc}")

            for i in range(1,nx+1):
                binc = nomhist.GetBinContent(i)
                bine = nomhist.GetBinError(i)
                totalbin = firstbin+i
                outHistDict[proc].SetBinContent(totalbin,binc)
                outHistDict[proc].SetBinError(totalbin,bine)
                outHistDict[proc].GetXaxis().SetBinLabel(totalbin,binlabels[totalbin-1])

        lineDict[cat]= firstbin if firstbin!=0 else 0.62
        firstbin+=nx
    lineDict[""] = firstbin

    return outHistDict, lineDict




def cmstext(pos = 'out', cmsadd ="Private Work",relx = 0.0, rely = 0.0,):
    #draw the CMS logo with additional text. option to position in or out of frame. various sloppy hacks added during CWR.
    xpos = r.gPad.GetLeftMargin()+relx
    ypos = 1.-r.gPad.GetTopMargin()+rely

    latex = TLatex(0., 0., 'Z')
    latex.SetNDC(True)
    latex.SetTextFont(63)
    latex.SetTextSize(48)

    latex_additional = TLatex(0., 0., 'Z')
    extra_space = 0.135
    latex_additional.SetNDC(True)
    latex_additional.SetTextFont(54)
    latex_additional.SetTextSize(34)

    if pos.startswith('in'):
        latex.SetTextAlign(13)
        latex.DrawLatex(xpos+0.03, ypos-0.02, 'CMS')
        if len(cmsadd) > 0:
            if pos.endswith("2"):
                latex_additional.SetTextAlign(13)
                latex_additional.DrawLatex(xpos+0.087+extra_space, ypos-0.0258, cmsadd)
            else:
                latex_additional.SetTextAlign(13)
                latex_additional.DrawLatex(xpos+0.04, ypos-40./latex.GetHeight(), cmsadd)
    if pos == 'out':
        latex.SetTextAlign(11)
        latex.DrawLatex(xpos, ypos+0.01, 'CMS')
        if len(cmsadd) > 0:
            latex_additional.SetTextAlign(11)
            latex_additional.DrawLatex(xpos+0.07, ypos+0.01, cmsadd)

def extratext(pos = 'in', word = "", relx = 0.0, rely = 0.0, fontnum = 63, fontsize=40,align=13):
    xpos = r.gPad.GetLeftMargin()  +relx
    ypos = 1.-r.gPad.GetTopMargin() +rely
    latex = TLatex(0., 0., 'Z')
    latex.SetNDC(True)
    latex.SetTextFont(fontnum)
    latex.SetTextSize(fontsize)
    if pos == 'in':
        latex.SetTextAlign(align)
        latex.DrawLatex(xpos+0.03, ypos-0.02, word)
    if pos == 'out':
        latex.SetTextAlign(11)
        latex.DrawLatex(xpos, ypos+0.01, word)

def lumitext(pos = 'out', lumi='1.21 fb^{-1}  (13.6 TeV)'):
    xpos = 1-r.gPad.GetRightMargin()
    ypos = 1.-r.gPad.GetTopMargin()
    latex = TLatex(0., 0., 'Z')
    latex.SetNDC(True)
    latex.SetTextFont(43)
    latex.SetTextSize(32)

    if pos == 'in':
        latex.SetTextAlign(33)
        latex.DrawLatex(xpos+0.03, ypos-0.02, lumi)
    if pos == 'out':
        latex.SetTextAlign(31)
        latex.DrawLatex(xpos, ypos+0.01, lumi)

def stackLegend(histDict,xmin=0.135,ymin=0.625,xmax=0.618,ymax=0.805, ncol=3, names = {}):
    processes = [k for k in histDict]
    processes.sort(key=lambda x: histDict[x].Integral())

    lg = TLegend(xmin, ymin, xmax, ymax)

    legtitle = ""
    lg.SetHeader(legtitle)
    lg.SetFillColor(0)
    lg.SetFillStyle(0)
    lg.SetLineColor(r.kWhite)
    lg.SetLineStyle(0)
    lg.SetBorderSize(0)
    lg.SetShadowColor(0)
    lg.SetTextFont(42)
    lg.SetTextSize(0.0632)

    for proc in processes:
        if proc != "total":
            if proc == "data":
                lg.AddEntry(histDict[proc],"Data","ep")
            else:
                if proc in names:
                    procname = names[proc]
                else:
                    procname = proc
                lg.AddEntry(histDict[proc],procname,"f")

    lg.SetNColumns(ncol)
    
    return lg


def stackPlot(histDict, lumiunc = 0, dividers = {}, legend = "", cmsadd="Private Work", fittext="", out="", min_log_draw = 2):
    # takes in a dictionary of histograms of different processes
    # returns a figure stacking them together in a nice-looking but messy way

    processes = [k for k in histDict]
    processes.sort(key=lambda x: histDict[x].Integral())

    # make a list of things to plot
    # because "THStack" frequently misbehaves, the stacking is achieved by a stupid trick:
    # making a cumulative sum of the histograms in order of largest contribution to smallest
    # and then drawing them on top of each other in reverse order

    cumulative_hist_list = []
    cumulative_proc_list = []

    NBINS = histDict["total"].GetNbinsX()
    totalhist = histDict["total"].Clone("totalhist")

    processes = [x for x in processes if x != "total"]

    for proc in processes:
        if proc == "data": continue
        if len(cumulative_hist_list)!=0:
            histDict[proc].Add(cumulative_hist_list[-1])
        cumulative_hist_list.append(histDict[proc])
        cumulative_proc_list.append(proc)

    # make the canvas, sets aspect ratio of final plot
    canvas = TCanvas()
    canvas.SetCanvasSize(1600, 1200)
    gStyle.SetOptStat(0000)
    

    pad_upper = TPad("pad_upper", "pad_upper", 0.00, 0.41, 1, 1)
    pad_upper.SetBottomMargin(0) 
    pad_upper.SetLeftMargin(0.1)
    pad_upper.SetRightMargin(0.05)
    pad_upper.SetTopMargin(0.15)
    pad_upper.Draw()
    canvas.cd()  

    pad_lower = TPad("pad_lower", "pad_lower", 0.00, 0.00, 1, 0.405)
    pad_lower.SetTopMargin(0.03) 
    pad_lower.SetBottomMargin(0.3)
    pad_lower.SetLeftMargin(0.1)
    pad_lower.SetRightMargin(0.05)
    pad_lower.Draw()

    # this controls how our unc. band will look

    gStyle.SetHatchesSpacing(0.6)

    # work on upper part of canvas 
    pad_upper.cd()
    pad_upper.SetLogy()

    xmin = r.gPad.GetLeftMargin()
    ymax = 1-r.gPad.GetTopMargin()
    xmax = 1-r.gPad.GetRightMargin()
    ymin = r.gPad.GetBottomMargin()
    xwidth = xmax-xmin
    ywidth = ymax-ymin

    #decide what range to draw. works ok for log plots, should be made adjustable in the method
    mc_max = totalhist.GetMaximum()
    data_max = histDict["data"].GetMaximum()
    mcdata_max = max(mc_max,data_max)
    draw_min = min_log_draw
    draw_max = mcdata_max*15

    # this will be the first hist drawn, so it controls the label size
    #!!! does it?
    cumulative_hist_list[-1].SetLabelSize(0.2)

    # set style for data
    histDict["data"].SetLineWidth(2)
    histDict["data"].SetLineColor(r.kBlack)
    histDict["data"].SetMarkerColor(r.kBlack)
    histDict["data"].SetMarkerSize(1.5)
    histDict["data"].SetMarkerStyle(8)


    #draw things, set styles
    for i in range(1,len(processes)):
        hist = cumulative_hist_list[-i]
        proc = cumulative_proc_list[-i]
        hist.Draw("same hist")
        hist.GetYaxis().SetRangeUser(draw_min,draw_max)
        hist.SetFillColor(colors[proc])
        hist.SetMarkerSize(0)
        hist.GetYaxis().SetLabelSize(0.055)
        hist.GetYaxis().SetTitleSize(0.066)
        hist.GetYaxis().SetTitle("Events / bin")
        hist.GetYaxis().SetTitleOffset(0.63)
        hist.GetXaxis().SetLabelSize(0)
        hist.SetLineWidth(0)
        hist.SetLineColor(r.kBlack)

    # draw the uncertainty on the toal counts
    totalhist.SetFillStyle(3354)
    totalhist.SetFillColor(r.kBlack)
    totalhist.Draw("E0E2 same")

    # have to do this
    pad_upper.RedrawAxis()

    # draw data last
    histDict["data"].SetLineWidth(2)
    histDict["data"].SetLineColor(r.kBlack)
    histDict["data"].SetMarkerColor(r.kBlack)
    histDict["data"].SetMarkerSize(1.5)
    histDict["data"].SetMarkerStyle(8)
    histDict["data"].Draw("e1x0 same")

    # add cms text, etc to plot
    cmstext(pos='out',cmsadd = cmsadd,relx=0.0016,rely=0.0027)
    extratext(relx=0.009,rely=0.0015,fontsize=36,fontnum=44,word=fittext)
    lumitext()

    # assemble the channel divider lines
    # you have to make this list because of the way PyROOT works, else they will overwrite one another...
    linelist = []

    for linename,lineval in dividers.items():
        if lineval < 1 : continue
        factor= 1.
        if "j" not in linename:
            factor = 0.08
        if "em" in linename:
            factor = 0.25

        newline = TLine(lineval,draw_min,lineval,factor*draw_max)
        linelist.append(newline)
        newline.Draw()
        newline.SetLineWidth(1)

    # draw legend if passed to method
    if legend != "":
        legend.Draw()

    # time for the lower pad of ratios
    pad_lower.cd()
    # min and max draw range for ratio, currently hard-coded numbers
    draw_min = 0.5
    draw_max = 1.5

    ratio_hist = histDict["data"].Clone("ratio_hist")
    ratio_hist.Sumw2()
    ratio_hist.Divide(totalhist)

    # set the uncertainties properly, lumi unc added here if provided
    for i in range(1,ratio_hist.GetNbinsX()+1):
        ratio_hist.SetBinError(i,histDict["data"].GetBinError(i)/totalhist.GetBinContent(i))

    # we will use a separate hist to draw uncertainties because ROOT plotting sucks
    unc_hist = TH1D("uncs","",NBINS,0.,1.*NBINS)
    for i in range(1,NBINS+1):
        unc_hist.SetBinContent(i,0.)
        unc_hist.SetBinError(i,0.)

    for i in range(1,NBINS+1):
        unc_hist.SetBinContent(i,1.)
        unc_hist.SetBinError(i,totalhist.GetBinError(i)/totalhist.GetBinContent(i))
        unc_hist.SetBinError(i,np.sqrt(unc_hist.GetBinError(i)**2+lumiunc**2))

    # set ratio hist style
    ratio_hist.SetTitle("")
    ratio_hist.GetYaxis().SetTitleSize(0.1)
    ratio_hist.GetYaxis().SetTitleOffset(0.47)
    ratio_hist.GetYaxis().SetRangeUser(draw_min,draw_max)
    ratio_hist.GetYaxis().SetLabelSize(0.086)

    if "post" in fittext.lower():
        ratio_hist.GetYaxis().SetTitle("Data / fit")
    else:
        ratio_hist.GetYaxis().SetTitle("Data / pred.")
    ratio_hist.GetYaxis().SetNdivisions(-40204)

    ratio_hist.GetXaxis().SetLabelSize(0.12)
    ratio_hist.GetXaxis().LabelsOption("v")
    ratio_hist.GetXaxis().SetTitleSize(0.11)
    ratio_hist.GetXaxis().SetTitle("Number of jets")
    ratio_hist.GetXaxis().SetTitleOffset(1.2)
    ratio_hist.Draw("e1x0 same")

    # choose the hatching of the error, surprisingly contentious!
    unc_hist.SetFillStyle(3354)
    unc_hist.SetFillColor(r.kBlack)
    unc_hist.SetLineColor(r.kBlack)
    unc_hist.Draw("LE0E2 same")

    # just a trick to draw a horizontal line at zero
    zero_hist = unc_hist.Clone()
    zero_hist.SetFillStyle(0)
    zero_hist.Draw("same hist")

    # now add the channel dividers
    for linename,lineval in dividers.items():
        linevals = [math.floor(val) for val in dividers.values()]
        linevals.sort()
        lineindex = linevals.index(math.floor(lineval))
        if lineindex == len(linevals)-1: continue
        nextval = linevals[lineindex+1]
        midval = float((lineval+nextval)/2)
        linelabel = r.TLatex(0.5, 0.5, 'Z')
        linelabel.SetTextFont(43)
        linelabel.SetTextSize(26.5)
        linelabel.SetTextAlign(23)
        linelabel.SetTextAngle(0)
        linelabel.DrawLatex(midval, draw_max*.98, chanNames[linename])

        if lineval < 1 : continue
        newline = TLine(lineval,draw_min,lineval,draw_max)
        linelist.append(newline)
        newline.Draw()

    # redraw axes and we are done
    pad_lower.RedrawAxis()

    if out != "":
        canvas.SaveAs(out)
        
    return canvas

def combineSysts(files,processes):
    # loops over datacard root files from different channels/categories
    # jams them together into one histogram per process
    # returns a dict of TH1s and a dict of channel dividers
    # and is generally a mess but works
    outHistDict = {}
    outfi = TFile("systs.root","RECREATE")

    sysDict = {}
    for proc in processes:
        sysDict[proc] = []

    categories = [k for k in files]
    fitname = "shapes_prefit"

    NBINS = 0

    # Figure out how many bins there are
    for cat in categories:
        testhist = files[cat].Get(processes[0])
        NBINS += testhist.GetNbinsX()

    starterhist = TH1D("dummy","",NBINS,0.,1.*NBINS)
    for i in range(1,NBINS+1):
        starterhist.SetBinContent(i,0.)
        starterhist.SetBinError(i,0.)

    firstbin = 0
    binlabels=[]
    lineDict={}

    for channel in processes:
        outHistDict[channel]= starterhist.Clone(channel)

    for cat,fi in files.items():

        keylist = fi.GetListOfKeys()
        namelist = []

        for tkey in keylist:
            name = tkey.GetName()
            classname = tkey.GetClassName()
            if classname!="TH1D": continue
            namelist.append(name)
            splitname = name.split("_",1)
            if len(splitname) > 1 and name.endswith("Up"):
                sysname = splitname[1][:-2]
                if not sysname in sysDict[splitname[0]]:
                    sysDict[splitname[0]].append(sysname)

        testChan = processes[0]
        testerhist = files[cat].Get(testChan)
        nx = testerhist.GetNbinsX()

        # bin labels
        for i in range(1,nx+1):
            binstring = ""
            lo_edge = testerhist.GetXaxis().GetBinLowEdge(i)
            hi_edge = testerhist.GetXaxis().GetBinUpEdge(i)-1
            if hi_edge*1.0-lo_edge*1.0 < 0.1:
                binstring = "{} ".format(int(lo_edge))
                if binstring.startswith("1"): binstring=binstring.replace("jets","jet ")
            else:
                binstring = "{}+".format(int(lo_edge),int(hi_edge))
            binlabels.append(binstring)
        lineDict[cat]= firstbin if firstbin!=0 else 0.62

        for proc in processes:
            if proc == "total" or proc=="data":
                continue
            if proc not in namelist: continue

            nomhist = fi.Get(f"{proc}").Clone("{}_{}_nom".format(proc,cat))

            for i in range(1,nx+1):
                binc = nomhist.GetBinContent(i)
                bine = nomhist.GetBinError(i)
                totalbin = firstbin+i
                outHistDict[proc].SetBinContent(totalbin,binc)
                outHistDict[proc].SetBinError(totalbin,bine)
                outHistDict[proc].GetXaxis().SetBinLabel(totalbin,binlabels[totalbin-1])

            for hist in sysDict[proc]:
                sysname = hist
                for ud in ["Up","Down"]:
                    histname = "{}_{}{}".format(proc,hist,ud)
                    histabs = fi.Get(histname).Clone("{}_{}{}_abs".format(proc,hist,ud))
                    histrel = fi.Get(histname).Clone("{}_{}{}_rel".format(proc,hist,ud))
                    histrel.Add(nomhist,-1.0)
                    histrel.Divide(nomhist)
                    if histname not in outHistDict.keys():
                        outHistDict[histname]=starterhist.Clone(histname)
                        outHistDict[histname].SetTitle("{} {}".format(procNames[proc],sysname))
                        outHistDict[histname].GetYaxis().SetTitle("relative effect")
                        outHistDict[histname].SetDirectory(outfi)
                        outHistDict[histname+"Abs"]=starterhist.Clone(histname+"Abs")
                        outHistDict[histname+"Abs"].SetTitle("{} {}".format(procNames[proc],sysname))
                        outHistDict[histname+"Abs"].GetYaxis().SetTitle("bin content")
                        outHistDict[histname+"Abs"].SetDirectory(outfi)
                    for i in range(1,histrel.GetNbinsX() +1):
                        binc = histrel.GetBinContent(i)
                        totalbin = firstbin+i
                        outHistDict[histname].SetBinContent(totalbin,binc)
                        outHistDict[histname].GetXaxis().SetBinLabel(totalbin,binlabels[totalbin-1])
                        binc = histabs.GetBinContent(i)
                        outHistDict[histname+"Abs"].SetBinContent(totalbin,binc)
                        outHistDict[histname+"Abs"].GetXaxis().SetBinLabel(totalbin,binlabels[totalbin-1]) 

        lineDict[cat]= firstbin if firstbin!=0 else 0.62
        firstbin+=nx
    lineDict[""] = firstbin

    return sysDict, outHistDict, lineDict

def sysPlots(sysDict, histDict,outdir, dividers = {}, filetype = "png"):

    gStyle.SetOptStat(0000)
    os.makedirs(outdir,exist_ok=True)

    for proc,sysList in sysDict.items():
        if proc == "data" or proc == "total": continue
        for sys in sysList:
            # here you could load a dict of more "readable" names but for now I don't
            sysname = sys
            #make canvas
            canvas = TCanvas()
            canvas.SetCanvasSize(1600, 1200)

            pad_upper = TPad("pad_upper", "pad_upper", 0.00, 0.505, 1, 1)
            pad_upper.SetBottomMargin(0) 
            pad_upper.SetLeftMargin(0.1)
            pad_upper.SetRightMargin(0.05)
            pad_upper.SetTopMargin(0.15)
            pad_upper.SetGridy()
            pad_upper.Draw()
            canvas.cd()  
            pad_lower = TPad("pad_lower", "pad_lower", 0.00, 0.00, 1, 0.50)
            pad_lower.SetTopMargin(0.02) 
            pad_lower.SetBottomMargin(0.2)
            pad_lower.SetLeftMargin(0.1)
            pad_lower.SetRightMargin(0.05)
            pad_lower.Draw()

            # draw upper panel

            pad_upper.cd()
            pad_upper.SetLogy()
            r.gStyle.SetTitleH(0.1)

            hist_nominal = histDict[proc]
            hist_nominal.SetTitle("{} {}".format(procNames[proc],sysname))
            hist_nominal.SetLineColor(r.kBlack)
            hist_nominal.SetLineWidth(2)

            hist_nominal.SetTitleSize(0.2,"t")

            hist_abs_up = histDict["{}_{}{}".format(proc,sys,"UpAbs")]
            hist_abs_up.SetLineWidth(2)
            hist_abs_up.GetYaxis().SetLabelSize(0.055)
            hist_abs_up.GetYaxis().SetTitleSize(0.085)
            hist_abs_up.GetXaxis().SetLabelSize(0.055)
            hist_abs_up.GetXaxis().SetTitleSize(0.055)
            hist_abs_up.GetYaxis().SetTitleOffset(0.55)
            hist_abs_up.SetTitleSize(0.2)

            hist_abs_dn = histDict["{}_{}{}".format(proc,sys,"DownAbs")]
            hist_abs_dn.SetLineWidth(2)

            hist_abs_up.SetLineColor(colors["Up"])
            hist_abs_dn.SetLineColor(colors["Down"])
            hist_abs_up.Draw("hist same")
            hist_abs_dn.Draw("hist same")
            hist_nominal.Draw("hist same")

            # draw lower panel

            pad_lower.cd()

            hist_ratio_up = histDict["{}_{}{}".format(proc,sys,"Up")]
            hist_ratio_dn = histDict["{}_{}{}".format(proc,sys,"Down")]
            minval = min(hist_ratio_up.GetMinimum(),hist_ratio_dn.GetMinimum())
            draw_min = minval-0.23*abs(minval)
            maxval = max(hist_ratio_up.GetMaximum(),hist_ratio_dn.GetMaximum())
            draw_max = maxval+0.3*abs(maxval)

            # labels come from this hist
            hist_ratio_up.SetLineColor(colors["Up"])
            hist_ratio_up.SetTitle("")
            hist_ratio_up.GetYaxis().SetLabelSize(0.055)
            hist_ratio_up.GetYaxis().SetTitleSize(0.075)
            hist_ratio_up.SetLineWidth(2)
            hist_ratio_up.GetYaxis().SetRangeUser(minval-0.23*abs(minval),maxval+0.3*abs(maxval))
            hist_ratio_up.GetXaxis().SetLabelSize(0.065)
            hist_ratio_up.GetXaxis().SetTitleSize(0.075)
            hist_ratio_up.GetXaxis().SetTitle("Jets")
            hist_ratio_up.GetYaxis().SetTitleOffset(0.65)
            hist_ratio_up.GetXaxis().LabelsOption("v")

            hist_ratio_dn.SetLineWidth(2)
            hist_ratio_dn.SetLineColor(colors["Down"])

            hist_ratio_up.Draw("hist")
            hist_ratio_dn.Draw("hist same")

            #vertical lines

            linelist=[]

            for linename,lineval in dividers.items():
                linevals = [math.floor(val) for val in dividers.values()]
                linevals.sort()
                lineindex = linevals.index(math.floor(lineval))
                if lineindex == len(linevals)-1: continue
                nextval = linevals[lineindex+1]
                midval = float((lineval+nextval)/2)
                linelabel = r.TLatex(0.5, 0.5, 'Z')
                linelabel.SetTextFont(43)
                linelabel.SetTextSize(26.5)
                linelabel.SetTextAlign(23)
                linelabel.SetTextAngle(0)
                linelabel.DrawLatex(midval, draw_max*.98, chanNames[linename])

                if lineval < 1 : continue
                newline = TLine(lineval,draw_min,lineval,draw_max)
                linelist.append(newline)
                newline.Draw()

            #horizontal line at zero

            zeroline = TLine(0.,0.,hist_ratio_up.GetNbinsX()*1.,0.)
            zeroline.SetLineStyle(3)
            zeroline.SetLineWidth(2)
            zeroline.Draw()

            canvas.SaveAs("{}/{}_{}.{}".format(outdir,sys,proc,filetype))
