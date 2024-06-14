#!/usr/bin/env python

# //--------------------------------------------
#-- Original script by Evan Ranken (in the context of TOP-19-008), adapted from the default combine script (https://github.com/cms-analysis/CombineHarvester/blob/master/CombineTools/scripts/plotImpacts.py)
#-- Slightly adapted by Nicolas Tonon for wider use within TOP PAG

#-- Script and instructions available in TOP PAG combine repository: https://gitlab.cern.ch/CMS-TOPPAG/Datacards
# //--------------------------------------------

# sample command for Evan's setup:
# python customImpacts.py -i out/RUN3_170223_newlumi_fakerate/alldata_out/impacts_alldata.json -ia out/RUN3_170223_newlumi_fakerate/all_out/impacts_all.json -o out/customImpacts -t names.json -n 20

import ROOT
import math
import json
import argparse
import CombineHarvester.CombineTools.plotting as plot
import CombineHarvester.CombineTools.combine.rounding as rounding

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)
ROOT.TH1.AddDirectory(0)

#-- Arguments
# //--------------------------------------------
parser = argparse.ArgumentParser()
#MAIN ARGS
parser.add_argument('--input' ,'-i', help='Input json file for observed results')
parser.add_argument('--asimov-input','-ia', help='Input json file for asimov/expected results')
parser.add_argument('--output', '-o', help='Name of output file to create (without .pdf extension)')
parser.add_argument('--translate', '-t', help='json file for remapping of parameter names')
parser.add_argument('--POI', default=None, help='Specify a POI to draw') #E.g. 'r'
parser.add_argument('--blind', action='store_true', help='Do not print best fit signal strength')
#OTHERS
parser.add_argument("--onlyfirstpage", metavar="onlyfirstpage", default=True, help="Only display the first page", nargs='?', const=1)
parser.add_argument('--units', default=None, help='Add units to the best-fit parameter value')
parser.add_argument('--per-page','-n', type=int, default=10, help='Number of parameters to show per page')
parser.add_argument('--cms-label', default='', help='Label next to the CMS logo')
parser.add_argument("--pullDef",  default=None, help="Choose the definition of the pull, see HiggsAnalysis/CombinedLimit/python/calculate_pulls.py for options")
args = parser.parse_args()

# //--------------------------------------------
# //--------------------------------------------

externalPullDef = False
if args.pullDef is not None:
    externalPullDef = True
    import HiggsAnalysis.CombinedLimit.calculate_pulls as CP

#-- Returns translated name if key found in json dict, else returns arg
def Translate(name, ndict):
    if name in ndict:
        name = ndict[name]
    if name in ndict["sys_names_root"]:
        name = ndict["sys_names_root"][name]
    if name.startswith("prop") or name.startswith("Junc"):
        for key in ndict["replace"]:
            name = name.replace(key,ndict["replace"][key])
    return name

#-- Rounding nominal, uperror, downerror
def GetRounded(nom, e_hi, e_lo):
    if e_hi < 0.0:
        e_hi = 0.0
    if e_lo < 0.0:
        e_lo = 0.0
    rounded = rounding.PDGRoundAsym(nom,
            e_hi if e_hi != 0.0 else 1.0,
            e_lo if e_lo != 0.0 else 1.0)
    s_nom = rounding.downgradePrec(rounded[0],rounded[2])
    s_hi = rounding.downgradePrec(rounded[1][0][0],rounded[2]) if e_hi != 0.0 else '0'
    s_lo = rounding.downgradePrec(rounded[1][0][1],rounded[2]) if e_lo != 0.0 else '0'
    return (s_nom, s_hi, s_lo)

#Load json output of combineTool.py -M Impacts (observed)
data = {}
with open(args.input) as jsonfile:
    data = json.load(jsonfile)

#Load json output of combineTool.py -M Impacts (expected/asimov)
asidata = {}
with open(args.asimov_input) as asifile:
    asidata = json.load(asifile)

#Set global plotting style
plot.ModTDRStyle(l=0.4, b=0.10, width=700)

#Assume the first POI is the one to plot
POIs = [ele['name'] for ele in data['POIs']]
POI = POIs[0]
if args.POI:
    POI = args.POI
for ele in data['POIs']:
    if ele['name'] == POI:
        POI_info = ele
        break
POI_fit = POI_info['fit']

#json dictionary to translate parameter names
translate = {}
if args.translate is not None:
    with open(args.translate) as jsonfile:
        translate = json.load(jsonfile)
poi_translated = Translate(POI,translate) #Get translated POI name (if available)

#Sort parameters by largest absolute impact on this POI
data['params'].sort(key=lambda x: abs(x['impact_%s' % POI]), reverse=True)

#Set the number of parameters per page (show) and the number of pages (n)
show = args.per_page
n = int(math.ceil(float(len(data['params'])) / float(show)))
if args.onlyfirstpage == True: n = 1

colors = {
    'Gaussian': 1,
    'Poisson': 8,
    'AsymmetricGaussian': 9,
    'Unconstrained': 1,
    'Unrecognised': 2
}
color_hists = {}
color_group_hists = {}
for name, col in colors.iteritems():
    color_hists[name] = ROOT.TH1F()
    plot.Set(color_hists[name], FillColor=col, Title=name)

seen_types = set()


# //--------------------------------------------
# //--------------------------------------------

for page in xrange(n):
    canv = ROOT.TCanvas(args.output, args.output)
    n_params = len(data['params'][show * page:show * (page + 1)])
    pdata = data['params'][show * page:show * (page + 1)]
    asipdata = asidata["params"]
    print '>> Doing page %i, have %i parameters' % (page, n_params)

    #-- Set top/bottom margins
    ROOT.gStyle.SetPadBottomMargin(0.1)
    ROOT.gStyle.SetPadTopMargin(0.10 if args.blind else 0.15) #If not displaying best-fit poi, reduce top margin

    #-- Set bkg color of every other line to gray
    boxes = []
    for i in xrange(n_params):
        y1 = ROOT.gStyle.GetPadBottomMargin()
        y2 = 1. - ROOT.gStyle.GetPadTopMargin()
        h = (y2 - y1) / float(n_params)
        y1 = y1 + float(i) * h
        y2 = y1 + h
        x1 = 0 #Start from canvas' left border
        x2 = 0.965 #Fine-tuned such that gray bkg (for every other param) stops at vertical black line #or 1 to reach right-end of canvas
        box = ROOT.TPaveText(x1, y1, x2, y2, 'NDC')
        plot.Set(box, TextSize=0.02, BorderSize=0, FillColor=0, TextAlign=12, Margin=0.005)
        if i % 2 == 0: box.SetFillColor(18) #Gray
        box.Draw()
        boxes.append(box)

    #-- Pad style
    pads = plot.MultiRatioSplitColumns([0.7], [0.], [0.])
    pads[0].SetGrid(1, 0)
    pads[0].SetTickx(1)
    pads[1].SetGrid(1, 0)
    pads[1].SetTickx(1)

    #-- Define main objects
    h_pulls = ROOT.TH2F("pulls", "pulls", 6, -2.9, 2.9, n_params, 0, n_params) #TH2F defining the frame, bkg style, etc.
    g_pulls = ROOT.TGraphAsymmErrors(n_params) #Pulls obs
    g_pullsA = ROOT.TGraphAsymmErrors(n_params) #Pulls exp
    g_impacts_hi = ROOT.TGraphAsymmErrors(n_params) #Obs impact
    g_impacts_lo = ROOT.TGraphAsymmErrors(n_params) #Obs impact
    g_impactsA_hi = ROOT.TGraphAsymmErrors(n_params) #Exp impact
    g_impactsA_lo = ROOT.TGraphAsymmErrors(n_params) #Exp impact
    g_check = ROOT.TGraphAsymmErrors()

    #-- Read values, fill pull histogram
    max_impact = 0.
    text_entries = []
    redo_boxes = []
    for p in xrange(n_params):
        i = n_params - (p + 1)
        thisname = pdata[p]['name']
        asipnum = -1
        asipnumfound = False
        while not asipnumfound:
            asipnum +=1
            if asipnum >= len(asipdata)-1:
                print "too many vars in asimov data ", thisname,asipnum
                break
                # raise Exception('ERROR: the parameter with name {}'.format(thisname)
                #        +' has no counterpart in the expected results!')
            if asipdata[asipnum]["name"]==thisname:
     			asipnumfound = True
        pre = pdata[p]['prefit']
        fit = pdata[p]['fit']
        preA = asipdata[asipnum]['prefit']
        fitA = asipdata[asipnum]['fit']
        tp = pdata[p]['type']
        seen_types.add(tp)

        if pdata[p]['type'] != 'Unconstrained':
            pre_err_hi = (pre[2] - pre[1])
            pre_err_lo = (pre[1] - pre[0])

            if externalPullDef:
                fit_err_hi = (fit[2] - fit[1])
                fit_err_lo = (fit[1] - fit[0])
                pull, pull_hi, pull_lo = CP.returnPullAsym(args.pullDef,fit[1],pre[1],fit_err_hi,pre_err_hi,fit_err_lo,pre_err_lo)
            else:
                pull = fit[1] - pre[1]
                pull = (pull/pre_err_hi) if pull >= 0 else (pull/pre_err_lo)
                pull_hi = fit[2] - pre[1]
                pull_hi = (pull_hi/pre_err_hi) if pull_hi >= 0 else (pull_hi/pre_err_lo)
                pull_hi = pull_hi - pull
                pull_lo = fit[0] - pre[1]
                pull_lo = (pull_lo/pre_err_hi) if pull_lo >= 0 else (pull_lo/pre_err_lo)
                pull_lo =  pull - pull_lo

            if thisname == "top_pt":
                pull_lo = pull

            g_pulls.SetPoint(i, pull, float(i) + 0.5)
            g_pulls.SetPointError(i, pull_lo, pull_hi, 0., 0.)

            pre_err_hi = (preA[2] - preA[1])
            pre_err_lo = (preA[1] - preA[0])

            if externalPullDef:
                fit_err_hi = (fit[2] - fit[1])
                fit_err_lo = (fit[1] - fit[0])
                pull, pull_hi, Apull_lo = CP.returnPullAsym(args.pullDef,fit[1],pre[1],fit_err_hi,pre_err_hi,fit_err_lo,pre_err_lo)
            else:
                pull = fitA[1] - preA[1]
                pull = (pull/pre_err_hi) if pull >= 0 else (pull/pre_err_lo)
                pull_hi = fitA[2] - preA[1]
                pull_hi = (pull_hi/pre_err_hi) if pull_hi >= 0 else (pull_hi/pre_err_lo)
                pull_hi = pull_hi - pull
                pull_lo = fitA[0] - preA[1]
                pull_lo = (pull_lo/pre_err_hi) if pull_lo >= 0 else (pull_lo/pre_err_lo)
                pull_lo =  pull - pull_lo

            if thisname == "top_pt":
                pull_lo = pull

            g_pullsA.SetPoint(i, pull, float(i) + 0.5)
            g_pullsA.SetPointError(
                i, pull_lo, pull_hi, 0.5, 0.5)
        else: #Unconstrained, hide point
            g_pulls.SetPoint(i, 0., 9999.)
            y1 = ROOT.gStyle.GetPadBottomMargin()
            y2 = 1. - ROOT.gStyle.GetPadTopMargin()
            x1 = ROOT.gStyle.GetPadLeftMargin()
            h = (y2 - y1) / float(n_params)
            y1 = y1 + ((float(i)+0.5) * h)
            x1 = x1 + (1 - pads[0].GetRightMargin() -x1)/2.
            # s_nom, s_hi, s_lo = GetRounded(fit[1], fit[2] - fit[1], fit[1] - fit[0])
            # this part stores text/shapeU vars and sets the digits to show
            s_nom, s_hi, s_lo = "{:.3f}".format(fit[1]), "{:.3f}".format( fit[2] - fit[1]), "{:.3f}".format( fit[1] - fit[0])
            if thisname == "btagsf_id":
                s_nom = "{:.3f}".format(fit[1]*0.1+1.)
                s_hi = "{:.3f}".format( (fit[2] - fit[1])*0.1)
                s_lo = "{:.3f}".format( (fit[1] - fit[0])*0.1)
            text_entries.append((x1, y1, '%s^{#plus%s}_{#minus%s}' % (s_nom, s_hi, s_lo)))
            redo_boxes.append(i)

        g_impacts_hi.SetPoint(i, 0, float(i) + 0.5)
        g_impacts_lo.SetPoint(i, 0, float(i) + 0.5)
        g_impactsA_hi.SetPoint(i, 0, float(i) + 0.5)
        g_impactsA_lo.SetPoint(i, 0, float(i) + 0.5)
        imp = pdata[p][POI]
        impA = asipdata[asipnum][POI]
        if imp[2]-imp[1]>0:
            g_impacts_hi.SetPointError(i, 0, imp[2] - imp[1], 0.0, 0.0)
            g_impacts_lo.SetPointError(i, imp[1] - imp[0], 0, 0.0, 0.0)
        else:
            g_impacts_hi.SetPointError(i,  imp[1] - imp[2],0, 0.0, 0.0)
            g_impacts_lo.SetPointError(i, 0, imp[0] - imp[1],  0.0, 0.0)
        max_impact = max(max_impact, abs(imp[1] - imp[0]), abs(imp[2] - imp[1]))
        max_impact = max(max_impact, abs(impA[1] - impA[0]), abs(impA[2] - impA[1]))
        col = colors.get(tp, 2)
        if impA[2]-impA[1]>0:
            g_impactsA_hi.SetPointError(i, 0, impA[2] - impA[1], 0.49, 0.49)
            g_impactsA_lo.SetPointError(i, impA[1] - impA[0], 0, 0.49, 0.49)
        else:
            g_impactsA_hi.SetPointError(i,  impA[1] - impA[2],0, 0.49, 0.49)
            g_impactsA_lo.SetPointError(i, 0, impA[0] - impA[1],  0.49, 0.49)

        thisname = Translate(thisname,translate)
        h_pulls.GetYaxis().SetBinLabel(i + 1, ('#color[%i]{%s}'% (col, thisname)))

    #-- Style and draw the pulls histo
    if externalPullDef:
        plot.Set(h_pulls.GetXaxis(), TitleSize=0.04, LabelSize=0.03, Title=CP.returnTitle(args.pullDef))
    else:
        plot.Set(h_pulls.GetXaxis(), TitleSize=0.04, LabelSize=0.03, Title='(#hat{#theta}-#theta_{0})/#Delta#theta')
    plot.Set(h_pulls.GetYaxis(), LabelSize=0.04, TickLength=0.0)
    h_pulls.GetYaxis().SetLabelFont(42)
    h_pulls.GetYaxis().LabelsOption('v')
    h_pulls.Draw()

    for i in redo_boxes:
        newbox = boxes[i].Clone()
        newbox.Clear()
        newbox.SetY1(newbox.GetY1()+0.005)
        newbox.SetY2(newbox.GetY2()-0.005)
        newbox.SetX1(ROOT.gStyle.GetPadLeftMargin()+0.002)
        newbox.SetX2(0.7-0.001)
        newbox.Draw()
        boxes.append(newbox)
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    # this part draws the shapeU/text constraints and you can change the size here
    latex.SetTextSize(0.024)
    latex.SetTextAlign(22)
    for entry in text_entries:
        latex.DrawLatex(*entry)

    #-- Draw impacts histogram
    pads[1].cd()
    if max_impact == 0.: max_impact = 1E-6  # otherwise the plotting gets screwed up
    if max_impact > .9: max_impact = .5
    h_impacts = ROOT.TH2F(
        "impacts", "impacts", 6, -max_impact * 1.06, max_impact * 1.06, n_params, 0, n_params)
    plot.Set(h_impacts.GetXaxis(), LabelSize=0.03, TitleSize=0.04, Ndivisions=505, Title=
     # '#Delta#mu')
     '#Delta#hat{%s}' % poi_translated)
    plot.Set(h_impacts.GetYaxis(), LabelSize=0, TickLength=0.0)
    h_impacts.Draw()

    #-- Draw arXiv entry (if needed)
    draw_arxiv = False
    if draw_arxiv == True:
        pads[0].cd()
        xpos = ROOT.gPad.GetLeftMargin()
        xposR = 1-ROOT.gPad.GetRightMargin()
        ypos = 1.-ROOT.gPad.GetTopMargin()+0.01
        lx = ROOT.TLatex(0., 0., 'Z')
        lx.SetNDC(True)
        lx.SetTextAlign(13)
        lx.SetTextFont(42)
        lx.SetTextSize(0.04)
        lx.SetTextColor(16)
        lx.DrawLatex(0.014, .99, 'arXiv:xxx')

    #-- Draw the pulls graph
    pads[0].cd()
    alpha = 0.7
    plot.Set(g_pulls, MarkerSize=0.8, LineWidth=2)
    g_pullsA.SetLineWidth(0)
    g_pullsA.SetFillColor(plot.CreateTransparentColor(15, alpha))
    g_pullsA.Draw('2SAME')
    g_pulls.Draw('PSAME')

    #-- Draw impacts graphs
    pads[1].cd()
    alpha = 0.7
    g_impactsA_hi.SetFillColor(plot.CreateTransparentColor(46, alpha))
    g_impactsA_hi.SetLineWidth(0)
    g_impactsA_lo.SetLineWidth(0)
    g_impactsA_lo.SetFillColor(plot.CreateTransparentColor(38, alpha))
    g_impacts_hi.SetLineWidth(2)
    g_impacts_lo.SetLineWidth(2)
    g_impacts_hi.SetLineStyle(1)
    g_impactsA_hi.SetMarkerSize(0)
    g_impacts_hi.SetLineColor(ROOT.kRed+1)
    g_impacts_lo.SetLineColor(ROOT.kAzure+4)
    g_impactsA_lo.SetLineStyle(1)
    g_impactsA_lo.SetLineWidth(0)
    g_impacts_lo.SetMarkerSize(0)
    g_impacts_hi.SetMarkerSize(0)
    g_impactsA_hi.Draw('2 SAME')
    g_impactsA_lo.Draw('2 SAME')
    g_impacts_hi.Draw('E SAME')
    g_impacts_lo.Draw('E SAME')
    pads[1].RedrawAxis()
    pads[1].RedrawAxis()

    #-- Legend, CMS logo, best-fit value
    # legendbox = [0.001, 0.995, 0.77, 0.915] #x1, y1, x2, y2
    legendbox = [0.23, 0.995, 0.995, 0.915] #x1, y1, x2, y2
    legendtextsize = 0.03
    legend = ROOT.TLegend(legendbox[0], legendbox[1], legendbox[2], legendbox[3], '', 'NBNDC')
    legend.SetFillStyle(0)
    legend.SetTextSize(legendtextsize)
    legend.SetNColumns(3)
    legend.AddEntry(g_pulls, 'Fit constraint (obs.)', 'LP')
    legend.AddEntry(g_impacts_hi, '+1#sigma impact (obs.)', 'l')
    legend.AddEntry(g_impacts_lo, '-1#sigma impact (obs.)', 'l')
    legend.AddEntry(g_pullsA, 'Fit constraint (exp.)', 'F')
    legend.AddEntry(g_impactsA_hi, '+1#sigma impact (exp.)', 'F')
    legend.AddEntry(g_impactsA_lo, '-1#sigma impact (exp.)', 'F')
    legend.Draw()

    #-- Draw CMS logo
    # docmslogo = True
    # cmslogopad = pads[0]
    # left = 0.35; top = 0.87 #Position of 'CMS' label
    # if args.blind:
    #     left = 0.87; top = 0.92
    #     if args.cms_label != "": left = 0.78
    docmslogo = True
    cmslogopad = pads[0]
    left = 0.01; top = .99 #Position of 'CMS' label
    if args.blind:
        left = 0.87; top = 0.92
        if args.cms_label != "": left = 0.78

    if docmslogo:

        #-- Use built-in CombineHarvester function
        # plot.DrawCMSLogo(pad=cmslogopad, cmsText='CMS', extraText=args.cms_label, iPosX=0,
        #        relPosX=cmslogo_extrax, relPosY=0, relExtraDY=0, extraText2='', cmsTextSize=0.3)

        #-- Draw labels manually (more flexible)
        CMS_text = ROOT.TLatex(left, top, "CMS")
        CMS_text.SetNDC()
        CMS_text.SetTextColor(ROOT.kBlack)
        CMS_text.SetTextFont(61)
        CMS_text.SetTextAlign(13)
        CMS_text.SetTextSize(0.055)

        extraText = ROOT.TLatex(left+0.09, top, args.cms_label)
        extraText.SetNDC()
        extraText.SetTextFont(52)
        extraText.SetTextSize(0.055)

        CMS_text.Draw('same') #Draw 'CMS'
        extraText.Draw('same') #Draw extra label, if any

    #-- Draw best-fit value and uncertainties
    # s_nom, s_hi, s_lo = GetRounded(POI_fit[1], POI_fit[2] - POI_fit[1], POI_fit[1] - POI_fit[0])
    # THIS PART DRAWS THE POI and sets how many digits to display
    s_nom, s_hi, s_lo = "{:.3f}".format(POI_fit[1]), "{:.3f}".format( POI_fit[2] - POI_fit[1]), "{:.3f}".format( POI_fit[1] - POI_fit[0])
    if not args.blind:
        plot.DrawTitle(pad=pads[1], text='%s' % (
                '#hat{%s}' % poi_translated
                )
            +' = %s^{#plus%s}_{#minus%s}%s' % (
                s_nom, s_hi, s_lo,
                '' if args.units is None else ' '+args.units
            ), align=3, textOffset=0.12, textSize=0.25)
    # pads[0].RedrawAxis()
    pads[1].RedrawAxis()

    #-- Write to .pdf file
    extra = ''
    if page == 0:
        extra = '('
    if page == n - 1:
        extra = ')'
    canv.Print('.pdf%s' % extra)
# //--------------------------------------------
# //--------------------------------------------
