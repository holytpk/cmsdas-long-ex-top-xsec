import sys
import os

import ROOT as r
from ROOT import TFile, TTree, TCanvas, TH1D


inDir = str(sys.argv[1])
channel = str(sys.argv[2])
xleaf = str(sys.argv[3])
yleaf = str(sys.argv[4])

inputFileName = os.path.join(inDir,'scans',xleaf,channel+'.root')
outputFileName = os.path.join(inDir,'scans',xleaf,channel+'_'+yleaf+'_vs_'+xleaf+'.png')

file = TFile.Open(inputFileName,'READ')
tree = file.Get('limit')

leaves = tree.GetListOfLeaves()

for i in range(0,leaves.GetEntries() ) :
	leaf = leaves.At(i)

nEntries = tree.GetEntries()

tree.GetEntry(1)
xmin = tree.GetLeaf(xleaf).GetValue()
tree.GetEntry(2)
xsecond = tree.GetLeaf(xleaf).GetValue()
tree.GetEntry(nEntries-1)
xmax = tree.GetLeaf(xleaf).GetValue()

xstep = xsecond-xmin
bins = int(round((xmax-xmin)/xstep)) + 1

plot = TH1D(yleaf+'_vs_'+xleaf,yleaf+' vs '+xleaf+' ('+channel+')',bins,xmin-xstep/2,xmax+xstep/2)

ymin = 100000000
ymax = -100000000

for i in range(1,nEntries):
	tree.GetEntry(i)
	xval = tree.GetLeaf(xleaf).GetValue()
	yval = tree.GetLeaf(yleaf).GetValue()
	if (yval<ymin and yval !=0):
		ymin = yval
	if yval > ymax:
		ymax = yval
	plot.Fill(xval,yval)

yrange = ymax - ymin

c1 = TCanvas()
plot.GetXaxis().SetTitle(xleaf)
plot.GetYaxis().SetTitle(yleaf)
plot.GetYaxis().SetRangeUser(ymin-0.1*yrange, ymax+0.1*yrange)
plot.SetLineWidth(2)
plot.Draw("hist")

c1.SaveAs(outputFileName)
