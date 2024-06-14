from functools import reduce
import numpy as np
import awkward as ak
import coffea
import coffea.lumi_tools
import coffea.jetmet_tools
import uproot
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import pepper
from pepper import sonnenschein, betchart
import pepper.config


@dataclass
class VariationArg:
    """Holds information on a systematic variation of the jet energies

    Attributes
    ----------
    name
        Name of the variation
    junc
        First element names the jet energy uncertinty source, second element
        names the direction in which it is varied
    jer
        Direction into which the jet energy resolution is varied. Can also be
        "central" for no variation, but smearing still being applied
    met
        Direction into which the MET uncertainty is varied
    """
    name: Optional[str] = None
    junc: Optional[Tuple[str, str]] = None
    jer: Optional[str] = "central"
    met: Optional[str] = None


logger = logging.getLogger(__name__)


class ProcessorBasicPhysics(pepper.Processor):
    """Processor containing basic object definitions and cuts useful
       for most physics analyses."""

    config_class = pepper.ConfigBasicPhysics

    def __init__(self, config, eventdir):
        super().__init__(config, eventdir)

    def get_jetmet_variation_args(self):
        """Get a list of varations that should be done for the Jet/MET
        uncertainties"""
        ret = []
        if ("jet_resolution" not in self.config
                or "jet_ressf" not in self.config):
            jer = None
        else:
            jer = "central"
        ret.append(VariationArg("UncMET_up", met="up", jer=jer))
        ret.append(VariationArg("UncMET_down", met="down", jer=jer))
        if "jet_uncertainty" in self.config:
            junc = self.config["jet_uncertainty"]
            if "junc_sources_to_use" in self.config:
                levels = self.config["junc_sources_to_use"]
            else:
                levels = junc.levels
            for source in levels:
                if source not in junc.levels and not "OnLight" in source:
                    raise pepper.config.ConfigError(
                        f"Source not in jet uncertainties: {source}")
                if source == "jes":
                    name = "Junc_"
                else:
                    name = f"Junc{source.replace('_', '')}_"
                ret.append(VariationArg(
                    name + "up", junc=("up", source), jer=jer))
                ret.append(VariationArg(
                    name + "down", junc=("down", source), jer=jer))
        if "jet_resolution" in self.config and "jet_ressf" in self.config:
            ret.append(VariationArg("Jer_up", jer="up"))
            ret.append(VariationArg("Jer_down", jer="down"))
        return ret

    def get_jetmet_nominal_arg(self):
        """Get a ``VariationArg`` describing the nominal evaluation"""
        if "jet_resolution" in self.config and "jet_ressf" in self.config:
            return VariationArg(None)
        else:
            return VariationArg(None, jer=None)

    def gentop(self, data):
        """Return generator-level tops."""
        part = data["GenPart"]
        part = part[~ak.is_none(part.parent, axis=1)]
        part = part[part.hasFlags("isLastCopy")]
        part = part[abs(part.pdgId) == 6]
        part = part[ak.argsort(part.pdgId, ascending=False)]
        return part

    def do_top_pt_reweighting(self, data):
        """Top pt reweighting according to
        https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting
        """
        pt = data["gent_lc"].pt
        weiter = self.config["top_pt_reweighting"]
        rwght = weiter(pt[:, 0], pt[:, 1])
        if weiter.sys_only:
            if self.config["compute_systematics"]:
                return np.full(len(data), True), {"TopPtReweighting_0": rwght}
            else:
                return np.full(len(data), True)
        else:
            if self.config["compute_systematics"]:
                return rwght, {"TopPtReweighting_0": 1/rwght}
            else:
                return rwght

    def do_pileup_reweighting(self, dsname, data):
        """Pileup reweighting"""
        ntrueint = data["Pileup"]["nTrueInt"]
        weighter = self.config["pileup_reweighting"]
        weight = weighter(dsname, ntrueint)
        if self.config["compute_systematics"]:
            # If central is zero, let up and down factors also be zero
            weight_nonzero = np.where(weight == 0, np.inf, weight)
            up = weighter(dsname, ntrueint, "up")
            down = weighter(dsname, ntrueint, "down")
            sys = {"pileup": (up / weight_nonzero, down / weight_nonzero)}
            return weight, sys
        return weight

    def add_me_uncertainties(self, dsname, selector, data):
        """Matrix-element renormalization and factorization scale"""
        # Get describtion of individual columns of this branch with
        # Events->GetBranch("LHEScaleWeight")->GetTitle() in ROOT
        data = selector.data
        if dsname + "_LHEScaleSumw" in self.config["mc_lumifactors"]:
            norm = self.config["mc_lumifactors"][dsname + "_LHEScaleSumw"]
            if len(norm) == 44:
                # See https://github.com/cms-nanoAOD/cmssw/issues/537
                idx = [34, 5, 24, 15]
            elif len(norm) == 9:
                # This appears to be the standard case for most data sets
                idx = [7, 1, 5, 3]
            elif len(norm) == 8:
                # Same as length 9, just missing the nominal weight at index 4
                idx = [6, 1, 4, 3]
            elif len(norm) == 18:
                # Two sets of scale varations. From the titles in NanoAOD
                # the exact order is not clear. Assume the right one comes
                # first.
                idx = [14, 2, 10, 6]
            else:
                raise RuntimeError(
                    "Unexpected length of the norm for LHEScaleWeight: "
                    f"{len(norm)}")
            selector.set_systematic(
                "MEren",
                data["LHEScaleWeight"][:, idx[0]] * abs(norm[idx[0]]),
                data["LHEScaleWeight"][:, idx[1]] * abs(norm[idx[1]]))
            selector.set_systematic(
                "MEfac",
                data["LHEScaleWeight"][:, idx[2]] * abs(norm[idx[2]]),
                data["LHEScaleWeight"][:, idx[3]] * abs(norm[idx[3]]))

    def add_pdf_uncertainties(self, dsname, selector, data):
        """Add PDF uncertainties, using the methods described here:
        https://arxiv.org/pdf/1510.03865.pdf#section.6"""
        if ("LHEPdfWeight" not in data.fields
                or "pdf_types" not in self.config):
            return

        split_pdf_uncs = False
        if "split_pdf_uncs" in self.config:
            split_pdf_uncs = self.config["split_pdf_uncs"]
        pdfs = data["LHEPdfWeight"]

        normalize_pdf_uncs = False
        if "normalize_pdf_uncs" in self.config:
            normalize_pdf_uncs = self.config["normalize_pdf_uncs"]

        pdf_doc = pdfs.__doc__
        pdf_type = None
        for LHA_ID, _type in self.config["pdf_types"].items():
            if LHA_ID in pdf_doc:
                pdf_type = _type.lower()

        if normalize_pdf_uncs:
            if dsname + "_LHEPdfSumw" not in self.config["mc_lumifactors"]:
                raise pepper.config.ConfigError(
                    "Missing lumifactors for PDF uncertainties for dataset "
                    f"'{dsname}'. Please run compute_mc_lumifactors.py with "
                    "the '-p' option.")
            norm = self.config["mc_lumifactors"][dsname + "_LHEPdfSumw"]
            pdfs = pdfs * abs(np.array(norm)[np.newaxis, :])

        # Check if sample has alpha_s variations - currently assuming number of
        # regular variations is a multiple of 10
        if len(data) == 0:
            has_as_unc = False
        else:
            has_as_unc = (len(pdfs[0]) % 10) > 1
            # Workaround for "order of scale and pdf weights not consistent"
            # See https://twiki.cern.ch/twiki/bin/view/CMS/MCKnownIssues
            if ak.mean(pdfs[0]) < 0.6:  # approximate, see if factor 2 needed
                pdfs = ak.without_parameters(pdfs)
                pdfs = ak.concatenate([pdfs[:, 0:1], pdfs[:, 1:] * 2], axis=1)
        n_offset = -2 if has_as_unc else None

        if split_pdf_uncs:
            # Just output variations - user
            # will need to combine these for limit setting
            num_variation = len(pdfs[0]) + (n_offset or 0)
            if pdf_type == "true_hessian" or pdf_type == "hessian":
                # First element is central value - adjust all other
                # elements relative to this
                selector.set_systematic(
                    "PDF", *[pdfs[:, i] - pdfs[:, 0] + 1
                             for i in range(1, num_variation)],
                    scheme="numeric")
                if has_as_unc:
                    selector.set_systematic(
                        "PDFalphas",
                        pdfs[:, -1] - pdfs[:, 0] + 1,
                        pdfs[:, -2] - pdfs[:, 0] + 1)
            elif pdf_type.startswith("mc"):
                selector.set_systematic(
                    "PDF",
                    *[pdfs[:, i] for i in range(num_variation)],
                    scheme="numeric")
                if has_as_unc:
                    selector.set_systematic(
                        "PDFalphas", pdfs[:, -1], pdfs[:, -2])
            elif pdf_type is None:
                raise pepper.config.ConfigError(
                    "PDF LHA Id not included in config. PDF docstring is: "
                    + pdf_doc)
            else:
                raise pepper.config.ConfigError(
                    f"PDF type {pdf_type} not recognised. Valid options "
                    "are 'True_Hessian', 'Hessian', 'MC' and 'MC_Gaussian'")
        else:
            if pdf_type == "true_hessian":
                # Treatment of true hessian uncertainties, for e.g. CTEQ
                # or HERA sets
                eigen_vals = ak.to_numpy(pdfs[:, 1:n_offset])
                eigen_vals = eigen_vals.reshape(
                    (eigen_vals.shape[0], eigen_vals.shape[1] // 2, 2))
                central, eigenvals = ak.broadcast_arrays(
                    pdfs[:, 0, None, None], eigen_vals)
                var_up = ak.max((eigen_vals - central), axis=2)
                var_up = ak.where(var_up > 0, var_up, 0)
                var_up = np.sqrt(ak.sum(var_up ** 2, axis=1))
                var_down = ak.max((central - eigen_vals), axis=2)
                var_down = ak.where(var_down > 0, var_down, 0)
                var_down = np.sqrt(ak.sum(var_down ** 2, axis=1))
                unc = None
            if pdf_type == "hessian":
                # Treatment of pseudo hessian uncertainties, for e.g. NNPDF
                # or pdf4LHC sets
                eigen_vals = ak.to_numpy(pdfs[:, 1:n_offset])
                variations = eigen_vals - pdfs[:, 0, None]
                unc = np.sqrt(ak.sum(variations ** 2, axis=1))
            elif pdf_type == "mc":
                # ak.sort produces an error here. Work-around:
                variations = np.sort(ak.to_numpy(pdfs[:, 1:n_offset]))
                nvar = ak.num(variations)[0]
                unc = (variations[:, int(round(0.841344746*nvar))]
                       - variations[:, int(round(0.158655254*nvar))]) / 2
            elif pdf_type == "mc_gaussian":
                mean = ak.mean(pdfs[:, 1:n_offset], axis=1)
                unc = np.sqrt((ak.sum(pdfs[:, 1:n_offset] - mean) ** 2)
                              / (ak.num(pdfs)[0] - (3 if n_offset else 1)))
            elif pdf_type is None:
                raise pepper.config.ConfigError(
                    "PDF LHA Id not included in config. PDF docstring is: "
                    + pdf_doc)
            else:
                raise pepper.config.ConfigError(
                    f"PDF type {pdf_type} not recognised. Valid options "
                    "are 'True_Hessian', 'Hessian', 'MC' and 'MC_Gaussian'")

            # Add PDF alpha_s uncertainties
            if has_as_unc:
                if ("combine_alpha_s" in self.config and
                        self.config["combine_alpha_s"]):
                    alpha_s_unc = (pdfs[:, -1] - pdfs[:, -2]) / 2
                    if unc is not None:
                        unc = np.sqrt(unc ** 2 + alpha_s_unc ** 2)
                    else:
                        var_up = np.sqrt(var_up ** 2 + alpha_s_unc ** 2)
                        var_down = np.sqrt(var_down ** 2 + alpha_s_unc ** 2)
                else:
                    selector.set_systematic(
                        "PDFalphas",
                        pdfs[:, -1] - pdfs[:, 0] + 1,
                        pdfs[:, -2] - pdfs[:, 0] + 1)
            if unc is not None:
                selector.set_systematic("PDF", 1 + unc, 1 - unc)
            else:
                selector.set_systematic("PDF", 1 + var_up, 1 - var_down)



    def add_generator_uncertainies(self, dsname, selector):
        """Add MC generator uncertainties: ME, PS and PDF"""
        data = selector.data
        self.add_me_uncertainties(dsname, selector, data)
        self.add_ps_uncertainties(selector, data)
        self.add_pdf_uncertainties(dsname, selector, data)

    def crosssection_scale(self, dsname, data):
        """Cross section uncertainties. These are values depending only on the
        data set"""
        num_events = len(data)
        lumifactors = self.config["mc_lumifactors"]
        factor = np.full(num_events, lumifactors[dsname])
        if "stitching_factors" in self.config:
            for ds, sfs in self.config["stitching_factors"].items():
                if dsname.startswith(ds):
                    factor = np.full(num_events,
                                     lumifactors[ds + "_inclusive"])
                    edges = sfs["edges"] + [np.inf]
                    for i, fac in enumerate(sfs["factors"]):
                        var = pepper.hist_defns.DataPicker(sfs["axis"])(data)
                        factor[(var >= edges[i]) & (var < edges[i + 1])] *= fac
        if (self.config["compute_systematics"]
                and not ("skip_nonshape_systematics" in self.config
                         and self.config["skip_nonshape_systematics"])
                and dsname in self.config["crosssection_uncertainty"]
                and dsname not in self.config["dataset_for_systematics"]):
            xsuncerts = self.config["crosssection_uncertainty"]
            groups = set(v[0] for v in xsuncerts.values() if v is not None)
            systematics = {}
            for group in groups:
                if xsuncerts[dsname] is None or group != xsuncerts[dsname][0]:
                    continue
                uncert = xsuncerts[dsname][1]
                systematics[group + "XS"] = (np.full(num_events, 1 + uncert),
                                             np.full(num_events, 1 - uncert))
            return factor, systematics
        else:
            return factor

    def blinding(self, is_mc, data):
        """Skip every nth event in the experimental data. One way to blind your
        analysis"""
        if not is_mc:
            return np.mod(data["event"], self.config["blinding_denom"]) == 0
        else:
            return np.full(len(data), 1/self.config["blinding_denom"])

    def good_lumimask(self, is_mc, dsname, data):
        """Keep only data events that are in the golden JSON files.
           For MC, add luminosity uncertainty"""
        if is_mc:
            weight = np.ones(len(data))
            if (self.config["compute_systematics"]
                    and not ("skip_nonshape_systematics" in self.config
                             and self.config["skip_nonshape_systematics"])):
                sys = {}
                if self.config["year"] in ("2018", "2016", "ul2018",
                                           "ul2016pre", "ul2016post"):
                    sys["lumi"] = (np.full(len(data), 1 + 0.025),
                                   np.full(len(data), 1 - 0.025))
                elif self.config["year"] in ("2017", "ul2017"):
                    sys["lumi"] = (np.full(len(data), 1 + 0.023),
                                   np.full(len(data), 1 - 0.023))
                return weight, sys
            else:
                return weight
        elif "lumimask" not in self.config:
            return np.full(len(data), True)
        else:
            run = np.array(data["run"])
            luminosity_block = np.array(data["luminosityBlock"])
            lumimask = coffea.lumi_tools.LumiMask(self.config["lumimask"])
            return lumimask(run, luminosity_block)

    def get_era(self, data, is_mc):
        """Return data-taking eras based on run number."""
        if is_mc:
            return self.config["year"] + "MC"
        else:
            if len(data) == 0:
                return "no_events"
            run = np.array(data["run"])[0]
            # Assumes all runs in file come from same era
            for era, startstop in self.config["data_eras"].items():
                if ((run >= startstop[0]) & (run <= startstop[1])):
                    return era
            #raise ValueError(f"Run {run} does not correspond to any era")
            return "no_era"

    def passing_trigger(self, pos_triggers, neg_triggers, data):
        """Return mask for events that pass the trigger."""
        hlt = data["HLT"]
        available = ak.fields(hlt)
        triggered = np.full(len(data), False)
        for trigger_path in pos_triggers:
            if trigger_path not in available:
                logger.debug(f"HLT_{trigger_path} not in file")
                continue
            triggered |= np.asarray(hlt[trigger_path])
        for trigger_path in neg_triggers:
            if trigger_path not in available:
                continue
            triggered &= ~np.asarray(hlt[trigger_path])
        return triggered

    def add_l1_prefiring_weights(self, data):
        """Prefiring weights needed for 2016 and 2017 data"""
        w = data["L1PreFiringWeight"]
        nom = w["Nom"]
        if self.config["compute_systematics"]:
            sys = {"L1prefiring": (w["Up"] / nom, w["Dn"] / nom)}
            return nom, sys
        return nom

    def mpv_quality(self, data):
        """Check quality of primary vertex."""
        # Does not include check for fake. Is this even needed?
        R = np.hypot(data["PV_x"], data["PV_y"])
        return ((data["PV_chi2"] != 0)
                & (data["PV_ndof"] > 4)
                & (abs(data["PV_z"]) <= 24)
                & (R <= 2))

    def met_filters(self, is_mc, data):
        """Apply met filters."""
        year = str(self.config["year"]).lower()
        if not self.config["apply_met_filters"]:
            return np.full(data.shape, True)
        else:
            passing_filters =\
                (data["Flag"]["goodVertices"]
                 & data["Flag"]["globalSuperTightHalo2016Filter"]
                 & data["Flag"]["HBHENoiseFilter"]
                 & data["Flag"]["HBHENoiseIsoFilter"]
                 & data["Flag"]["EcalDeadCellTriggerPrimitiveFilter"]
                 & data["Flag"]["BadPFMuonFilter"])
            if not is_mc:
                passing_filters = (
                    passing_filters & data["Flag"]["eeBadScFilter"])
        if year in ("2018", "2017"):
            passing_filters = (
                passing_filters & data["Flag"]["ecalBadCalibFilterV2"])
        if year in ("ul2018", "ul2017"):
            passing_filters = (
                passing_filters & data["Flag"]["ecalBadCalibFilter"])
        if year in ("ul2018", "ul2017", "ul2016post", "ul2016pre"):
            passing_filters = (
                passing_filters & data["Flag"]["eeBadScFilter"])

        return passing_filters

    def in_transreg(self, abs_eta):
        """Check if object is in detector transition region."""
        return (1.444 < abs_eta) & (abs_eta < 1.566)

    def electron_id(self, e_id, electron):
        """Check if electrons have ID specified in the config file."""
        if e_id == "skip":
            has_id = True
        if e_id == "cut:loose":
            has_id = electron["cutBased"] >= 2
        elif e_id == "cut:medium":
            has_id = electron["cutBased"] >= 3
        elif e_id == "cut:tight":
            has_id = electron["cutBased"] >= 4
        elif e_id == "mva:noIso80":
            has_id = electron["mvaFall17V2noIso_WP80"]
        elif e_id == "mva:noIso90":
            has_id = electron["mvaFall17V2noIso_WP90"]
        elif e_id == "mva:Iso80":
            has_id = electron["mvaFall17V2Iso_WP80"]
        elif e_id == "mva:Iso90":
            has_id = electron["mvaFall17V2Iso_WP90"]
        else:
            raise ValueError("Invalid electron id string")
        return has_id

    def electron_cuts(self, electron, good_lep):
        """Apply some basic electron quality cuts.
        If good_lep is True, config values prefixed with 'good_' for pt and ID
        are used. Otherwise the ones with prefix 'additional_' are used."""
        if self.config["ele_cut_transreg"]:
            sc_eta_abs = abs(electron["eta"]
                             + electron["deltaEtaSC"])
            is_in_transreg = self.in_transreg(sc_eta_abs)
        else:
            is_in_transreg = np.array(False)
        if good_lep:
            e_id, pt_min = self.config[[
                "good_ele_id", "good_ele_pt_min"]]
        else:
            e_id, pt_min = self.config[[
                "additional_ele_id", "additional_ele_pt_min"]]
        eta_min, eta_max = self.config[["ele_eta_min", "ele_eta_max"]]
        return (self.electron_id(e_id, electron)
                & (~is_in_transreg)
                & (eta_min < electron["eta"])
                & (electron["eta"] < eta_max)
                & (pt_min < electron["pt"]))

    def pick_electrons(self, data):
        """Get electrons that pass basic quality cuts."""
        electrons = data["Electron"]
        return electrons[self.electron_cuts(electrons, good_lep=True)]

    def muon_id(self, m_id, muon):
        """Check if muons have ID specified in the config file."""
        if m_id == "skip":
            has_id = True
        elif m_id == "cut:loose":
            has_id = muon["looseId"]
        elif m_id == "cut:medium":
            has_id = muon["mediumId"]
        elif m_id == "cut:tight":
            has_id = muon["tightId"]
        elif m_id == "mva:loose":
            has_id = muon["mvaId"] >= 1
        elif m_id == "mva:medium":
            has_id = muon["mvaId"] >= 2
        elif m_id == "mva:tight":
            has_id = muon["mvaId"] >= 3
        else:
            raise ValueError("Invalid muon id string")
        return has_id

    def muon_iso(self, iso, muon):
        """Check if muons have isolation specified in the config file."""
        if iso == "skip":
            return True
        elif iso == "cut:very_loose":
            return muon["pfIsoId"] > 0
        elif iso == "cut:loose":
            return muon["pfIsoId"] > 1
        elif iso == "cut:medium":
            return muon["pfIsoId"] > 2
        elif iso == "cut:tight":
            return muon["pfIsoId"] > 3
        elif iso == "cut:very_tight":
            return muon["pfIsoId"] > 4
        elif iso == "cut:very_very_tight":
            return muon["pfIsoId"] > 5
        else:
            iso, iso_value = iso.split(":")
            value = float(iso_value)
            if iso == "dR<0.3_chg":
                return muon["pfRelIso03_chg"] < value
            elif iso == "dR<0.3_all":
                return muon["pfRelIso03_all"] < value
            elif iso == "dR<0.4_all":
                return muon["pfRelIso04_all"] < value
            elif iso == "miniIso":
                return muon["miniPFRelIso_all"] < value
        raise ValueError("Invalid muon iso string")

    def muon_cuts(self, muon, good_lep):
        """Apply some basic muon quality cuts
        If good_lep is True, config values prefixed with 'good_' for pt, ID
        and iso are used. Otherwise the ones with prefix 'additional_' are
        used."""
        if self.config["muon_cut_transreg"]:
            is_in_transreg = self.in_transreg(abs(muon["eta"]))
        else:
            is_in_transreg = np.array(False)
        if good_lep:
            m_id, pt_min, iso = self.config[[
                "good_muon_id", "good_muon_pt_min", "good_muon_iso"]]
        else:
            m_id, pt_min, iso = self.config[[
                "additional_muon_id", "additional_muon_pt_min",
                "additional_muon_iso"]]
        eta_min, eta_max = self.config[["muon_eta_min", "muon_eta_max"]]
        return (self.muon_id(m_id, muon)
                & self.muon_iso(iso, muon)
                & (~is_in_transreg)
                & (eta_min < muon["eta"])
                & (muon["eta"] < eta_max)
                & (pt_min < muon["pt"]))

    def pick_muons(self, data):
        """Get muons that pass basic quality cuts."""
        muons = data["Muon"]
        return muons[self.muon_cuts(muons, good_lep=True)]

    def apply_rochester_corr(self, muons, rng, is_mc):
        """Apply Rochester corrections for muons."""
        if not is_mc:
            dtSF = self.config["muon_rochester"].kScaleDT(
                muons["charge"], muons["pt"], muons["eta"], muons["phi"]
            )
            muons["pt"] = muons["pt"] * dtSF
        else:
            # if reco pt has corresponding gen pt
            mcSF1 = self.config["muon_rochester"].kSpreadMC(
                muons["charge"],
                muons["pt"],
                muons["eta"],
                muons["phi"],
                muons.matched_gen.pt,
            )
            # if reco pt has no corresponding gen pt
            counts = ak.num(muons["pt"])
            mc_rand = rng.uniform(size=ak.sum(counts))
            mc_rand = ak.unflatten(mc_rand, counts)
            mcSF2 = self.config["muon_rochester"].kSmearMC(
                muons["charge"],
                muons["pt"],
                muons["eta"],
                muons["phi"],
                muons["nTrackerLayers"],
                mc_rand,
            )
            # Combine the two scale factors and scale the pt
            mcSF = ak.where(
                ak.is_none(muons.matched_gen.pt, axis=1), mcSF2, mcSF1)
            # Remove masking from layout, none of the SF are masked here
            mcSF = ak.fill_none(mcSF, 0)
            muons["pt"] = muons["pt"] * mcSF
        return muons

    def build_lepton_column(self, is_mc, rng, data):
        """Build a lepton column containing electrons and muons."""
        electron = data["Electron"]
        muon = data["Muon"]
        # Apply Rochester corrections to muons
        if "muon_rochester" in self.config:
            muon = self.apply_rochester_corr(muon, rng, is_mc)
        columns = ["pt", "eta", "phi", "mass", "pdgId"]
        lepton = {}
        for column in columns:
            lepton[column] = ak.concatenate([electron[column], muon[column]],
                                            axis=1)
        lepton = ak.zip(lepton, with_name="PtEtaPhiMLorentzVector",
                        behavior=data.behavior)

        # Sort leptons by pt
        # Also workaround for awkward bug using ak.values_astype
        # https://github.com/scikit-hep/awkward-1.0/issues/1288
        lepton = lepton[
            ak.values_astype(ak.argsort(lepton["pt"], ascending=False), int)]
        return lepton

    def compute_lepton_sf(self, data):
        """Compute identification and isolation scale factors for
           leptons (electrons and muons)."""
        eles = data["Electron"]
        muons = data["Muon"]

        weight = np.ones(len(data))
        systematics = {}

        # Scale factors as defined in the config
        # These are automatically transformed into callable objects

        # --- Electrons ---

        # Electron identification efficiency
        sffunc_ele = self.config["electron_sf"][0]

        # For the electrons only: the SF are given in terms of the abs
        # of supercluster eta. Calculate this like this
        sceta_ele = eles.eta + eles.deltaEtaSC

        # Get the SF for each individual electron
        sf_ele_central = sffunc_ele(abseta=abs(sceta_ele), pt=eles.pt)
        # Multiply the SFs for all electrons in a given event
        sf_ele_central = ak.prod(sf_ele_central, axis=1)

        # Multiply the total per-event weight with the electron SF
        weight = weight * sf_ele_central

        # Compute the systematics by varying the SF up / down
        if self.config["compute_systematics"]:
            sf_ele_up = sffunc_ele(abseta=abs(sceta_ele), pt=eles.pt, variation="up")
            sf_ele_up = ak.prod(sf_ele_up, axis=1)

            sf_ele_down = sffunc_ele(abseta=abs(sceta_ele), pt=eles.pt, variation="down")
            sf_ele_down = ak.prod(sf_ele_down, axis=1)

            systematics["electron_id_sf"] = (sf_ele_up / sf_ele_central, sf_ele_down / sf_ele_central)

        # --- Muon reconstruction & identification ---
        #
        # ...

        return weight, systematics

    def lepton_pair(self, is_mc, data):
        """Select events that contain at least two leptons."""
        accept = np.asarray(ak.num(data["Lepton"]) >= 2)
        if is_mc:
            weight, systematics = self.compute_lepton_sf(data[accept])
            accept = accept.astype(float)
            accept[accept.astype(bool)] *= np.asarray(weight)
            return accept, systematics
        else:
            return accept

    def opposite_sign_lepton_pair(self, data):
        """Select events that contain two opposite-sign leptons."""
        return (np.sign(data["Lepton"][:, 0].pdgId)
                != np.sign(data["Lepton"][:, 1].pdgId))

    def same_flavor_lepton_pair(self, data):
        """Select events that contain two same-flavor leptons."""
        return (abs(data["Lepton"][:, 0].pdgId)
                == abs(data["Lepton"][:, 1].pdgId))

    def compute_jec_factor(self, is_mc, era, data, pt=None, eta=None, area=None,
                           rho=None, raw_factor=None):
        """Return jet energy correction factor."""
        if pt is None:
            pt = data["Jet"].pt
        if eta is None:
            eta = data["Jet"].eta
        if area is None:
            area = data["Jet"].area
        if rho is None:
            if "Rho" in data.fields:
                rho = data["Rho"]["fixedGridRhoFastjetAll"]
            else:
                rho = data["fixedGridRhoFastjetAll"]
        if raw_factor is None:
            raw_factor = 1 - data["Jet"]["rawFactor"]
        if is_mc:
            jec = self.config["jet_correction_mc"]
        else:
            jec = self.config["jet_correction_data"]
            if isinstance(jec, dict):
                jec = jec[era]

        raw_pt = pt * raw_factor
        l1l2l3 = jec.getCorrection(
            JetPt=raw_pt, JetEta=eta, JetA=area, Rho=rho)
        return raw_factor * l1l2l3

    def get_junc_factor_mask(self, data, source, pt, eta, flavor):
        """
        Some Jet enery uncertainties are for specific jet flavors only. This
        method decides which jets to apply the uncertainties to.

        Returns None to indicate to apply the uncertainty to all jets.
        Otherwise a bool ak.Array with either True or False for every jet
        """
        return None  # To be modified in subclasses

    def compute_junc_factor(self, data, variation, source="jes", pt=None,
                            eta=None, flavor=None):
        """Return jet energy correction uncertainty factor."""
        if variation not in ("up", "down"):
            raise ValueError("variation must be either 'up' or 'down'")
        if source not in self.config["jet_uncertainty"].levels:
            raise ValueError(f"Jet uncertainty not found: {source}")
        if pt is None:
            pt = data["Jet"].pt
        if eta is None:
            eta = data["Jet"].eta
        if flavor is None:
            flavor = data["Jet"].partonFlavour
        counts = ak.num(pt)
        if ak.sum(counts) == 0:
            return ak.unflatten([], counts)
        junc = dict(self.config["jet_uncertainty"].getUncertainty(
            JetPt=pt, JetEta=eta))[source]
        junc = junc[:, :, 0 if variation == "up" else 1]
        mask = self.get_junc_factor_mask(data, source, pt, eta, flavor)
        if mask is not None:
            junc = ak.where(mask, junc, ak.ones_like(junc))
        return junc

    def compute_jer_factor(self, data, rng, variation="central", pt=None,
                           eta=None, hybrid=True):
        """Return jet energy resolution factor."""
        # Coffea offers a class named JetTransformer for this. Unfortunately
        # it is more inconvinient and bugged than useful.
        if pt is None:
            pt = data["Jet"].pt
        if eta is None:
            eta = data["Jet"].eta
        counts = ak.num(pt)
        if ak.sum(counts) == 0:
            return ak.unflatten([], counts)
        if "Rho" in data.fields:
            rho = data["Rho"]["fixedGridRhoFastjetAll"]
        else:
            rho = data["fixedGridRhoFastjetAll"]
        jer = self.config["jet_resolution"].getResolution(
            JetPt=pt, JetEta=eta, Rho=rho)
        jersf = self.config["jet_ressf"].getScaleFactor(
            JetPt=pt, JetEta=eta, Rho=rho)
        jersmear = jer * rng.normal(size=len(jer))
        if variation == "central":
            jersf = jersf[:, :, 0]
        elif variation == "up":
            jersf = jersf[:, :, 1]
        elif variation == "down":
            jersf = jersf[:, :, 2]
        else:
            raise ValueError("variation must be one of 'central', 'up' or "
                             "'down'")
        factor_stoch = 1 + np.sqrt(np.maximum(jersf**2 - 1, 0)) * jersmear
        if hybrid:
            # Hybrid method: Apply scaling relative to genpt if possible
            genpt = data["Jet"].matched_gen.pt
            factor_scale = 1 + (jersf - 1) * (pt - genpt) / pt
            factor = ak.where(
                ak.is_none(genpt, axis=1), factor_stoch, factor_scale)
        else:
            factor = factor_stoch
        return factor

    def compute_jet_factors(self, is_mc, era, jec, junc, jer, rng, data):
        """Return total jet factor."""
        factor = ak.ones_like(data["Jet"].pt)
        if jec:
            jecfac = self.compute_jec_factor(is_mc, era, data)
            factor = factor * jecfac
        if is_mc and junc is not None:
            juncfac = self.compute_junc_factor(data, *junc)
            factor = factor * juncfac
        if is_mc and jer is not None:
            jerfac = self.compute_jer_factor(data, rng, jer)
            factor = factor * jerfac
        ret = {}
        if jec or (is_mc and (junc is not None or jer is not None)):
            ret["jetfac"] = factor
        if is_mc and jer is not None:
            ret["jerfac"] = jerfac
        return ret

    def good_jet(self, data):
        """Apply some basic jet quality cuts."""
        jets = data["Jet"]
        leptons = data["Lepton"]
        j_id, lep_dist, eta_min, eta_max, pt_min = self.config[[
            "good_jet_id", "good_jet_lepton_distance",
            "good_jet_eta_min", "good_jet_eta_max", "good_jet_pt_min"]]
        if j_id == "skip":
            has_id = True
        elif j_id == "cut:loose":
            has_id = jets.isLoose
            # Always False in 2017 and 2018
        elif j_id == "cut:tight":
            has_id = jets.isTight
        elif j_id == "cut:tightlepveto":
            has_id = jets.isTightLeptonVeto
        else:
            raise pepper.config.ConfigError(
                    "Invalid good_jet_id: {}".format(j_id))

        j_pt = jets.pt
        if "jetfac" in ak.fields(data):
            j_pt = j_pt * data["jetfac"]
        has_lepton_close = ak.any(
            jets.metric_table(leptons) < lep_dist, axis=2)

        return (has_id
                & (~has_lepton_close)
                & (eta_min < jets.eta)
                & (jets.eta < eta_max)
                & (pt_min < j_pt))

    def has_puid(self, jets):
        """Whether jets satisfy the configured pileup ID"""
        j_puId = self.config["good_jet_puId"]
        if j_puId == "skip":
            has_puId = True
        elif j_puId == "cut:loose":
            has_puId = ak.values_astype(jets["puId"] & 0b100, bool)
        elif j_puId == "cut:medium":
            has_puId = ak.values_astype(jets["puId"] & 0b10, bool)
        elif j_puId == "cut:tight":
            has_puId = ak.values_astype(jets["puId"] & 0b1, bool)
        else:
            raise pepper.config.ConfigError(
                    "Invalid good_jet_id: {}".format(j_puId))
        # Only apply PUID if pT < 50 GeV
        has_puId = has_puId | (jets.pt >= 50)
        return has_puId

    def build_jet_column(self, is_mc, data):
        """Build a column of jets passing the jet quality cuts,
           including a 'btag' key (containing the value of the
           chosen btag discriminator) and a 'btagged' key
           (to select jets that are tagged as b-jets)."""
        is_good_jet = self.good_jet(data)
        jets = data["Jet"][is_good_jet]
        if "jetfac" in ak.fields(data):
            jets["pt"] = jets["pt"] * data["jetfac"][is_good_jet]
            jets["mass"] = jets["mass"] * data["jetfac"][is_good_jet]
            jets = jets[ak.argsort(jets["pt"], ascending=False)]

        # Evaluate b-tagging
        tagger, wp = self.config["btag"].split(":")
        if tagger == "deepcsv":
            jets["btag"] = jets["btagDeepB"]
        elif tagger == "deepjet":
            jets["btag"] = jets["btagDeepFlavB"]
        else:
            raise pepper.config.ConfigError(
                "Invalid tagger name: {}".format(tagger))
        year = self.config["year"]
        wptuple = pepper.scale_factors.BTAG_WP_CUTS[tagger][year]
        if not hasattr(wptuple, wp):
            raise pepper.config.ConfigError(
                "Invalid working point \"{}\" for {} in year {}".format(
                    wp, tagger, year))
        jets["btagged"] = jets["btag"] > getattr(wptuple, wp)
        jets["pass_pu_id"] = self.has_puid(jets)
        if is_mc:
            # A jet is considered to be a pileup jet if there is no gen jet
            # within Delta R < 0.4
            jets["has_gen_jet"] = ak.fill_none(
                jets.delta_r(jets.matched_gen) < 0.4, False)
        return jets

    def jets_with_puid(self, data):
        """Get all jets satisfying the configured pileup ID"""
        jets = data["Jet"]
        return jets[jets.pass_pu_id]

    def build_lowptjet_column(self, is_mc, era, junc, jer, rng, data):
        """Build a column of low-pt jets, needed to propagate jet
           corrections to low-pt jets and consequently build the
           MET column."""
        jets = data["CorrT1METJet"]
        # For MET we care about jets close to 15 GeV. JEC is derived with
        # pt > 10 GeV and |eta| < 5.2, thus cut there
        jets = jets[(jets.rawPt > 10) & (abs(jets.eta) < 5.2)]
        l1l2l3 = self.compute_jec_factor(
            is_mc, era, data, jets.rawPt, jets.eta, jets.area,
            raw_factor=ak.ones_like(jets.rawPt))
        jets["pt"] = l1l2l3 * jets.rawPt
        jets["pt_nomuon"] = jets["pt"] * (1 - jets["muonSubtrFactor"])

        # Actually if reapply_met is True one would need to take into account
        # the difference between JEC applied in NanoAOD and the one that is
        # applied in the Processor for MET type-1 corrections. However, as
        # NanoAOD doesn't provide a raw factor for low pt jets, assume the
        # difference is negligible.
        if junc is not None:
            jets["juncfac"] = self.compute_junc_factor(
                data, *junc, pt=jets["pt"], eta=jets["eta"],
                flavor=np.zeros_like(jets["pt"]))
        else:
            jets["juncfac"] = ak.ones_like(jets["pt"])
        if jer is not None:
            jets["jerfac"] = self.compute_jer_factor(
                data, rng, jer, jets["pt"], jets["eta"], False)
        else:
            jets["jerfac"] = ak.ones_like(jets["pt"])

        jets["emef"] = jets["mass"] = ak.zeros_like(jets["pt"])
        jets.behavior = data["Jet"].behavior
        jets = ak.with_name(jets, "Jet")

        return jets

    def build_met_column(self, is_mc, junc, jer, rng, era, data,
                         variation="central"):
        """Build a column for missing transverse energy.
        If varation is 'up' or 'down', the unclustered MET varation (up or
        down) will be applied. If it is None or 'central', nominal MET is
        used."""
        met = data["MET"]
        metx = met.pt * np.cos(met.phi)
        mety = met.pt * np.sin(met.phi)
        if ("MET_xy_shifts" in self.config and era != "no_events"):
            metshifts = self.config["MET_xy_shifts"]
            metx = metx - (metshifts["METxcorr"][era][0]
                           * data["PV"]["npvs"]
                           + metshifts["METxcorr"][era][1])
            mety = mety - (metshifts["METycorr"][era][0]
                           * data["PV"]["npvs"]
                           + metshifts["METycorr"][era][1])

        if variation == "up":
            metx = metx + met.MetUnclustEnUpDeltaX
            mety = mety + met.MetUnclustEnUpDeltaY
        elif variation == "down":
            metx = metx - met.MetUnclustEnUpDeltaX
            mety = mety - met.MetUnclustEnUpDeltaY
        elif variation != "central" and variation is not None:
            raise ValueError(
                "variation must be one of None, 'central', 'up' or 'down'")
        if "jetfac" in ak.fields(data):
            factors = data["jetfac"]
            if "smear_met" in self.config:
                smear_met = self.config["smear_met"]
            else:
                smear_met = False
            if "jerfac" in ak.fields(data) and not smear_met:
                factors = factors / data["jerfac"]
            # Do MET type-1 and smearing corrections
            jets = data["OrigJet"]
            jets = ak.zip({
                "pt": jets.pt,
                "pt_nomuon": jets.pt * (1 - jets.muonSubtrFactor),
                "factor": factors - 1,
                "eta": jets.eta,
                "phi": jets.phi,
                "mass": jets.mass,
                "emef": jets.neEmEF + jets.chEmEF
            }, with_name="Jet", behavior=jets.behavior)
            lowptjets = self.build_lowptjet_column(is_mc, era, junc, jer, rng, data)
            # Cut according to MissingETRun2Corrections Twiki
            jets = jets[(jets["pt_nomuon"] > 15) & (jets["emef"] < 0.9)]
            lowptjets = lowptjets[(lowptjets["pt_nomuon"] > 15)
                                  & (lowptjets["emef"] < 0.9)]
            # lowptjets lose their type here. Probably a bug, workaround
            lowptjets = ak.with_name(lowptjets, "Jet")
            if smear_met:
                lowptfac = lowptjets["juncfac"] * lowptjets["jerfac"] - 1
            else:
                lowptfac = lowptjets["juncfac"] - 1
            metx = metx - (
                ak.sum(jets.x * jets["factor"], axis=1)
                + ak.sum(lowptjets.x * lowptfac, axis=1))
            mety = mety - (
                ak.sum(jets.y * jets["factor"], axis=1)
                + ak.sum(lowptjets.y * lowptfac, axis=1))
        met["pt"] = np.hypot(metx, mety)
        met["phi"] = np.arctan2(mety, metx)
        return met

    def in_hem1516(self, phi, eta):
        """Return mask to select objects in faulty sector of hadronic
           calorimeter encap (HEM 15/16 issue in 2018 data)."""
        return ((-3.0 < eta) & (eta < -1.3) & (-1.57 < phi) & (phi < -0.87))

    def hem_cut(self, data):
        """Keep objects without the HEM 15/16 issue in 2018 data."""
        cut_ele = self.config["hem_cut_if_ele"]
        cut_muon = self.config["hem_cut_if_muon"]
        cut_jet = self.config["hem_cut_if_jet"]

        keep = np.full(len(data), True)
        if cut_ele:
            ele = data["Electron"]
            keep = keep & (~self.in_hem1516(ele.phi, ele.eta).any())
        if cut_muon:
            muon = data["Muon"]
            keep = keep & (~self.in_hem1516(muon.phi, muon.eta).any())
        if cut_jet:
            jet = data["Jet"]
            keep = keep & (~self.in_hem1516(jet.phi, jet.eta).any())
        return keep

    def lep_pt_requirement(self, data):
        """Require leptons with minimum pT threshold."""
        n = np.zeros(len(data))
        # This assumes leptons are ordered by pt highest first
        for i, pt_min in enumerate(self.config["lep_pt_min"]):
            # flatten to workaround awkward bug
            # https://github.com/scikit-hep/awkward-1.0/issues/1305
            mask = ak.flatten(ak.num(data["Lepton"]) > i, axis=None)
            n[mask] += np.asarray(
                pt_min < data["Lepton"].pt[mask, i]).astype(int)
        return n >= self.config["lep_pt_num_satisfied"]
    
    def no_additional_leptons(self, is_mc, data):
        """Veto events with >= 3 leptons."""
        add_ele = self.electron_cuts(data["Electron"], good_lep=False)
        add_muon = self.muon_cuts(data["Muon"], good_lep=False)
        return ak.sum(add_ele, axis=1) + ak.sum(add_muon, axis=1) <= 2

    def has_jets(self, data):
        """Require events with minimum number of jets."""
        return self.config["num_jets_atleast"] <= ak.num(data["Jet"])

    def compute_puid_sys(self, central, weighter, wp, eta, pt,
                         pass_puid, has_gen_jet, sf_type):
        """Jet pileup ID systematic weights"""
        up = weighter(wp, eta, pt, pass_puid, has_gen_jet, sf_type, "up")
        down = weighter(wp, eta, pt, pass_puid, has_gen_jet, sf_type, "down")
        return (up / central, down / central)

    def jet_puid_sfs(self, data):
        """Compute event weights and systematics, if requested, for the jet
        pileup ID"""
        # Only apply SFs to jets for which the PU ID cut is applied,
        # i.e. pT < 50 GeV
        jets = data["Jet"][data["Jet"].pt < 50]
        wp = self.config["good_jet_puId"].split(":", 1)[1]
        pass_puid = jets["pass_pu_id"]
        has_gen_jet = jets["has_gen_jet"]
        eta = jets.eta
        pt = jets.pt
        weighter = self.config["jet_puid_sf"]
        systematics = {}
        weight = weighter(wp, eta, pt, pass_puid, has_gen_jet, "eff")
        if self.config["compute_systematics"]:
            systematics["jet_puid_eff"] = self.compute_puid_sys(
                weight, weighter, wp, eta, pt, pass_puid, has_gen_jet, "eff")
        if weighter.has_mis_prob:
            central = weighter(wp, eta, pt, pass_puid, has_gen_jet, "mis")
            weight = weight * central
            if self.config["compute_systematics"]:
                systematics["jet_puid_mis"] = self.compute_puid_sys(
                    weight, weighter, wp, eta, pt, pass_puid,
                    has_gen_jet, "mis")
        return weight, systematics

    def jet_pt_requirement(self, data):
        """Require jets with minimum pT threshold."""
        n = np.zeros(len(data))
        # This assumes jets are ordered by pt highest first
        for i, pt_min in enumerate(self.config["jet_pt_min"]):
            mask = ak.num(data["Jet"]) > i
            n[mask] += np.asarray(pt_min < data["Jet"].pt[mask, i]).astype(int)
        return n >= self.config["jet_pt_num_satisfied"]

    def compute_btag_sys(self, central, up_name, down_name, weighter, wp, flav,
                         eta, pt, discr, efficiency):
        """b tagging systematic weights"""
        up = weighter(wp, flav, eta, pt, discr, up_name, efficiency)
        down = weighter(wp, flav, eta, pt, discr, down_name, efficiency)
        return (up / central, down / central)

    def compute_weight_btag(self, data, efficiency="central", never_sys=False):
        """Compute event weights and systematics, if requested, for the b
        tagging"""
        jets = data["Jet"]
        wp = self.config["btag"].split(":", 1)[1]
        flav = jets["hadronFlavour"]
        eta = jets.eta
        pt = jets.pt
        discr = jets["btag"]
        weight = np.ones(len(data))
        systematics = {}
        for i, weighter in enumerate(self.config["btag_sf"]):
            central = weighter(wp, flav, eta, pt, discr, "central", efficiency)
            if not never_sys and self.config["compute_systematics"]:
                if "btag_splitting_scheme" in self.config:
                    scheme = self.config["btag_splitting_scheme"].lower()
                elif ("split_btag_year_corr" in self.config and
                        self.config["split_btag_year_corr"]):
                    scheme = "years"
                else:
                    scheme = None
                if scheme is None:
                    light_unc_splits = heavy_unc_splits = {"": ""}
                elif scheme == "years":
                    light_unc_splits = heavy_unc_splits = \
                        {"corr": "_correlated", "uncorr": "_uncorrelated"}
                elif scheme == "sources":
                    heavy_unc_splits = {name: f"_{name}"
                                        for name in weighter.sources}
                    light_unc_splits = {"corr": "_correlated",
                                        "uncorr": "_uncorrelated"}
                else:
                    raise ValueError(
                        f"Invalid btag uncertainty scheme {scheme}")

                for name, split in heavy_unc_splits.items():
                    systematics[f"btagsf{i}" + name] = self.compute_btag_sys(
                        central, "heavy up" + split, "heavy down" + split,
                        weighter, wp, flav, eta, pt, discr, efficiency)
                for name, split in light_unc_splits.items():
                    systematics[f"btagsf{i}light" + name] = \
                        self.compute_btag_sys(
                            central, "light up" + split, "light down" + split,
                            weighter, wp, flav, eta, pt, discr, efficiency)
            weight = weight * central
        if never_sys:
            return weight
        else:
            return weight, systematics

    def scale_systematics_for_btag(self, selector, variation, dsname):
        """Modifies factors in the systematic table to account for differences
        in b-tag efficiencies. This is only done for variations the efficiency
        ROOT file contains a histogram with the name of the variation."""
        if "btag_sf" not in self.config or len(self.config["btag_sf"]) == 0:
            return
        available = set.intersection(
            *(w.available_efficiencies for w in self.config["btag_sf"]))
        data = selector.data
        systematics = selector.systematics
        central = self.compute_weight_btag(data, never_sys=True)
        if (variation == self.get_jetmet_nominal_arg()
                and dsname not in self.config["dataset_for_systematics"]):
            for name in ak.fields(systematics):
                if name == "weight":
                    continue
                if name not in available:
                    continue
                sys = systematics[name]
                varied_sf = self.compute_weight_btag(data, name, True)
                selector.set_systematic(name, sys / central * varied_sf)
        elif dsname in self.config["dataset_for_systematics"]:
            name = self.config["dataset_for_systematics"][dsname][1]
            if name in available:
                sys = systematics[name]
                varied_sf = self.compute_weight_btag(data, name, True)
                selector.set_systematic("weight", sys / central * varied_sf)
        elif variation.name in available:
            sys = systematics[name]
            varied_sf = self.compute_weight_btag(data, variation.name, True)
            selector.set_systematic("weight", sys / central * varied_sf)

    def btag_cut(self, is_mc, data):
        """Select events with minimum number of b-tagegd jets."""
        num_btagged = ak.sum(data["Jet"]["btagged"], axis=1)
        accept = np.asarray(num_btagged >= self.config["num_atleast_btagged"])
        if is_mc and (
                "btag_sf" in self.config and len(self.config["btag_sf"]) != 0):
            weight, systematics = self.compute_weight_btag(data[accept])
            accept = accept.astype(float)
            accept[accept.astype(bool)] *= np.asarray(weight)
            return accept, systematics
        else:
            return accept

