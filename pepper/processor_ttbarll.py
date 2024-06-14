import sys
from functools import partial
import numpy as np
import awkward as ak
import logging
from copy import copy

import pepper
import pepper.config


logger = logging.getLogger(__name__)


class Processor(pepper.ProcessorBasicPhysics):
    """Processor for the Top-pair with dileptonic decay selection"""

    config_class = pepper.ConfigTTbarLL

    def __init__(self, config, eventdir):
        super().__init__(config, eventdir)

    def _check_config_integrity(self, config):
        """Check integrity of configuration file."""

        super()._check_config_integrity(config)

        if "lumimask" not in config:
            logger.warning("No lumimask specified")

        if "pileup_reweighting" not in config:
            logger.warning("No pileup reweigthing specified")

        if ("electron_sf" not in config
                or len(config["electron_sf"]) == 0):
            logger.warning("No electron scale factors specified")

        if "muon_sf" not in config or len(config["muon_sf"]) == 0:
            logger.warning("No muon scale factors specified")

        if "btag_sf" not in config or len(config["btag_sf"]) == 0:
            logger.warning("No btag scale factor specified")

        if ("jet_uncertainty" not in config and config["compute_systematics"]):
            logger.warning("No jet uncertainty specified")

        if ("jet_resolution" not in config or "jet_ressf" not in config):
            logger.warning("No jet resolution or no jet resolution scale "
                           "factor specified. This is necessary for "
                           "smearing, even if not computing systematics")
        if "jet_correction_mc" not in config and (
                ("jet_resolution" in config and "jet_ressf" in config) or
                ("reapply_jec" in config and config["reapply_jec"])):
            raise pepper.config.ConfigError(
                "Need jet_correction_mc for propagating jet "
                "smearing/variation to MET or because reapply_jec is true")
        if ("jet_correction_data" not in config and "reapply_jec" in config
                and config["reapply_jec"]):
            raise pepper.config.ConfigError(
                "Need jet_correction_data because reapply_jec is true")

        if "muon_rochester" not in config:
            logger.warning("No Rochster corrections for muons specified")

        if "jet_puid_sf" not in config:
            logger.warning("No jet PU ID SFs specified")

        if ("reco_algorithm" in config and "reco_info_file" not in config):
            raise pepper.config.ConfigError(
                "Need reco_info_file for kinematic reconstruction")

        if "pdf_types" not in config and config["compute_systematics"]:
            logger.warning(
                "pdf_type not specified; will not compute pdf "
                "uncertainties. (Options are 'Hessian', 'MC' and "
                "'MC_Gaussian')")

        # Skip if no systematics, as currently only checking syst configs
        if not config["compute_systematics"]:
            return
        inv_datasets_for_systematics = {}
        dataset_for_systematics = config["dataset_for_systematics"]
        for sysds, (replaceds, variation) in dataset_for_systematics.items():
            if sysds not in config["mc_datasets"]:
                raise pepper.config.ConfigError(
                    "Got systematic dataset that is not mentioned in "
                    f"mc_datasets: {sysds}")
            if replaceds not in config["mc_datasets"]:
                raise pepper.config.ConfigError(
                    "Got dataset to be replaced by a systematic dataset and "
                    f"that is not mentioned in mc_datasets: {replaceds}")
            if (replaceds, variation) in inv_datasets_for_systematics:
                prevds = inv_datasets_for_systematics[(replaceds, variation)]
                raise pepper.config.ConfigError(
                    f"{replaceds} already being replaced for {variation} by "
                    f"{prevds} but is being repeated with {sysds}")
            inv_datasets_for_systematics[(replaceds, variation)] = sys

        if "crosssection_uncertainty" in config:
            xsuncerts = config["crosssection_uncertainty"]
            for dsname in xsuncerts.keys():
                if dsname not in config["mc_datasets"]:
                    raise pepper.config.ConfigError(
                        f"{dsname} in crosssection_uncertainty but not in "
                        "mc_datasets")

        for dsname in config["mc_datasets"].keys():
            if dsname not in config["mc_lumifactors"]:
                raise pepper.config.ConfigError(
                    f"{dsname} is not in mc_lumifactors")

        for dsname in config["exp_datasets"].keys():
            if dsname not in config["dataset_trigger_map"]:
                raise pepper.config.ConfigError(
                    f"{dsname} is not in dataset_trigger_map")
            if isinstance(config["dataset_trigger_order"], dict):
                trigorder = set()
                for datasets in config["dataset_trigger_order"].values():
                    trigorder |= set(datasets)
            else:
                trigorder = config["dataset_trigger_order"]
            if dsname not in trigorder:
                raise pepper.config.ConfigError(
                    f"{dsname} is not in dataset_trigger_order")

        if "drellyan_sf" not in config:
            logger.warning("No Drell-Yan scale factor specified")

        if "trigger_sfs" not in config:
            logger.warning("No trigger scale factors specified")

    def is_dy_dataset(self, key):
        if "DY_datasets" in self.config:
            return key in self.config["DY_datasets"]
        else:
            return key.startswith("DY")

    def process_selection(self, selector, dsname, is_mc, filler):
        era = self.get_era(selector.data, is_mc)
        if dsname.startswith("TTTo"):
            selector.set_column("gent_lc", self.gentop, lazy=True)
            if "top_pt_reweighting" in self.config:
                selector.add_cut(
                    "TopPtReweighting", self.do_top_pt_reweighting,
                    no_callback=True)
        if is_mc and "pileup_reweighting" in self.config:
            selector.add_cut("PileupReweighting", partial(
                self.do_pileup_reweighting, dsname))
        if self.config["compute_systematics"] and is_mc:
            self.add_generator_uncertainies(dsname, selector)
        if is_mc:
            selector.add_cut(
                "CrossSection", partial(self.crosssection_scale, dsname))

        if "blinding_denom" in self.config:
            selector.add_cut("Blinding", partial(self.blinding, is_mc))
        selector.add_cut("Lumi", partial(self.good_lumimask, is_mc, dsname))

        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.config["dataset_trigger_map"],
            self.config["dataset_trigger_order"], era=era)
        selector.add_cut("Trigger", partial(
            self.passing_trigger, pos_triggers, neg_triggers))
        if is_mc and self.config["year"] in ("2016", "2017", "ul2016pre",
                                             "ul2016post", "ul2017"):
            selector.add_cut("L1Prefiring", self.add_l1_prefiring_weights)

        selector.add_cut("METFilters", partial(self.met_filters, is_mc))

        selector.add_cut("NoAddLeps",
                         partial(self.no_additional_leptons, is_mc))
        selector.set_column("Electron", self.pick_electrons)
        selector.set_column("Muon", self.pick_muons)
        selector.set_column("Lepton", partial(
            self.build_lepton_column, is_mc, selector.rng))
        # Wait with hists filling after channel masks are available
        selector.add_cut("AtLeast2Leps", partial(self.lepton_pair, is_mc),
                         no_callback=True)
        selector.set_cat("channel", {"is_ee", "is_em", "is_mm"})
        selector.set_multiple_columns(self.channel_masks)

        selector.set_column("dilep_pt", self.dilep_pt, lazy=True)

        selector.applying_cuts = False

        selector.add_cut("OppositeSign", self.opposite_sign_lepton_pair)
        selector.add_cut("ChnTrigMatch",
                         partial(self.channel_trigger_matching, era))
        if "trigger_sfs" in self.config and is_mc:
            selector.add_cut(
                "TriggerSFs", partial(self.apply_trigger_sfs, dsname))
        selector.add_cut("ReqLepPt", self.lep_pt_requirement)
        selector.add_cut("Mll", self.good_mass_lepton_pair)
        selector.add_cut("ZWindow", self.z_window,
                         categories={"channel": ["is_ee", "is_mm"]})

        if (is_mc and self.config["compute_systematics"]
                and dsname not in self.config["dataset_for_systematics"]):
            if hasattr(filler, "sys_overwrite"):
                assert filler.sys_overwrite is None
            for variarg in self.get_jetmet_variation_args():
                selector_copy = copy(selector)
                filler.sys_overwrite = variarg.name
                self.process_selection_jet_part(selector_copy, is_mc,
                                                variarg, dsname, filler, era)
                if self.eventdir is not None:
                    logger.debug(f"Saving per event info for variation"
                                 f" {variarg.name}")
                    self.save_per_event_info(
                        dsname + "_" + variarg.name, selector_copy, False)
            filler.sys_overwrite = None

        # Do normal, no-variation run
        self.process_selection_jet_part(selector, is_mc,
                                        self.get_jetmet_nominal_arg(),
                                        dsname, filler, era)
        logger.debug("Selection done")

    def process_selection_jet_part(self, selector, is_mc, variation, dsname,
                                   filler, era):
        """Part of the selection that needs to be repeated for
        every systematic variation done for the jet energy correction,
        resultion and for MET"""
        logger.debug(f"Running jet_part with variation {variation.name}")
        reapply_jec = ("reapply_jec" in self.config
                       and self.config["reapply_jec"])
        selector.set_multiple_columns(partial(
            self.compute_jet_factors, is_mc, era, reapply_jec, variation.junc,
            variation.jer, selector.rng))
        selector.set_column("OrigJet", selector.data["Jet"])
        selector.set_column("Jet", partial(self.build_jet_column, is_mc))
        if "jet_puid_sf" in self.config and is_mc:
            selector.add_cut("JetPUIdSFs", self.jet_puid_sfs)
        selector.set_column("Jet", self.jets_with_puid)
        smear_met = "smear_met" in self.config and self.config["smear_met"]
        selector.set_column(
            "MET", partial(self.build_met_column, is_mc, variation.junc,
                           variation.jer if smear_met else None, selector.rng,
                           era, variation=variation.met))
        selector.set_multiple_columns(
            partial(self.drellyan_sf_columns, selector))
        if "drellyan_sf" in self.config and is_mc:
            selector.add_cut("DYScale", partial(self.apply_dy_sfs, dsname))
        selector.add_cut("HasJets", self.has_jets)
        if (self.config["hem_cut_if_ele"] or self.config["hem_cut_if_muon"]
                or self.config["hem_cut_if_jet"]):
            selector.add_cut("HEMCut", self.hem_cut)
        selector.add_cut("JetPtReq", self.jet_pt_requirement)
        if is_mc and self.config["compute_systematics"]:
            self.scale_systematics_for_btag(selector, variation, dsname)
        selector.add_cut("HasBtags", partial(self.btag_cut, is_mc))
        selector.add_cut("ReqMET", self.met_requirement,
                         categories={"channel": ["is_ee", "is_mm"]})

        if "reco_algorithm" in self.config:
            reco_alg = self.config["reco_algorithm"]
            selector.set_column("recolepton", self.pick_lepton_pair,
                                all_cuts=True)
            selector.set_column("recob", self.pick_bs_from_lepton_pair,
                                all_cuts=True)
            selector.set_column("recot", partial(
                self.ttbar_system, reco_alg.lower(), selector.rng),
                all_cuts=True, no_callback=True)
            selector.add_cut("Reco", self.has_ttbar_system)
            selector.set_column("reconu", self.build_nu_column_ttbar_system,
                                all_cuts=True, lazy=True)
            selector.set_column("dark_pt", self.calculate_dark_pt,
                                all_cuts=True, lazy=True)
            selector.set_column("chel", self.calculate_chel, all_cuts=True,
                                lazy=True)

    def channel_masks(self, data):
        """Get the channel masks (bool arrays) for ee, eµ and µµ decays"""
        leps = data["Lepton"]
        channels = {}
        channels["is_ee"] = ((abs(leps[:, 0].pdgId) == 11)
                             & (abs(leps[:, 1].pdgId) == 11))
        channels["is_mm"] = ((abs(leps[:, 0].pdgId) == 13)
                             & (abs(leps[:, 1].pdgId) == 13))
        channels["is_em"] = (~channels["is_ee"]) & (~channels["is_mm"])
        return channels

    def pick_lepton_pair(self, data):
        """Get one pair of leptons of opposite charge per event. The negative
        lepton comes first"""
        return data["Lepton"][
            ak.argsort(data["Lepton"]["pdgId"], ascending=False)]

    def dilep_pt(self, data):
        """pt of the lepton pair system"""
        return (data["Lepton"][:, 0] + data["Lepton"][:, 1]).pt

    def apply_trigger_sfs(self, dsname, data):
        """Apply the weights due to differences of the triggers between
        data and simulation"""
        leps = data["Lepton"]
        ones = np.ones(len(data))
        central = ones
        channels = ["is_ee", "is_em", "is_mm"]
        trigger_sfs = self.config["trigger_sfs"]
        for channel in channels:
            sf = trigger_sfs[channel](lep1_pt=leps[:, 0].pt,
                                      lep2_pt=leps[:, 1].pt)
            central = ak.where(data[channel], sf, central)
        if self.config["compute_systematics"]:
            up = ones
            down = ones
            for channel in channels:
                sf = trigger_sfs[channel](lep1_pt=leps[:, 0].pt,
                                          lep2_pt=leps[:, 1].pt,
                                          variation="up")
                up = ak.where(data[channel], sf, up)
                sf = trigger_sfs[channel](lep1_pt=leps[:, 0].pt,
                                          lep2_pt=leps[:, 1].pt,
                                          variation="down")
                down = ak.where(data[channel], sf, down)
            return central, {"triggersf": (up / central, down / central)}
        return central

    def apply_dy_sfs(self, dsname, data):
        """Apply scale factors to compensate mismodeling of Drell-Yan"""
        if self.is_dy_dataset(dsname):
            channel = ak.where(data["is_ee"], 0, ak.where(data["is_em"], 1, 2))
            if ("bin_dy_sfs" in self.config and
                    self.config["bin_dy_sfs"] is not None):
                params = {
                    "channel": channel, "axis":
                    pepper.hist_defns.DataPicker(self.config["bin_dy_sfs"])}
            else:
                params = {"channel": channel}
            dysf = self.config["drellyan_sf"]
            central = dysf(**params)
            if self.config["compute_systematics"]:
                up = dysf(**params, variation="up")
                down = dysf(**params, variation="down")
                return central, {"DYsf": (up / central, down / central)}
            return central
        elif self.config["compute_systematics"]:
            ones = np.ones(len(data))
            return np.full(len(data), True), {"DYsf": (ones, ones)}
        return np.full(len(data), True)

    def drellyan_sf_columns(self, data, selector):
        """Dummy function, overwritten when computing DY SFs"""
        return {}

    def channel_trigger_matching(self, era, data):
        is_ee = data["is_ee"]
        is_mm = data["is_mm"]
        is_em = data["is_em"]
        triggers = self.config["channel_trigger_map"]

        ret = np.full(len(data), False)
        check = [
            (is_ee, "ee"), (is_mm, "mumu"), (is_em, "emu"),
            (is_ee | is_em, "e"), (is_mm | is_em, "mu")]
        for mask, channel in check:
            if (channel + "_" + era) in triggers:
                trigger = [pepper.misc.normalize_trigger_path(t)
                           for t in triggers[channel + "_" + era]]
                ret = ret | (
                    mask & self.passing_trigger(trigger, [], data))
            elif channel in triggers:
                trigger = [pepper.misc.normalize_trigger_path(t)
                           for t in triggers[channel]]
                ret = ret | (
                    mask & self.passing_trigger(trigger, [], data))

        return ret

    def met_requirement(self, data):
        """Require a minimum MET"""
        met = data["MET"].pt
        return met > self.config["ee/mm_min_met"]

    def build_nu_column(self, data):
        """Get four momenta for the neutrinos coming from top pair decay"""
        lep = data["recolepton"][:, 0:1]
        antilep = data["recolepton"][:, 1:2]
        b = data["recob"][:, 0:1]
        antib = data["recob"][:, 1:2]
        top = data["recot"][:, 0:1]
        antitop = data["recot"][:, 1:2]
        nu = top - b - antilep
        antinu = antitop - antib - lep
        return ak.concatenate([nu, antinu], axis=1)

    def calculate_dark_pt(self, data):
        """Get the pt of the four vector difference of the MET and the
        neutrinos"""
        nu = data["reconu"][:, 0]
        antinu = data["reconu"][:, 1]
        met = data["MET"]
        return met - nu - antinu

    def pick_bs_from_lepton_pair(self, data):
        """Pick a bottom quark and a bottom antiquark that fit best to a pair
        of leptons assuming they come from a top pair decay. This is using
        the mlb histogram method"""
        recolepton = data["recolepton"]
        lep = recolepton[:, 0]
        antilep = recolepton[:, 1]
        # Build a reduced jet collection to avoid loading all branches and
        # make make this function faster overall
        columns = ["pt", "eta", "phi", "mass", "btagged"]
        jets = ak.with_name(data["Jet"][columns], "PtEtaPhiMLorentzVector")
        btags = jets[data["Jet"].btagged]
        jetsnob = jets[~data["Jet"].btagged]
        num_btags = ak.num(btags)
        b0, b1 = ak.unzip(ak.where(
            num_btags > 1, ak.combinations(btags, 2),
            ak.where(
                num_btags == 1, ak.cartesian([btags, jetsnob]),
                ak.combinations(jetsnob, 2))))
        bs = ak.concatenate([b0, b1], axis=1)
        bs_rev = ak.concatenate([b1, b0], axis=1)
        mass_alb = reduce(
            lambda a, b: a + b, ak.unzip(ak.cartesian([bs, antilep]))).mass
        mass_lb = reduce(
            lambda a, b: a + b, ak.unzip(ak.cartesian([bs_rev, lep]))).mass
        with uproot.open(self.config["reco_info_file"]) as f:
            mlb_prob = pepper.scale_factors.ScaleFactors.from_hist(f["mlb"])
        p_m_alb = mlb_prob(mlb=mass_alb)
        p_m_lb = mlb_prob(mlb=mass_lb)
        bestbpair_mlb = ak.unflatten(
            ak.argmax(p_m_alb * p_m_lb, axis=1), np.full(len(bs), 1))
        return ak.concatenate([bs[bestbpair_mlb], bs_rev[bestbpair_mlb]],
                              axis=1)

    def ttbar_system(self, reco_alg, rng, data):
        """Do ttbar reconstruction, obtaining four vectors for top pairs
        from their decay products"""
        lep = data["recolepton"][:, 0]
        antilep = data["recolepton"][:, 1]
        b = data["recob"][:, 0]
        antib = data["recob"][:, 1]
        met = data["MET"]
        if reco_alg == "sonnenschein":
            if self.config["reco_num_smear"] is None:
                energyfl = energyfj = 1
                alphal = alphaj = 0
                num_smear = 1
                mlb = None
            else:
                with uproot.open(self.config["reco_info_file"]) as f:
                    energyfl = f["energyfl"]
                    energyfj = f["energyfj"]
                    alphal = f["alphal"]
                    alphaj = f["alphaj"]
                    mlb = f["mlb"]
                num_smear = self.config["reco_num_smear"]
            if isinstance(self.config["reco_w_mass"], (int, float)):
                mw = self.config["reco_w_mass"]
            else:
                with uproot.open(self.config["reco_info_file"]) as f:
                    mw = f[self.config["reco_w_mass"]]
            if isinstance(self.config["reco_t_mass"], (int, float)):
                mt = self.config["reco_t_mass"]
            else:
                with uproot.open(self.config["reco_info_file"]) as f:
                    mt = f[self.config["reco_t_mass"]]
            top, antitop = sonnenschein(
                lep, antilep, b, antib, met, mwp=mw, mwm=mw, mt=mt, mat=mt,
                energyfl=energyfl, energyfj=energyfj, alphal=alphal,
                alphaj=alphaj, hist_mlb=mlb, num_smear=num_smear, rng=rng)
            return ak.concatenate([top, antitop], axis=1)
        elif reco_alg == "betchart":
            top, antitop = betchart(lep, antilep, b, antib, met)
            return ak.concatenate([top, antitop], axis=1)
        else:
            raise ValueError(f"Invalid value for reco algorithm: {reco_alg}")

    def has_ttbar_system(self, data):
        """Whether the recontruction of the ttbar system was successful"""
        return ak.num(data["recot"]) > 0

    def build_nu_column_ttbar_system(self, data):
        """Get four momenta for the neutrinos coming from top pair decay"""
        lep = data["recolepton"][:, 0:1]
        antilep = data["recolepton"][:, 1:2]
        b = data["recob"][:, 0:1]
        antib = data["recob"][:, 1:2]
        top = data["recot"][:, 0:1]
        antitop = data["recot"][:, 1:2]
        nu = top - b - antilep
        antinu = antitop - antib - lep
        return ak.concatenate([nu, antinu], axis=1)

    def calculate_chel(self, data):
        """Calculate the angle between the leptons in their helicity frame"""
        top = data["recot"]
        lep = data["recolepton"]
        ttbar_boost = -top.sum().boostvec
        top = top.boost(ttbar_boost)
        lep = lep.boost(ttbar_boost)

        top_boost = -top.boostvec
        lep_ZMFtbar = lep[:, 0].boost(top_boost[:, 1])
        lbar_ZMFtop = lep[:, 1].boost(top_boost[:, 0])

        chel = lep_ZMFtbar.dot(lbar_ZMFtop) / lep_ZMFtbar.rho / lbar_ZMFtop.rho
        return chel