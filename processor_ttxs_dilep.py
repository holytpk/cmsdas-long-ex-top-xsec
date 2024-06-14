from functools import partial
import numpy as np
import awkward as ak
import logging
from copy import copy

import pepper
from pepper.processor_basic import VariationArg

logger = logging.getLogger(__name__)

class ProcessorTTXS(pepper.ProcessorTTbarLL):

    def _check_config_integrity(self, config):
        """Check integrity of configuration file."""

        super()._check_config_integrity(config)

    def process_selection(self, selector, dsname, is_mc, filler):
        era = self.get_era(selector.data, is_mc)

        if not is_mc:
            selector.add_cut("Era", partial(self.has_era, era), no_callback=True)

        if dsname.startswith("TTTo"):
            selector.set_column("gent_lc", self.gentop, lazy=True)
            if "top_pt_reweighting" in self.config:
                selector.add_cut(
                    "Top pt reweighting", self.do_top_pt_reweighting,
                    no_callback=False)
       
        if is_mc and "pileup_reweighting" in self.config:
            selector.add_cut("PileupReweighting", partial(
                self.do_pileup_reweighting, dsname))
                        
        if self.config["compute_systematics"] \
            and self.config["do_generator_uncertainties"] and is_mc:
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
        selector.add_cut("OppositeSign", self.opposite_sign_lepton_pair)

        selector.add_cut("ChanTrigMatch",
                         partial(self.channel_trigger_matching, era))
        if "trigger_sfs" in self.config and is_mc:
            selector.add_cut(
                "Trigger SFs", partial(self.apply_trigger_sfs, dsname))
        elif "trigger_sfs_tnp" in self.config and is_mc:
            selector.add_cut("TriggerSFs", self.apply_trigger_sfs_tnp_sl)

        selector.add_cut("ReqLepPT", self.lep_pt_requirement)

        if (is_mc and self.config["compute_systematics"]
                and self.config["do_jet_variations"]
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
        logger.debug(f"Running jet_part with variation {variation.name}")

        reapply_jec = ("reapply_jec" in self.config
                       and self.config["reapply_jec"])
        selector.set_multiple_columns(partial(
            self.compute_jet_factors, is_mc, era, reapply_jec, variation.junc,
            variation.jer, selector.rng))
        selector.set_column("OrigJet", selector.data["Jet"])
        selector.set_column("Jet", partial(self.build_jet_column, is_mc))
        selector.set_column("nbtag", self.num_btags)

        selector.set_multiple_columns(partial(self.btag_categories, selector))

        selector.add_cut("HasJets", self.has_jets)
        
        selector.add_cut("HasBtags", partial(self.btag_cut, is_mc))

    def has_era(self, era, data):
        if era == "no_era":
            return np.full(len(data), False)
        else:
            return np.full(len(data), True)

    def num_btags(self, data):
        jets = data["Jet"]
        nbtags = ak.sum(jets["btagged"], axis=1)
        return ak.where(ak.num(jets) > 0, nbtags, 0)


    def btag_categories(self, selector, data):
        cats = {}
        num_btagged = data["nbtag"]

        cats["b1"] = (num_btagged == 1)
        cats["b2+"] = (num_btagged >= 2)

        selector.set_cat("btags", {"b1", "b2+"})

        return cats

    def apply_trigger_sfs_tnp_sl(self, data):
        leps = data["Lepton"]
        is_dl = ak.num(leps) > 1

        trigger_sfs = self.config["trigger_sfs_tnp"]

        abseta = abs(leps.eta)
        pt = leps.pt

        sources = ["EleMC","EleData","MuMC","MuData"]
        vars = [("nominal", None)]
        sfs_vars = {}
        if self.config["compute_systematics"]:
            vars.extend([(var,d) for var in sources for d in ["up", "down"]])

        for var, d in vars:

            effs_e_mc = trigger_sfs[("e", "eff_mc")](abseta=abseta, pt=pt, variation=(d if var == "EleMC" else "central"))
            effs_e_data = trigger_sfs[("e", "eff_data")](abseta=abseta, pt=pt, variation=(d if var == "EleData" else "central"))
            effs_m_mc = trigger_sfs[("mu", "eff_mc")](abseta=abseta, pt=pt, variation=(d if var == "MuMC" else "central"))
            effs_m_data = trigger_sfs[("mu", "eff_data")](abseta=abseta, pt=pt, variation=(d if var == "MuData" else "central"))

            effs_mc = ak.where(abs(leps.pdgId) == 11, effs_e_mc, effs_m_mc)
            eff_event_mc_dl = ak.sum(effs_mc, axis=1) - ak.prod(effs_mc, axis=1)
            eff_event_mc_sl = effs_mc[:,0]
            eff_event_mc = ak.where(is_dl, eff_event_mc_dl, eff_event_mc_sl)
                
            effs_data = ak.where(abs(leps.pdgId) == 11, effs_e_data, effs_m_data)
            
            eff_event_data_dl = ak.sum(effs_data, axis=1) - ak.prod(effs_data, axis=1)
            eff_event_data_sl = effs_data[:,0]
            eff_event_data = ak.where(is_dl, eff_event_data_dl, eff_event_data_sl)
        
            sfs_event = eff_event_data / eff_event_mc
            sfs_vars[(var, d)] = sfs_event

        sfs_nom = sfs_vars[("nominal", None)]
        if self.config["compute_systematics"]:
            systs = {f"triggersf{var}": (sfs_vars[(var, "up")] / sfs_nom, sfs_vars[(var, "down")] / sfs_nom) for var in sources}
            return sfs_nom, systs
        else:
            return sfs_nom

if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(
        ProcessorTTXS,
        "Select events from nanoAODs using the TTXS processor."
        "This will save cutflows, histograms and, if wished, per-event data. "
        "Histograms are saved in a Coffea format and are ready to be plotted "
        "by plot_control.py")