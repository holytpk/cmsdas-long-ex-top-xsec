#!/usr/bin/env python3

import numpy as np
import uproot
import hjson

import pepper
from pepper.scale_factors import ScaleFactors


class ConfigTTbarLL(pepper.ConfigBasicPhysics):
    def __init__(self, path_or_file, textparser=hjson.load, cwd="."):
        """Initialize the configuration.

        Parameters
        ----------
        path
            Path to the file containing the configuration
        textparser
            Callable to be used to parse the text contained in path_or_file
        cwd
            Path to use as the working directory for relative paths in the
            config. The actual working directory of the process might change
        """
        super().__init__(path_or_file, textparser, cwd)

        self.behaviors.update(
            {
                "drellyan_sf": self._get_drellyan_sf,
                "trigger_sfs": self._get_trigger_sfs,
                "jes_sf": self._get_jes_sf,
                "pileup_reweighting_exp": self._get_scalefactor,
                "trigger_sfs_tnp": self._get_trigger_sfs_tnp,
                "fakerate_electrons": self._get_scalefactor,
                "fakerate_muons": self._get_scalefactor
            }
        )

    def _get_drellyan_sf(self, value):
        if isinstance(value, list):
            path, histname = value
            with uproot.open(self._get_path(path)) as f:
                hist = f[histname]
            dy_sf = ScaleFactors.from_hist(hist)
        else:
            data = self._get_maybe_external(value)
            dy_sf = ScaleFactors(
                bins=data["bins"],
                factors=np.array(data["factors"]),
                factors_up=np.array(data["factors_up"]),
                factors_down=np.array(data["factors_down"]))
        return dy_sf

    def _get_trigger_sfs(self, value):
        path, histnames = value
        ret = {}
        if len(histnames) != 3:
            raise pepper.config.ConfigError(
                "Need 3 histograms for trigger scale factors. Got "
                f"{len(histnames)}")
        with uproot.open(self._get_path(path)) as f:
            for chn, histname in zip(("is_ee", "is_em", "is_mm"), histnames):
                ret[chn] = ScaleFactors.from_hist(
                    f[histname], dimlabels=["lep1_pt", "lep2_pt"])
        return ret

    def _get_jes_sf(self, value):
        data = self._get_maybe_external(value)
        sf = ScaleFactors(
            bins=data["bins"],
            factors=np.array(data["factors"]),
            factors_up=np.array(data["factors_up"]),
            factors_down=np.array(data["factors_down"]))
        return sf

    def _get_pileup_weights_exp(self, value):
        rootfiles = {k: uproot.open(self._get_path(v)) for k,v in value.items()}
        weighter = PileupWeighterExp(rootfiles)
        for rf in rootfiles.values():
            rf.close()
        return weighter

    def _get_trigger_sfs_tnp(self, value):
        path = value
        ret = {}
        with uproot.open(self._get_path(path)) as f:
            for chn in ["e", "mu"]:
                for typ in ["sf", "eff_data", "eff_mc"]:
                    ret[(chn, typ)] = ScaleFactors.from_hist(
                        f[chn + "_" + typ], dimlabels=["pt", "abseta"])
        return ret

