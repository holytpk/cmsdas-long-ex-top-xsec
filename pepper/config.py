#!/usr/bin/env python3

import hjson
import os
import logging
from collections.abc import MutableMapping

import pepper
from pepper import HistDefinition


logger = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    pass


class Config(MutableMapping):
    def __init__(self, path, textparser=hjson.load, cwd="."):
        """Initialize the configuration.

        Parameters
        ----------
        path
            Path to the file containing the configuration
        textparser
            Callable to be used to parse the text contained in ``path``
        cwd
            Path to use as the working directory for relative paths in the
            config. The actual working directory of the process might change
            e.g. after submitting to HTCondor
        """
        self._config_loaded = None
        self._path = os.path.realpath(path)
        self._cache = {}
        self._overwritten = {}
        self._textparser = textparser
        self._cwd = os.path.realpath(cwd)

        logger.debug("Configuration read")
        if "datadir" in self._config:
            datadir = self._config['datadir']
            if not os.path.exists(datadir):
                raise ConfigError(f"datadir does not exist: {datadir}")
            logger.debug(f"datadir: {datadir}")
        if "configdir" in self._config:
            logger.debug(f"configdir: {self._config['configdir']}")
        if "store" in self._config:
            logger.debug(f"store: {self._config['store']}")

        self.special_vars = {
            "$DATADIR": "datadir",
            "$CONFDIR": "configdir",
            "$STOREDIR": "store",
        }
        self.behaviors = {
            "bad_file_paths": self._get_maybe_external,
            "xrootd_url_blacklist": self._get_maybe_external,
            "hists": self._get_hists
        }

    @property
    def _config(self):
        if self._config_loaded is not None:
            return self._config_loaded
        with open(self._path) as f:
            self._config_loaded = self._textparser(f)
        self._config_loaded["configdir"] = os.path.dirname(self._path)
        return self._config_loaded

    def _replace_special_vars(self, s):
        if isinstance(s, dict):
            return {self._replace_special_vars(key):
                    self._replace_special_vars(val) for key, val in s.items()}
        elif isinstance(s, list):
            return [self._replace_special_vars(item) for item in s]
        elif isinstance(s, str):
            for name, configvar in self.special_vars.items():
                if name in s:
                    if configvar in self._overwritten:
                        configvar = self._overwritten[configvar]
                    elif configvar in self._config:
                        configvar = self._config[configvar]
                    else:
                        configvar = None
                    if configvar is None:
                        raise ConfigError(
                            f"{name} contained in config but {configvar} was "
                            "not specified")
                    s = s.replace(name, configvar)
        return s

    def _get_path(self, value):
        value = self._replace_special_vars(value)
        if not os.path.isabs(value):
            value = os.path.join(self._cwd, value)
        return os.path.realpath(value)

    def _get_maybe_external(self, value):
        if isinstance(value, str):
            with open(self._get_path(value)) as f:
                value = self._textparser(f)
        return value

    def _get_hists(self, value):
        hist_config = self._get_maybe_external(value)
        return {k: HistDefinition(c) for k, c in hist_config.items()}

    def _get(self, key):
        if key in self._overwritten:
            if self._overwritten[key] is None:
                raise KeyError(key)
            else:
                return self._overwritten[key]
        if key in self._cache:
            return self._cache[key]

        if key not in self._config or self._config[key] is None:
            raise KeyError(key)

        if key in self.behaviors:
            value = self.behaviors[key](self._config[key])
            self._cache[key] = value
            return value

        return self._replace_special_vars(self._config[key])

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self._get(k) for k in key]
        else:
            return self._get(key)

    def __contains__(self, key):
        return (
            # __setitem__ and not __delitem__
            self._overwritten.get(key) is not None
            or (
                # In file config and not None
                self._config.get(key) is not None and
                # But only if not __delitem__
                self._overwritten.get(key, True) is not None
            )
        )

    def __setitem__(self, key, value):
        self._overwritten[key] = value

    def __delitem__(self, key):
        if key in self._cache:
            del self._cache[key]
        if ((key in self._config and self._config[key] is not None)
                or (key in self._overwritten
                    and self._overwritten[key] is not None)):
            self._overwritten[key] = None
        else:
            raise KeyError(key)

    def __iter__(self):
        for key in self._config:
            yield key

    def __len__(self):
        return len(self._config)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Do not pickle the cache or loaded config
        state["_cache"] = {}
        state["_config_loaded"] = None
        return state

    def get_datasets(self, include=None, exclude=None, dstype="any",
                     return_inverse=False):
        """Helper method to access mc_datasets and exp_datasets more easily.

        Parameters
        ----------
        include
            List of dataset names to restrict the result to
        exclude
            List of dataset names to exclude from the result
        dstype
            Either 'any', 'mc' or 'data'. 'mc' restrcits the result to be
            from mc_datasets, while 'data' restricts to exp_datasets.
            'any' does not impose restrictions.
        return_inverse
            Additionally return a mapping of path to dataset name

        Returns
        ------
        datasets
            Maps data set names to lists of full paths to the
            dataset's files. If return_inverse is true, also return the inverse
            mapping
        paths2dsname
            Only if ``return_inverse`` is ``True``. Maps full paths to data set
            names. The inverse of ``datasets``
        """
        if dstype not in ("any", "mc", "data"):
            raise ValueError("dstype must be either 'any', 'mc' or 'data'")
        datasets = {s: v for s, v in self["exp_datasets"].items()
                    if v is not None}
        duplicate = set(datasets.keys()) & set(self["mc_datasets"])
        if len(duplicate) > 0:
            raise ConfigError("Got duplicate dataset names: {}".format(
                ", ".join(duplicate)))
        datasets.update({s: v for s, v in self["mc_datasets"].items()
                         if v is not None})
        for dataset in set(datasets.keys()):
            if ((dstype == "mc" and dataset not in self["mc_datasets"])
                    or (dstype == "data"
                        and dataset not in self["exp_datasets"])
                    or (include is not None and dataset not in include)
                    or (exclude is not None and dataset in exclude)):
                del datasets[dataset]
        requested_datasets = datasets.keys()

        mode = self["file_mode"] if "file_mode" in self else "local"
        store = self["store"] if "store" in self else None
        ignore_path = None
        if mode == "local" and "bad_file_paths" in self:
            # In local mode, we want to ignore completely files that
            # are in the bad file list
            skippaths = set(self["bad_file_paths"])

            def is_lfn_in_bad_paths(lfn):
                return not skippaths.isdisjoint(
                    pepper.datasets.resolve_lfn(lfn, self["store"]))
            ignore_path = is_lfn_in_bad_paths
        logger.debug("Finding files for data sets")
        datasets, paths2dsname = pepper.datasets.expand_datasetdict(
            datasets, store=store, mode=mode,
            ignore_path=ignore_path)
        missing_datasets = requested_datasets - datasets.keys()
        if len(missing_datasets) > 0:
            raise ConfigError("Could not find files for: "
                              + ", ".join(missing_datasets))
        if not return_inverse:
            return datasets
        else:
            return datasets, paths2dsname

    def get_paths_for_lfn(self, lfn):
        """Get paths that can be used in ``uproot.open`` for a logical file
        name
        The results depends on the config parameters ``file_mode``, ``store``
        and ``xrootddomain``.

        Parameters
        ----------
        lfn
            The logical file name, if prefixed with "cmslfn://", otherwise
            it is treated a physical file name and included in the return value
            as is

        Returns
        -------
        filepaths
            List of physical file names found for ``lfn``
        """
        store = None
        xrootddomain = None
        xrootd_url_blacklist = None
        skippaths = self["bad_file_paths"] if "bad_file_paths" in self else\
            None
        filemode = self["file_mode"] if "file_mode" in self else "local"
        if filemode == "local" or filemode == "local+xrootd":
            store = self["store"]
        if filemode == "xrootd" or filemode == "local+xrootd":
            xrootddomain = self["xrootddomain"]
            if "xrootd_url_blacklist" in self:
                xrootd_url_blacklist = self["xrootd_url_blacklist"]

        if lfn.startswith("cmslfn://"):
            filepaths = pepper.datasets.resolve_lfn(lfn, store, xrootddomain,
                                                    xrootd_url_blacklist)
        else:
            filepaths = [lfn]
        if skippaths is not None:
            filepaths = [p for p in filepaths if p not in skippaths]

        return filepaths
