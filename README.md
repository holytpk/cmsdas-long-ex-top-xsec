# CMSDAS CERN 2024 instructions

Please follow the instructions here if you are part of this school: [cmsdas-2024-ttxs.docs.cern.ch](https://cmsdas-2024-ttxs.docs.cern.ch/)

Installation guidelines are part of the "pre-exercise" section.

## Pepper

Most of the standard pepper README is repeated below, which is useful in understanding the general functionality of the framework. 

## Usage -->

### Getting started

In Pepper an analysis is implemented as a Processor class. A short example of such a Processor with many explanatory comments can be found in [here](example/example_processor.py). This processor can be run by executing `python3 -m pepper.runproc example_processor.py example_config.json` (when inside the example directory). Also running `python -m pepper.runproc -h` will show the available command line options.

### Running on HTCondor
In order to run on HTCondor using `pepper.runproc`, one only has to specify the `--condor` parameter followed by the number of jobs desired. Events will be split across jobs as evenly as possible.
To control which environment is employed on the HTCondor node, the parameter `--condorinit` can be used. `--condorinit` should point to a Shell script that can be sourced setting up the environment. If `--condorinit` is not present, Pepper will instead use the script that is pointed at by the local environment variable `PEPPER_CONDOR_ENV`. If this is also not set, the jobs will be run in the default environment of your HTCondor system. For an example environment script setting up LCG on CentOS7 see [here](example/environment.sh).

When running on HTCondor, the local process, that has started also the jobs, needs to be kept open until everything has finished. In order to not accidentally kill the process by an unstable connection or similar, it is recommended to run it inside a `byobu`, `tmux` or `screen` session, or to prepend the command `nohup`.

A directory with logs from the jobs will be present under `pepper_logs`. Directories inside `pepper_logs` are numbered, the highest number is the one of the latest run. The log level of the logs inside is controller via the `--loglevel` option. Set it to `debug` to get full logging.

### XRootD usage
Pepper is able to access data sets from remote servers using XRootD. You should specify `"file_mode": "local+xrootd"` and `"xrootddomain": "xrootd-cms.infn.it"` in your config to enable it. There are four requirements to get it working (also in conjunction with HTCondor):
 - xrootd must be installed. You can check with `python3 -m pip show xrootd`
 - The CMS Grid environment needs to be sourced (also inside the HTCondor job). This is done by the [example environment script](example/environment.sh).
 - The environment variable `X509_USER_PROXY` needs to be set to a file path accessible by Condor (/tmp/ is not). As above this is also needed inside the Condor job and is cone by the script.
 - A VOMS proxy needs to be created at the path pointed to by `X509_USER_PROXY`. To do this please once run: `voms-proxy-init --voms cms --out $X509_USER_PROXY`.

If you get the error 'sslv3 alert certificate expired', please run the voms-proxy-init command again.

### Main scripts
The scripts directory of this repository contains several helper scripts to obtain inputs and plot outputs:

 - `calulate_stitching_factors.py`: Calculate factors to stitch MC samples from an already produced histogram (currently only 1D histograms supported)
 - `compute_mc_lumifactors.py`: Compute <img align="top" src="https://latex.codecogs.com/gif.latex?{\cal L}\sigma/\sum w_{\mathrm{gen}}" />, the factors needed to scale MC to data
 - `compute_pileup_weights.py`: Compute scale factors for pileup reweighting
 - `delete_duplicate_outputs.py`: Check for duplication in the per event data produced by select_events.py, and move or delete any duplicates
 - `export_hists_from_state.py`: Save all histograms contained in a Pepper processor state, even if processing of all data hasn't been finished yet
 - `generate_btag_efficiencies.py`: Generate a ROOT file containing efficiency histograms needed for b-tagging scale factors
 - `generate_jet_puid_efficiencies`: Generate a ROOT file containing efficiency histograms needed for computing jet pile-up ID scale factors
 - `get_bad_local_files.py`: Find NanoAOD files that exist in the store directory but are not accessible, possibly due to technical issues
 - `hdf5_to_ttree.py`: Merge and convert Pepper HDF5 files to Root files containing TTrees
 - `merge_hists.py`: Caluclate weighted average of two SF histograms
 - `plot_histograms.py`: Plot histograms in a ratioplot style
 - `produce_met_xy_nums.py`: Convert MET-xy correction numbers from the C++ headers provided centrally to json files
 - `rucio_create_rules.py`: Creates Rucio rules for all data sets specified in a Pepper config. Once the rules are approved, the data sets will be transfered to the local site.
 - `ttbarll_dy_sf_calculate.py`: Calculate scale factors for DY reweighting from the output of ttbarll_dy_sf_produce.py
 - `ttbarll_dy_sf_produce.py`: Produce the numbers needed for DY SF calculation
 - `ttbarll_kinreco_hists_produce.py`: Generate histograms needed for top-quark kinematic reconstruction
 - `ttbarll_select_events.py`: Run the main ttbarll analysis procedure, outputting histograms and per event data
 - `ttbarll_trigger_sf_calculate.py`: Calculate SFs for ttbar dileptonic triggers using the output of ttbarll_trigger_sf_produce.py by cross-trigger method
 - `ttbarll_trigger_sf_produce.py`: Produce the numbers needed for trigger SF calculation



## Configuration
Configuration is done via JSON files. Examples can be found in the `example` directory. Additional data needed for configuration, for example scale factors and cross sections, can be found in a separate data repository here https://gitlab.cern.ch/pepper/data
After downloading it, make sure to set the configuration variable "datadir" to the path where the data repository was downloaded.
For a detailed explanation on the configuration variables, see `config_documentation.md`.



## Contributing
Feel free to submit merge requests to have your code included in this repository! Your code must comply with pep8. You can check this by running the following inside the Pepper directory:
```sh
python3 -m pip install .[dev] --user
python3 -m flake8
```
