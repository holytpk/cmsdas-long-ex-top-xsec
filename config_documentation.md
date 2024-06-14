# Pepper configuration
The configuration file uses a JSON format. JSON arrays correspond to Python lists and JSON objects correspond to Python dicts with only strings as keys. A variable with value `Null` in the JSON configuration is equivalent to it not being present. If a variable is supposed to contain a path to a JSON file, it can also directly have the content of said JSON file as value directly. Masses and pT are always in GeV.

## Special variables
Inside the string values of a configuration variable the following placeholders are allowed and a replaced by their values at runtime
- `"$CONFDIR`: Path to the directory where the configuration file is located.
- `"$STOREDIR"`: Path given in the `store` configuration variable.
- `"$DATADIR"`: Path given in the `datadir` configuration variable.

## General
- `year`: String. Indicates the year. Things like b-tag working points depend on this.
- `sqrt_s`: Optional, float. Center-of-mass energy in GeV.
- `rng_seed_file`: Optional, path. A text file to save to and load (if it exists) an integer from which is used as a seed for the random number generator. This seed will be combined with a number unique to the chunk of events being processed, so that the randomness is still different from chunk to chunk.
- `blinding_denom`: Optional, float. Only use `1/blinding_denom` of the data events, MC is scaled accordingly.
- `compute_systematics`: Boolean, if true compute all systematic uncertainties.
- `skip_nonshape_systematics`: Optional, boolean. If false and `compute_systematics` is true, histograms will be generated even for uncertainties like lumi, that aren't shape uncertainties.

## Data sets
- `file_mode`: Optional, a string, either `"local"`, `"xrootd"` or `"local+xrootd"`. Determines how data set names are evaluated. With `"local"` (default) files from data sets will only be searched for and opened from in the subdirectories of `store`. With `"xrootd"` data sets will be looked up online and their files will be accessed via xrootd. With `"local+xrootd"` files will only be looked up in `store` and online and will be accessed via xrootd only if they do not exist in `store`. For paths or glob patterns, this option does not have any effect.
- `xrootddomain`: The domain to use for xrootd requests, often also called the Redirector.
- `bad_file_paths`: Optional, path to a JSON file containing an array of paths (or URLs) to ignore. If `file_mode` is `"local+xrootd"`, a file that is ignored locally is still opened via xrootd if it exists online.
- `xrootd_url_blacklist`: Optional, a list of xrootd URLs (e.g. `"rl.ac.uk"`) that should not be used for resolving files via xrootd. Parts of an URL are acceptable, i.e. `".fr"` to blacklist all French storage sites.
- `store`: A string used as value for `$STOREDIR` and is used to find files from plain data set names.
- `exp_datasets`: Object containing arrays. The arrays contain strings, which are either paths (or xrootd URLs) to NanoAOD ROOT files or data set names (containing three slashes). Paths can contain wildcards. The keys inside this object indicates what name to use as data set name during processing. These NanoAODs are used at experimental data input.
- `MET_trigger_datasets`: Object containing arrays. Datasets to use as cross-trigger datasets for trigger SF calculation. Same format as `exp_datasets`
- `data_eras`: Object containing arrays. Each list contains two elements defining a range of run numbers which correspond to a specific era. The name of the era is indicated by the key.
- `mc_datasets`: Object containing arrays. The arrays contain strings, which are either paths (or xrootd URLs) to NanoAOD ROOT files or data set names (containing three slashes). Paths can contain wildcards. The keys inside this object indicates what name to use as data set name during processing. These NanoAODs are used as MC data input.
- `dataset_for_systematics`: Object containing arrays. Each array has two strings as elements. The keys of the object agree with a key from `mc_datasets`. If an MC data set is present here, it means it is a variation of another MC data set and is used to estimate a systematic uncertainty. The first string inside the array is a key in `mc_datasets` and names the data set this is a variation of, while the second string names the uncertainty and its direction.
- `dataset_trigger_map`: Object of arrays. The arrays contain trigger paths to use, while the keys indicate which `exp_datasets` they are used on. If a trigger is not present in the NanoAOD, it will be ignored. The user has to make sure manually that there are no typos here.
- `MET_triggers`: List of strings, specifying trigger paths to use as cross-triggers for trigger SF calculation.
- `dataset_trigger_order`: Array, naming each of the keys in `exp_datasets`. If an event is triggered in more than one data set, it will only be processed in the data set listed first here.
- `crosssection_uncertainty`: Optional, path to a JSON file. This file contains an object of arrays. This indicated the uncertainty as a factor of each MC data set. The name of the data set is given by the keys. The arrays contain two elements, a string naming the cross section uncertainty and a float giving the factor. Two cross section uncertainties with the same name are assumed to be fully correlated. Instead of a float, the factor can also be `Null` to assume no cross section uncertainty. This does not affect data sets that are part of `dataset_for_systematics`. If an MC data set is not named here, zero uncertainty will be assumed.
- `pdf_types`: Optional, object where the keys are the LHA IDs of the PDF sets used by these data sets, and the values describe the type of these PDF sets - Options are `"Hessian"` for hessian sets, and `"MC_Gaussian"` and `"MC"` for MC replica data sets, where `"MC_Gaussian"` calculates the standard deviation of these sets, and `"MC"` takes the weights from the ordered distribution closest to the bounds of the 1 sigma region, i.e., 16% and 84%. Pepper's example configuration should cover most run 2 data sets, but new sets may be added with time.
- `split_pdf_uncs`: Optional, bool. Just output the full set of pdf variations, rather than trying to compute the combined uncertainty in pepper. NB: for NLO samples, one should output the full set of variations and calculate the combined uncertainty on the histogram for most accurate results.
- `normalize_pdf_uncs`: Optional, bool. Normalize the PDF variations before to the nominal cross sections before any cuts, removing effects from varied cross sections while keeping shape effects as well as rate effects from the acceptance. This requires running `compute_mc_lumifactors.py` with the `-p` option.
- `DY_datasets`: Optional, list. List of MC datasets for calculating and applying DY SFs. If not specified, will use all MC datasets starting "DY".

## Scale factors, weights and other inputs
These determine various calibrations and weightings. In case they are optional and are not present, the corresponding procedure will be skipped.
- `datadir`: Optional, string. Use as value for `$DATADIR`
- `lumimask`: Optional, path to the Golden JSON file.
- `luminosity`: Float, needed when computing `mc_lumifactors`. Luminosity in 1/fb resulting from applying lumimask.
- `crosssections`: Object containing floats or a path to a JSON file containing that, needed when computing `mc_lumifactors`. Cross sections of all the sets specified in `mc_datasets`. Needs to be spcified in fb.
- `norm_genweights`: Optional, bool. Whether to normalise all generator weights, considering only the sign (generally recommended by the gen group in case a sample has a few very large event weights).
- `genweights_to_norm`: Optional, array. If `norm_genweights` is not true, normalise only this list of datasets (considering only the sign of the gen weight).
- `mc_lumifactors`: Path to JSON file produced by the `compute_mc_lumifactors.py` script.
- `pileup_reweighting`: Optional, path to the pileup weights. This is a ROOT file generated by the `compute_pileup_weights.py` script.
- `stitching_factors`: Optional, dictionary of roots of dataset name to be stitched (i.e., all datasets with names starting with this string will be stitched) and a dictionary defining the stitching factors, which should contain "axis", defining a datapicker object for the axis along which the stitching should be performed, "edges" defining the edges of the bins on this axis (with the last edge assumed to be infinity) and "factors", which should be the output of `calculate_stitching_factors.py`.
- `electron_sf`: Optional, array of arrays. The inner ones contain exactly three elements, the first one being a path to a scale factor ROOT file provided by the electron POG, the second one being the name of the histogram to use inside the ROOT file, usually "EGamma_SF2D", the third one is an array specifying the order of the axes in the histogram. Axes normally are "eta" and "pt" but can also be other particle attributes.
- `muon_sf`: Optional, array of arrays. The inner ones contain exactly three elements, the first one being a path to a scale factor ROOT file provided by the muon POG, the second one being the name of the histogram to use inside the ROOT file, the third one is an array specifying the order of the axes in the histogram. Axes normally are "abseta" and "pt" but can also be other particle attributes.
- `split_muon_uncertainty` Optional, boolean. If true, split muon scale factor uncertainties into systematic and statistical parts. For combinations across years, the systematic part can be considered correlated, while the statistical part can be considered uncorrelated.
- `muon_rochester`: Optional, Path to the TXT file provided by the Muon POG for Rochester correction.
- `btag_sf`: Optional, array of arrays. The inner ones contain exactly to elements, which are both paths. The first path indicates a scale factor CSV file provided by the BTV POG. The second path indicates an efficiency ROOT file produced by the `generate_btag_efficiencies.py` script.
- `btag_method`: Optional, string. This it to choose the method for the computation of the b-tagging scale factors. Options are `fixedwp` or `iterativefit`. If not specified, default is `fixedwp`.
- `btag_measure_type`: Optional, string. Choose from the different options provided for the b-tagging scale factors. Options are `mujets`, referring to scale factors measured only in a muon-enriched QCD sample, and `comb`, referring to a combination of different datasets including ttbar. Default is `mujets`.
- `btag_ignoremissing`: Optional, boolean. If true, any values that have no correction in the btag scale factor file will return 1, instead of throwing an exception. If not specified, default is `false`.
- `btag_splitting_scheme`: Optional, string. Can be `Null` for no splitting, `"years"` for correlated and uncorrelated across years split or `"sources"` to split into all available subsources. Latter to options require the split uncertainties to be present in the btag SF file.
- `split_btag_year_corr`: Deprecated, use btag_splitting_scheme instead. Optional, boolean. If true, split btagging uncertainties into components correlated and uncorrelated across years (not this requires this splitting to exist in the SF csv). If not specified, default is `false`.
- `jet_puid_sf`: Optional, array. The first element should be the path of the Jet PU ID SF json (in correctionlib format). If this also contains the MC efficiencies (as is the case for Run 2 UL), only one element is required; otherwise the second should be the path to a root file containing histograms of the MC efficiencies, as can be produced by `generate_jet_puid_efficiencies.py`.
- `reapply_jec`: Optional, boolean. If true, the jet energy correction given in `jet_correction` (for MC) and `jet_correction_data` will be used to correct the jets. NanoAOD always comes with some jet energy correction applied. These will be undone in order to do the correction. Reapplying is only needed if a different version of jet energy corrections is required that the one in NanoAOD.
- `jet_correction_mc`: Needed if `jet_uncertainty` or `jet_resolution` is given, or if `reapply_jec` is true. Array of paths to AK4PFchs jet energy correction TXT files for MC provided by the JetMET POG. These are usually L1, L2 and L3 corrections. adsasd
- `jet_correction_data`: Needed if `reapply_jec` is true. Array of paths to AK4PFchs jet energy correction TXT files for data provided by the JetMET POG. These are usually L1, L2 and L3 corrections.
- `jet_uncertainty`: Optional, path to a AK4PFchs jet energy uncertainty TXT file provided by the JetMET POG. This can contain the total uncertainty or individual sources.
- `junc_sources_to_use`: Optional, array of strings. If given instead of using all uncertainties given in `jet_uncertainty`, only use the ones specified in this array.
- `jet_resolution`: Optional together with `jet_ressf`, path to a AK4PFchs jet pt resolution TXT file provided by the JetMET POG, used for jet smearing.
- `jet_ressf`: Optional together with `jet_resolution`, AK4PFchs jet SF TXT file provided by the JetMET POG
- `smear_met`: Optional, bool. Whether to propagate jet smearing to MET
- `MET_xy_shifts`: Optional, path to the JSON file produced by the `produce_met_xy_nums.py` script.
- `top_pt_reweighting`: Optional, object. Determines how the top pT reweighting is applied. It has the following keys being recognized: `"method"` determines the top pT reweighting method to use, either `"theory"` for the NNLO-NLO method or `"datanlo"` for the data-NLO method. When using the theory method, the keys `"a"`, `"b"`, `"c"`, `"d"` must be present, which are floats and determine the values inside the formula `sf=a*exp(b*pt)-c*pt+d`. When using the datanlo method, the keys `"a"`, `"b"` must be present, which are float and determine the values inside the formula `sf=exp(a+b*pt)`. Regardless of the method, the keys `"scale"` and `"sys_only"` can be given. The former is a float and is used as a constant factor in front of the computed scale factor. The latter is a boolean and will determine if the computed scale factor is used as a systematic uncertainty only instead of also multiplying it to the nominal weight.

## Output
- `columns_to_save`: Optional, array of data pickers or object of data pickers. The data pickers specify which observables should be saved as per-event output. If this is an object, the keys will be used inside the output file for the corresponding observable. Per-event output must be turned on manually, even if this variable is present.
- `column_output_format`: Optional, string indicating the file format which is used to save the columns named by `columns_to_save`. This can either be `"root"` or `"hdf5"`. By default the root format is used.
- `save_categories_per_event`: Optional, bool, default `True`. Whether to save all of the categories in the per-event output. If only specific categories are desired, this can be to `False` and individual columns specified in ``columns_to_save`.
- `hists`: Path to a JSON file containing the histogram definitions as object. The keys decide the name of the histogram. For its elements, see the Histogram Definition subparagraph.
- `cuts_to_histogram`: Optional, array of strings. If this is present, histograms will only be created for cuts whose names are in this array.
- `systs_to_histogram`: Optional, array of strings. If this is present, only systematics in this array will be included in the histograms. For two-sided or numeric systematic, the directions/indices should not be included (i.e. just `muonsf` instead of `muonsf_down`, and `PDF` instead of `PDF_0`).
- `datasets_to_group`: Optional, object of strings. If a data set name is present as a key inside this object, its value will be used as name instead inside the histograms.
- `histogram_format`: Optional, either `"hist"`, `"coffea"` or `"root"`. This decides the format which will be used for saving the histograms. Defaults to `"hist"`.
### Histogram definition
Elements of the `hists` object are objects themselves. They have these keys:
- `bins`: Optional, array of objects, each defining a binned axis. It has the following elements:
  - `name`: Name of the axis
  - `label`: Label of the axis, to be displayed when plotting. You can put Latex expressions within dollar signs.
  - `n_or_arr`: Number of bins (if the spacing is regular) or an array with bin edges.
  - `lo`: Only if regular bin spacing, position of the lowest bin edge.
  - `hi`: Only if regular bin spacing, position of the highest bin edge.
  - `unit`: Optional, a string specifying the unit of the axis (for example "GeV"). Used during plotting.
  - `type`: Optional, either `float` or `int`. Specifies the data type for regularly spaced histograms. Default is `float`.
- `cats`: Optional, array of objects, each defining a category axis. The purpose of a category axis is to group events by some criteria. Similar to `bins` it also has a `name` and a `label`. In addition there is an optional key `default_key`, with a string as value, which if present defines a value for all events for which the fill can not be evaluated from the data present. If `default_key` is not given and the fill is not present, the histogram will not be filled. Note that category axes for data set and channel are automatically created and should not be specified here.
- `fill`: Object, each of its keys must be either a key in `bins` or in `cats`. In latter case, it its elements are objects, where each key names a category and where the value is a data picker defining which events belong to this category. In the former case it is just a data picker, defining the observable to use for the binned axes. If the fill is not present in the data, the histogram will not be filled.
- `step_requirement`: Optional, string. The histogram will only be created once a specific step is done. This can be either `"cut:"` followed by a cut name, or `"column:"` followed by a column name. The latter requires that a specific column has been set during the processing.
- `do_systs`: Optional, bool. If `false`, do not include systematics for this histogram. Default is `true`.
- `weight`: Optional, data picker. Specifies a customs event weight using a data picker. If it is not present, the event weight will be used.
- `label`: Optional, string. To use as the label of the bin height axis. If not given, will be chosen so it best matches CMS guidelines (something like "Events / bin").
### Data pickers
Data pickers are arrays, specifying to use something from the `data` array of the processor instance. This is used in histogramming and per-event output. Each element of such a data picker array gives a further specification that is used on the result of the previous element. The fist element will be used upon the data table of the processor. An element can be
- a column name or an object of the form `{"key": columnname}`,
- an attribute or an object of the form `{"attribute": attribute}`,
- a method name, which will be executed and its return value will be used further,
- an object of the form `{"function": functionname}`, where `functionname` is one of a range of functions defined in `hist_defns.py`, for example `"sin"`,
- an object of the form `{"leading": integer}` or `{"leading": [integer1, integer2]}`, which can be used to select only the leading particles and is equivalent in Python to `[:, integer - 1]` and `[:, integer1 - 1:integer2 - 1]` respectively.

## Particle definitions and cuts
- `apply_met_filters`: Boolean, if `true`, events are required to pass MET filters.
- `ele_cut_transreg`: Boolean, if `true` electrons are required to be outside the transition region between barrel and endcap calorimeter.
- `ele_eta_min`: Float, minimum eta electrons must have.
- `ele_eta_max`: Float, maximum eta electrons can have.
- `good_ele_id`: String, indicating the ID electrons must have. Can be `"cut:"` for cut-based followed by the working point (either `"loose"`, `"medium"` or `"tight"`) or it is `"mva:"` for MVA-based followed by either `"Iso80"` or `"Iso90"`. If its value is`"skip"`, no ID will be required.
- `good_ele_pt_min`: Float, minimum pT electrons must have.
- `additional_ele_id`: Same as `good_ele_id`, just that it will be used in order to determine if there are more than 2 leptons present.
- `additional_ele_pt_min`: Same as `good_ele_pt_min`, just that it will be used in order to determine if there are more than 2 leptons present.
- `muon_cut_transreg`: Boolean, if `true` muons are required to be outside the transition region between barrel and endcap calorimeter.
- `muon_eta_min`: Float, minimum eta muons must have.
- `muon_eta_max`: Float, maximum eta muons can have.
- `good_muon_id`: String, indicating the ID muons must have. Can be `"cut:"` for cut-based followed by the working point (either `"loose"`, `"medium"` or `"tight"`) the same just with `"mva:"` in front for MVA-based.
- `good_muon_iso`: String, indicating the isolation muons must have. Can be `"cut:"` followed by either `"very_loose"`, `"loose"`, `"medium"`, `"tight"`, `"very_tight"` or `"very_very_tight"` or one of `"dR<0.3_chg:"`, `"dR<0.3_all:"`, `"dR<0.4_all:"` followed by a float, indicating a custom isolation working point.
- `good_muon_pt_min`: Float, minimum pT muons must have.
- `additional_muon_id`: Same as `good_muon_id`, just that it will be used in order to determine if there are more than 2 leptons present. If its value is`"skip"`, no ID will be required.
- `additional_muon_iso`: Same as `good_muon_iso`, just that it will be used in order to determine if there are more than 2 leptons present. If its value is`"skip"`, no isolation will be required.
- `additional_muon_pt_min`: Same as `good_muon_pt_min`, just that it will be used in order to determine if there are more than 2 leptons present.
- `good_jet_id`: String, indicating the ID jets must have. Can be `"cut:"` followed by the working point (either `"loose"`, `"medium"` or `"tight"`) or  `"skip"` in which case no jet ID will be required.
- `good_jet_puId`: String, indicating the pileup ID jets must have. Can be `"cut:"` followed by the working point (either `"loose"`, `"medium"` or `"tight"`) or  `"skip"` in which case no jet pileup ID will be required.
- `good_jet_lepton_distance`: Float, distance in Delta R a jet must have to any lepton.
- `good_jet_eta_min`: Float, minimum eta jets must have.
- `good_jet_eta_max`: Float, maximum eta jets can have.
- `good_jet_pt_min`: Float, minimum pT jets must have.
- `hem_cut_if_ele`: Boolean, whether to ignore electrons that are in the HEM issue (a problem for 2018) region.
- `hem_cut_if_muon`: Same as `hem_cut_if_ele` for muons.
- `hem_cut_if_jet`: Same as `hem_cut_if_ele` for jets.
- `mll_min`: Float, minimum invariant mass for the mll cut.
- `lep_pt_min`: Array of floats. Each float is used as minimum for the n-th leading lepton (ordered by pT), where n is the position of the float. This will be used in the lepton pT cut and is further controlled by `lep_pt_num_satisfied`.
- `lep_pt_num_satisfied`: Integer, Minimum number of leptons that must have the minimum pT requirement fulfilled given by `lep_pt_min` in the lepton pT cut.
- `num_jets_atleast`: Integer, minimum number of jets, used in the jet number cut.
- `jet_pt_min`: Same as `lep_pt_min` for jets.
- `jet_pt_num_satisfied`: Same as `lep_pt_num_satisfied` for jets.
- `btag`: String, indicating what is considered as b tagged. Can be either `"deepcsv:"` or `"deepjet:"`, giving the b tagging algorithm to use, followed by a working point (either `"loose"`, `"medium"` or `"tight"`).
- `num_atleast_btagged`: Integer, number of b tagged jets in the corresponding cut.

## Kinematic reconstruction
- `reco_info_file`: Needed if `reco_algorithm` is present, path to a ROOT file produced by the `compute_kinreco_hists.py` script.
- `reco_w_mass`: Needed if `reco_algorithm` is `"Sonnenschein"`, float or string. If it is a float, its value is used as the W boson mass inside Sonnenschein. If it is a string, for example `"mw"`, the mass value is randomly drawn from the distribution defined by the histogram inside the `reco_info_file` named by the string.
- `reco_t_mass`: Needed if `reco_algorithm` is `"Sonnenschein"`, float or string. If it is a float, its value is used as the top quark mass inside Sonnenschein. If it is a string, for example `"mt"`, the mass value is randomly drawn from the distribution defined by the histogram inside the `reco_info_file` named by the string.
- `reco_num_smear`: Needed if `reco_algorithm` is `"Sonnenschein"`, integer determining the number of smearings inside Sonnenschein.

## ttbarll specific
- `channel_trigger_map`: Object of arrays. The keys name channels ("ee", "emu", "mumu", "e" or "mu"), while the arrays list trigger paths. If an event of a specific channel does not pass any of the triggered mentioned here for the corresponding key, it will be ignored.
- `drellyan_sf`: Optional, path to the Drell-Yan weights. This is either a ROOT file generated by the `calculate_DY_SFs.py` script or a JSON file with containing an object with the keys `bins`, `factors`, `factors_up` and `factors_down` describing a histogram used for extracting the scale factors.
- `trigger_sfs`: Optional, array where the first element is a ROOT file containing the histograms for the trigger scale factors, and the second element is an array of three strings, giving the histogram names for the ee, eµ and µµ trigger scale factors (in that order. The histograms need to be binned by leading lepton pt (x axis) and second leading lepton pt (y axis).
- `reco_algorithm`: Optional, string. Algorithm to use to reconstruct the top quarks. Can be either `"Sonnenschein"` or `"Betchart"`.
- `z_boson_window_start`: Float, lower bond of the invariant mass of the lepton system that is used in the Z window cut
- `z_boson_window_end`: Float, upper bound of the invariant mass of the lepton system that is used in the Z window cut.
- `ee/mm_min_met`: Minimum MET pT in the corresponding cut.

### DY Scale factors
- `fast_dy_sfs`: Optional, bool. Specifies if DYprocessor should run over just DY and observed data; not relevant for other processors.
- `bin_dy_sfs`: Optional, data picker defining a variable in which DY SFs are binned, used by the Processor when applying them.

### Plotting
- `plot_dataset_groups`: Optional, array. Each element is an object describing a group. These group several data sets and make them appear as one on the plot. The order of the objects describes the order of stacked bars from bottom to top. Each object has the following:
  - `label`: Label to be used on the plot for the group. Must be unique.
  - `color`: Color to be used for the group. Any valid Matplotlib colors are allowed
  - `datasets`: Array of data set names
- `plot_datasets_ignore`: Optional, list. A list of data sets to not plot.
- `plot_scale_sysuncertainty`: Optional, array. Factors to be multiplied to the scale of a systematic uncertainty. The keys are the systematic names, without the suffix (e.g. '_up').
