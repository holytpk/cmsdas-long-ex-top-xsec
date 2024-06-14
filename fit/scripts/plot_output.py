import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

def plot_full(config, nom_vals_dict, data_vals, stat_var, hi_tot, lo_tot, data_ax, use_stat_var=True, lumi=1.23, do_ratio_plot=True, fig_options={}):
    
    if do_ratio_plot:
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, dpi=200, **fig_options)
    else:
        fig, ax1 = plt.subplots(nrows=1, dpi=200, **fig_options)

    ax_len = len(data_ax.centers)
    bottom = np.zeros(ax_len)

    tot_mc_vals = np.array(list(nom_vals_dict.values())).sum(axis=0)

    if "DYlo" in nom_vals_dict:
        nom_vals_dict = nom_vals_dict.copy()
        nom_vals_dict["DY"] = nom_vals_dict["DY"] + nom_vals_dict["DYlo"]
        del nom_vals_dict["DYlo"]

    colors = config["colors"]
    display_names = config["display_names"]

    proclist = sorted(nom_vals_dict.keys(), key=lambda p: nom_vals_dict[p].sum())
        
    for proc in proclist:
        vals = nom_vals_dict[proc]
        cat_label = display_names[proc]
        ax1.bar(data_ax.centers, vals, width=data_ax.widths, bottom=bottom, label=cat_label, color=colors[proc])
        bottom += vals

    hi_tot_comb = np.sqrt(hi_tot + stat_var)
    lo_tot_comb = np.sqrt(lo_tot + stat_var)

    ax1.bar(data_ax.centers, hi_tot_comb + lo_tot_comb, width=data_ax.widths, bottom=bottom - lo_tot_comb, label="Uncertainty", alpha=0.3, color="black")

    ax1.errorbar(data_ax.centers, data_vals, yerr=np.sqrt(data_vals), markersize=5, color="black", label="Data", linestyle="None", marker="o")

    #ax1.set_xlabel(data_ax.label)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.set_ylabel("Event counts")
    ax1.legend(fontsize="small")
    hep.cms.label(ax=ax1, data=True, year="2022", lumi=f"{lumi:.1f}", com=13.6, pad=0.07)

    if do_ratio_plot:    
        ratio = np.nan_to_num(data_vals / tot_mc_vals, nan=1.)
        ratio_err = np.nan_to_num(np.sqrt(data_vals) / tot_mc_vals, nan=0.)

        relvar_mc_stat = np.nan_to_num(np.sqrt(stat_var) / tot_mc_vals, nan=0.)
        relvar_mc_tot = np.nan_to_num((hi_tot_comb + lo_tot_comb)/tot_mc_vals, nan=0.)
        relvar_mc_lo = np.nan_to_num(lo_tot_comb/tot_mc_vals, nan=0.)

        ax2.axhline(1, color="black", linestyle="dashed")
        if use_stat_var:
            ax2.bar(data_ax.centers, 2 * relvar_mc_stat, width=data_ax.widths, bottom=1-relvar_mc_stat, label="stat", alpha=0.5, color="black")
        ax2.bar(data_ax.centers, relvar_mc_tot, width=data_ax.widths, bottom=1-relvar_mc_lo, label="stat+syst", alpha=0.2, color="black")

        ax2.errorbar(data_ax.centers, ratio, yerr=ratio_err, markersize=5, color="black", label="Data", linestyle="None", marker="o")

        ax2.set_xlabel(data_ax.label)
        ax2.set_ylabel("Data / Pred")
        #ax2.legend(fontsize="small")
        ax2.set_ylim(0.5, 1.5)

        plt.subplots_adjust(hspace=0.2)

        return fig, (ax1, ax2)
    else:
        return fig, ax1
    
