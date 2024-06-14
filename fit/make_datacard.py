import argparse
import hjson
import os

def pad(s, length=12):
    nspace = max(1, length - len(s))
    return s + (" " * nspace)

def make_datacard(config, outfile, exclude=[], onlycats=None):

    card = ""
    divider = "-"*100 + "\n"

    cats = config["categories"]
    if onlycats is not None:
        cats = {k:v for k,v in cats.items() if k in onlycats}
    bg_procs = list(config["bg_procs"].keys())
    if "np_background" in config:
        bg_procs.append(config["np_background"]["proc"])
    signal_procs = list(config["signal_procs"].keys())
    systematics = config["systematics"]
    lnN_uncs = config["lnN_uncs"]

    card += f"imax {len(cats)}\n"
    card += f"jmax *\n"
    card += f"kmax *\n"
    card += divider

    for cat in cats:
        card += f"shapes * {cat} {cat}.root $PROCESS $PROCESS_$SYSTEMATIC\n"
    card += divider

    card += pad("bin")
    for cat in cats:
        card += pad(cat)
    card += "\n"

    card += pad("observation")
    card += pad("-1") * len(cats)
    card += "\n"
    card += divider

    all_procs = [*signal_procs, *bg_procs]

    card += pad("bin", 32)
    for cat in cats:
        for proc in all_procs:
            if not (cat,proc) in exclude:
                card += pad(cat)
    card += "\n"

    card += pad("process", 32)
    for cat in cats:
        for proc in all_procs:
            if not (cat,proc) in exclude:
                card += pad(proc)
    card += "\n"

    card += pad("process", 32)
    for cat in cats:
        j = 1-len(signal_procs)
        for proc in all_procs:
            if not (cat,proc) in exclude:
                card += pad(str(j))
            j += 1
    card += "\n"

    card += pad("rate", 32)
    for cat in cats:
        for proc in all_procs:
            if not (cat,proc) in exclude:
                card += pad("-1")
    card += "\n"
    card += divider

    for sys, sys_conf in lnN_uncs.items():
        card += pad(sys, 20) + pad("lnN")
        for cat in cats:
            for proc in all_procs:
                if not (cat,proc) in exclude:
                    if ("procs" in sys_conf and proc not in sys_conf["procs"]) \
                        or ("cats" in sys_conf and cat not in sys_conf["cats"]):
                        card += pad("-")
                    else:
                        card += pad(str(1 + sys_conf["unc"]))
        card += "\n"

    card += "\n"

    for sys, sys_conf in systematics.items():
        param_type = "shapeU" if sys == "btagsf_id" else "shape"
        card += pad(sys, 20) + pad(param_type)
        for cat in cats:
            for proc in [*signal_procs, *bg_procs]:
                if not (cat,proc) in exclude:
                    if ("procs" in sys_conf and proc not in sys_conf["procs"]) \
                        or ("cats" in sys_conf and cat not in sys_conf["cats"]) :
                        card += pad("-")
                    else:
                        card += pad("1")
                
        card += "\n"

    card += "\n"
    card += divider

    card += "* autoMCStats 100 1\n"

    with open(outfile, "w") as f:
        f.write(card)

    return card

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=str)
    parser.add_argument("outdir", type=str)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.json) as f:
        config = hjson.load(f)

    outfile = os.path.join(args.outdir, "all.txt")

    print(make_datacard(config, outfile))