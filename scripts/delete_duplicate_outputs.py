import os
from argparse import ArgumentParser
from functools import partial
from tqdm import tqdm
import uproot

import pepper
import pepper.htcondor


def process_dir(dir, delete=False):
    duplicate_files = []
    corrupted_files = []
    processed_chunks = set()
    for in_file in os.listdir(dir):
        if not in_file.endswith(".hdf5") and not in_file.endswith(".h5") \
                and not in_file.endswith(".root"):
            continue
        in_file = os.path.join(dir, in_file)
        try:
            if in_file.endswith(".root"):
                with uproot.open(in_file) as f:
                    identifier = str(f["identifier"])
            else:
                with pepper.HDF5File(in_file, "r") as f:
                    identifier = f["identifier"]
        except (OSError, uproot.DeserializationError):
            if delete:
                os.remove(in_file)
            else:
                os.rename(in_file, in_file + ".corrupted")
            corrupted_files.append(os.path.relpath(in_file, dir))
            continue
        if identifier in processed_chunks:
            if delete:
                os.remove(in_file)
            else:
                os.rename(in_file, in_file + ".duplicate")
            duplicate_files.append(os.path.relpath(in_file, dir))
        processed_chunks.add(identifier)
    return dir, (duplicate_files, corrupted_files)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Check if any of the samples in the event directory"
        "produced by select_events.py are duplicated or corrupted, and if so, "
        "rename or delete them")
    parser.add_argument(
        "eventdir", help="Directory containing sample directories. Should be "
        "the same as the eventdir argument when running a processor")
    parser.add_argument(
        "-c", "--condor", type=int, metavar="simul_jobs",
        help="Number of HTCondor jobs to launch")
    parser.add_argument(
        "-r", "--retries", type=int, help="Number of times to retry if there "
        "is exception in an HTCondor job. If not given, retry infinitely."
    )
    parser.add_argument(
        "-o", "--offset", type=int, help="Skip the first <offset> directories",
        default=0)
    parser.add_argument(
        "-d", "--delete", action="store_true", help="Delete duplicate or "
        "corrupt files instead of renaming them")
    parser.add_argument(
        "-i", "--condorinit",
        help="Shell script that will be sourced by an HTCondor job after "
        "starting. This can be used to setup environment variables, if using "
        "for example CMSSW. If not provided, the local content of the "
        "environment variable PEPPER_CONDOR_ENV will be used as path to the "
        "script instead.")
    parser.add_argument(
        "--condorsubmit",
        help="Text file containing additional parameters to put into the "
        "HTCondor job submission file that is used for condor_submit"
    )
    parser.add_argument(
        "--condorlogdir", help="Directory to store stdout and stderr logs "
        "running on HTCondor. Default is pepper_logs", default="pepper_logs")
    args = parser.parse_args()

    if args.condorinit is not None:
        with open(args.condorinit) as f:
            condorinit = f.read()
    else:
        condorinit = None
    if args.condorsubmit is not None:
        with open(args.condorsubmit) as f:
            condorsubmit = f.read()
    else:
        condorsubmit = None

    sample_dir = os.path.realpath(args.eventdir)

    dirs = next(os.walk(sample_dir))[1][args.offset:]

    results = {}
    process_dir = partial(process_dir, delete=args.delete)
    with pepper.htcondor.Cluster(
        args.condor,
        condorsubmitfile=args.condorsubmit,
        condorinit=args.condorinit,
        retries=args.retries,
        logdir=args.condorlogdir
    ) as cluster:
        cluster.set_global_config()
        for result in tqdm(cluster.process(
            process_dir,
            [os.path.join(sample_dir, d) for d in dirs]
        ), total=len(dirs)):
            dir, result = result
            results[dir] = result

    print("Duplicate or corrupted files: ")
    have_moved = False
    for d, (dups, corrupt) in results.items():
        moved_files = dups + corrupt
        if len(moved_files) == 0:
            continue
        print(f"{d}: {', '.join(moved_files)}")
        have_moved = True
    if not have_moved:
        print("None")
