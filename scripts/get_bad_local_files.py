#!/usr/bin/env python3

import os
import signal
import pepper
import uproot
import json
import gzip
import concurrent.futures
from argparse import ArgumentParser
from tqdm import tqdm


def check(path):
    with uproot.open(path):
        pass


def wait_and_check(path, timeout):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        pid = executor.submit(os.getpid).result()
        future = executor.submit(check, path)
        try:
            future.result(timeout)
        except concurrent.futures.TimeoutError:
            os.kill(pid, signal.SIGKILL)
            raise


parser = ArgumentParser(
    description="Search for NanoAOD files that exist in the store directory "
    "but are not accessible, possibly due to technical issues. The output is "
    "a JSON list usable the bad_file_paths config. Searching is done by "
    "checking all NanoAOD files a config points to")
parser.add_argument(
    "config", help="Path to the JSON config file containing the MC and "
    "experimental data set names")
parser.add_argument("out", help="Path to the output JSON file")
parser.add_argument(
    "--timeout", "-t", type=int, default=5, help="Time to wait in seconds "
    "when searching bad files in directories and trying to open them. "
    "Default 5.")
parser.add_argument(
    "--processes", "-p", type=int, default=10, help="Number of processes to "
    "use when searching in directories. Too high numbers can slow down the "
    "file system, which may cause a file to be falsely identified as bad. "
    "Default 10")
parser.add_argument(
    "--no_search", "-n", action="store_true", help="Disable the search in "
    "directories")
parser.add_argument(
    "--list", "-l", action="append", default=[], help="Path to text file "
    "containing a file path per line. If a file is found in one of these "
    "lists, it will be considered bad.")
args = parser.parse_args()

config = pepper.ConfigBasicPhysics(args.config)
config["file_mode"] = "local"
<<<<<<< HEAD
=======
config["bad_file_paths"] = []
>>>>>>> pepper/master
datasets = config.get_datasets()
lfns = []
for dsname, dspaths in datasets.items():
    lfns.extend(dspaths)

paths = []
for lfn in lfns:
    paths.extend(config.get_paths_for_lfn(lfn))
paths = set(paths)

corrupt = []
for list in args.list:
    opener = gzip.open if list.endswith(".gz") else open
    with opener(list, "rt") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line in paths:
                corrupt.append(line)


if not args.no_search:
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.processes) as executor:
        futures = {executor.submit(wait_and_check, path, args.timeout):
                   path for path in sorted(paths)}

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except (OSError, concurrent.futures.TimeoutError,
                        concurrent.futures.process.BrokenProcessPool):
                    corrupt.append(futures[future])
                    print(f"Failure in {futures[future]}")
                pbar.update(1)

with open(args.out, "w") as f:
    json.dump(corrupt, f, indent=4)

print(f"Found {len(corrupt)} bad files")
