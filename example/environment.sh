#!/bin/bash

# Set custom VOMS proxy path. This needs to be accessible from Condor
# Please do not forget to run voms-proxy-init --voms cms --out $X509_USER_PROXY
export X509_USER_PROXY=~/.globus/x509up
# Needed for eos access to work via condor
export EOS_MGM_URL=root://eosuser.cern.ch
# Load LCG
source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_104 x86_64-el9-gcc11-opt

# Use this script also as environment script when running Pepper on HTCondor
if test -n "$BASH_VERSION"; then
    export PEPPER_CONDOR_ENV="$(realpath $BASH_SOURCE)"
elif test -n "$ZSH_VERSION"; then
    export PEPPER_CONDOR_ENV="$(realpath ${(%):-%N})"
fi

# FILL your directory info here
#export PEPPERDIR=/absolute/path/to/your/pepper/repo

# THEN uncomment the following lines to point to your python venv
#source $PEPPERDIR/env_pepper/bin/activate
#export PYTHONPATH=$PEPPERDIR/env_pepper/lib/python3.9/site-packages:$PYTHONPATH
