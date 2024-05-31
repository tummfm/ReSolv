#!/bin/bash

if [ ! -f ct0c01229_si_006.zip ]
then
   echo "charges for CoRE-MOF-2019 from Snurr, JCTC, 2021"
   echo "GET IT FROM: 
 https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.0c01229/suppl_file/ct0c01229_si_006.zip"
   echo "they want a cookie and I didn't want to get there"
fi

if [ ! -f core-mof-1.0-ddec.tar ]
then
    echo "charges for CoRE-MOF-2014 from Sholl, 2016"
    echo "get it from https://zenodo.org/record/3986573/files/core-mof-1.0-ddec.tar?download=1"
    wget https://zenodo.org/record/3986573/files/core-mof-1.0-ddec.tar?download=1
fi

if [ ! -f 2019-11-01-ASR-internal_14142.csv ]
then
   mkdir -p coremof-2019
   cd coremof-2019
   wget https://zenodo.org/record/3677685/files/2019-11-01-ASR-internal_14142.csv
   wget https://zenodo.org/record/3677685/files/2019-11-01-ASR-public_12020.csv
   wget https://zenodo.org/record/3677685/files/2019-11-01-ASR-public_12020.tar.gz
   wget https://zenodo.org/record/3677685/files/2019-11-01-FSR-internal-overlap-freeONLY_9146.csv
   wget https://zenodo.org/record/3677685/files/2019-11-01-FSR-public_7061.csv
   wget https://zenodo.org/record/3677685/files/2019-11-01-FSR-public_7061.tar.gz
   wget https://zenodo.org/record/3677685/files/ASR-full_unmodified.csv
   wget https://zenodo.org/record/3677685/files/FSR-full_unmodified.csv
   cd $OLDPWD
fi
   
echo "
                 CoRE-MOF-2019-README
- there is a thousand different versions...
- ASR: solvent removed
- FSR: solvent removed and relaxed?
- also, only cif-files changed from the CSD are in the database...
- duplicates 10.1021/acs.chemmater.5b03836
"
