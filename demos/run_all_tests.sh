#!/bin/sh

svn co https://github.com/aaspip/data/trunk ./data #or git clone https://github.com/aaspip/data
cp -rf ./data/* ./

python test_pyseistr_somean2d.py
python test_pyseistr_somean3d.py
python test_pyseistr_somf2d.py
python test_pyseistr_somf3d.py


python test_pyseistr_passive_recon3d.py
python test_pyseistr_ssprecursor.py
python test_pyseistr_rf.py
python test_pyseistr_das.py

python test_pyseistr_soint2d.py
python test_pyseistr_soint2d_seafloor.py