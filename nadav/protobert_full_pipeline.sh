#! /bin/tcsh

source ~/my_python/bin/activate.csh
module load tensorflow/2.0.0

setenv PYTHONPATH "${PYTHONPATH}:/cs/phd/nadavb/github_projects/shared_utils:/cs/phd/nadavb/cafa_project/src"
python ~/cafa_project/bin/protobert_full_pipeline.py
