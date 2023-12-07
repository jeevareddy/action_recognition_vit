#!/bin/bash

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v="$2"
   fi
  shift
done

# for arg in "$@"
# do 
# echo Value ${arg#a}
# done
pip install -r requirements.txt
#Train model
python ${PWD}/run.py