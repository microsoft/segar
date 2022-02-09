#!/bin/bash

declare -a Dirs=("sim" "mdps" "rules" "tasks" "rendering" "factors")

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

for Dir in "${Dirs[@]}"
do
  cd "rpp/$Dir"
  jupyter nbconvert --to markdown README.ipynb
  cd -
done
