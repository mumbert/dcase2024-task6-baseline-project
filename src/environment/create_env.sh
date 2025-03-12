#!/bin/bash

# TO BE ABLE CONDA INSIDE THE SCRIPT
eval "$(conda shell.bash hook)"

# LOAD CONFIG AND ENV
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONFIG_FILE=$(realpath "$SCRIPT_DIR/dcase_baseline.config")
echo "Loading config from $CONFIG_FILE"
. $CONFIG_FILE # to load CONDA_ENV and PY_VERSION

# STEPS
conda activate base
conda remove -n $CONDA_ENV --all -y
conda create --name $CONDA_ENV python=$PY_VERSION -y

# FOR DCASE
conda activate $CONDA_ENV
pip install --force-reinstall pip==23.0
pip install -e . --use-deprecated=legacy-resolver
pip install -e .[dev] --use-deprecated=legacy-resolver
pre-commit install

# install Java 11 in Debian 11: https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-on-debian-11
sudo apt update
sudo apt install default-jre
java -version

# Validate version, should be >= 8 and <= 13
link="https://github.com/Labbeti/aac-metrics/blob/b2e4ace787bef36577935605faeef74bbeffcf15/src/aac_metrics/utils/checks.py#L16"
java -version
JAVA_VER=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | awk -F '.' '{sub("^$", "0", $2); print $1}')
if [[ "$JAVA_VER" -ge 8 ]] && [[ "$JAVA_VER" -le 13 ]]
then
    echo "Java version [$JAVA_VER] is within the expected range [8,13]"
else
    echo -e "\nERROR: Java version [$JAVA_VER] is outside the expected range [8,13]. This affects aac-metrics, check $link"
fi

# TEST INSTALLATION
echo "Testing installation"
cmd="python src/dcase24t6/test_installation.py"
echo $cmd && eval $cmd

# HELPER COMMANDS TO CONNECT TO THE ENV
conda env list
echo "conda activate $CONDA_ENV"
