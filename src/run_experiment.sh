#
# This script copies all files in the current directory into a subdirectory
# with the given name, removing the first row from each. It's meant for saving
# experiment results in a way that can be easily plotted by MATLAB.
#
# HISTORY
# -------
# 2016-09-05: Created by Jonathan D. Jones
#

# Estimate state for all samples using configuration defined in ukf.config
./main estimate -1

mkdir $1$2

# Copy data and remove first row
for filename in $1*.csv; do
    tail +2 $filename > $1$2/${filename#$1}
done

# Copy the config file that generated the data for this experiment
cp ukf.config $1$2/experiment.config
