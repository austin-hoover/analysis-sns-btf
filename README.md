# btf-scripts-analysis
Analysis of data from the Beam Test Facility (BTF).

`generic` holds scripts that process the raw measurement data (https://ics-srv-web2.sns.ornl.gov/BTF_DATA/Measurements/), make useful plots, etc. The structure mirrors that of https://code-int.ornl.gov/sns-rad/SE/btf-scripts; each analysis script is associated with one measurement script.

The scripts in `specific` are linked to specific measurements. Each folder in `specific` should be dated and include a README describing the goal of the analysis. The goal is that all output data/figures should be reproducible.

BTF data analysis/visualization tools are kept in `tools`. If the tools are sufficiently general (not specific to BTF data), they should probably be tracked in a different repository (like https://github.com/austin-hoover/psdist); then they can just be imported.

I do not know if this is the best system...