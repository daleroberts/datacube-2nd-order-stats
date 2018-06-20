# 2nd-Order Robust Temporal Statistics of Earth Observations

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Release](https://img.shields.io/badge/Release-Private-ff69b4.svg)

*Note: please do not release this code repository or details of the methodology to people outside of GA until the research paper is public. It is fine to release output products.*

This [datacube-stats](https://github.com/GeoscienceAustralia/datacube-stats) plug-in generates outputs to robustly understand variation in the landscape. These results have been preliminarily published/presented at IGARS 2018 in [this paper](https://github.com/daleroberts/datacube-2nd-order-stats/raw/master/docs/IGARS2018-2ndOrderStats.pdf).

## Example Outputs

<img src="https://github.com/daleroberts/datacube-2nd-order-stats/raw/master/docs/2ndorder-ternary.png" width="800">
<img src="https://github.com/daleroberts/datacube-2nd-order-stats/raw/master/docs/2ndorder-braycurtis.png" width="800">

## Temporal Statistics

## Tasselled Cap


## Installation

```
git clone https://github.com/daleroberts/datacube-2nd-order-stats
```

## Continental-scale or large area

Some additional tools are contained in this repository:

  - `job.pbs` is a PBS job script that reads the `tiles` file (tiles specified in Australian Albers) and parallelises across multiple nodes.

  - `retile` is a python script (used by `job.pbs`) that reads the `config.yaml` and generates a new tiling number scheme for `datacube-stats`
     that is based on the `tile_size` parameter in `config.yaml`.

  - `stat` is a python script that gives information about how many of the output tiles have been generated so far. It is useful for checking
      the status of continental runs.
