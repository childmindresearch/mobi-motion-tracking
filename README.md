
# `mobi_motion_tracking`

A python package for 3D motion tracking data processing.

[![Build](https://github.com/childmindresearch/mobi-motion-tracking/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/mobi-motion-tracking/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/mobi-motion-tracking/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/mobi-motion-tracking)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![LGPL--2.1 License](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](https://github.com/childmindresearch/mobi-motion-tracking/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/mobi-motion-tracking)

Welcome to `mobi_motion_tracking`, a Python library designed for processing and analyzing captured motion tracking data. This library provides a set of tools to analyze 3D skeletal data against a gold standard and calculate a similarity metric between them. We provide a labelled output file containing all specified, calculated metrics and metadata for post-processing analysis. 

## Supported formats & devices

The package currently supports the following formats:

| Format | Manufacturer | Device | Implementation status |
| --- | --- | --- | --- |
| XLSX | Microsoft | KINECT | ✅ |
| XLSX | StereoLabs | ZED 2i | ✅ |

**Special Note**
    This package works off the assumption that all input participant files are named according to their participant ID containing only digits in the correct format i.e. (0000.xlsx). It is also assummed that each participant file contains 1 or more sheets of data per sequence. The sheets are expected to be named based off the string 'seq' and an integer representing the sequence i.e. ('seq1'). The file names can be any length of digits and there can be an unlimited number of sequences. Sequences will only be processed if indicated in the input list.
    
    It is assumed that any gold file contains the string 'gold' or 'Gold' in its basename.
    
    It is also expected that there are no missing data points prior to running the package.

## Processing pipeline implementation

The main processing pipeline of the `mobi_motion_tracking` module can be described as follows:

- **Data loading**: Data is loaded per participant, per sequence as a dataframe. If a file is named incorrectly, is not in the right format, or the sequence does not exist for that participant, it is skipped.
- **Data preprocessing**: The raw subject and gold standard joint data are centered to the hip for every frame. The average lengths of all skeletal segments of the gold standard data are calculated, and the centered subject joint data is normalized to the average gold lengths.
- **Metrics Calculation**: Calculates specified similarity metrics on the preprocessed data, namely DTW (dynamic Time Warping). DTW is the only possible option at this current developmental stage.

## Installation

Install this package via :

```sh
pip install mobi_motion_tracking
```

Or get the newest development version via:

```sh
pip install git+https://github.com/childmindresearch/mobi-motion-tracking
```

## Quick start

### Using mobi_motion_tracking through the command-line:
#### Run single files:
```sh
mobi_motion_tracking -d /subject/file/path/000.xlsx -g /gold/file/path/gold.xlsx -s "1,2,3" -a "dtw"
```

#### Run entire directories:
```sh
mobi_motion_tracking -d /subject/file/dir -g /gold/file/path/gold.xlsx -s "1,2,3" -a "dtw"
```

### Using mobi_motion_tracking through a python script or notebook:

#### Running single files:
```Python

from mobi_motion_tracking.core import orchestrator
import pathlib

# Define data file path, gold file path, sequence list, and algorithm
# Support for saving as .ndjson
data_path = pathlib.Path("100.xlsx")
gold_path = pathlib.Path("Gold.xlsx")
sequence = [1, 2, 3]
algorithm = "dtw"

# Run the orchestrator
results = orchestrator.run(
    experimental_path=data_path,
    gold_path=gold_path,
    sequence=sequence,
    algorithm=algorithm,
)


# Data available in list of results.
subject1_seq1 = results[0][0]

participant_ID = subject1_seq1["participant_ID"]
sheetname = subject1_seq1["sheetname"]
method = subject1_seq1["method"]
distance = subject1_seq1["distance"]
```
#### Running entire directories:
```Python

from mobi_motion_tracking.core import orchestrator
import pathlib

# Define data file path, gold file path, sequence list, and algorithm
# Support for saving as .ndjson
data_path = pathlib.Path("")
gold_path = pathlib.Path("Gold.xlsx")
sequence = [1, 2, 3]
algorithm = "dtw"

# Run the orchestrator
results = orchestrator.run(
    experimental_path=data_path,
    gold_path=gold_path,
    sequence=sequence,
    algorithm=algorithm,
)


# Data available in list of results.
subject1_seq1 = results[0][0]

participant_ID = subject1_seq1["participant_ID"]
sheetname = subject1_seq1["sheetname"]
method = subject1_seq1["method"]
distance = subject1_seq1["distance"]
```
