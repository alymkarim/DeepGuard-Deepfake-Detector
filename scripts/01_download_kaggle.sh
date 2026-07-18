#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/kaggle
kaggle datasets download -d xdxd003/ff-c23 -p data/kaggle --unzip
