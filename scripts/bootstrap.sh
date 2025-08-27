#!/usr/bin/env bash

set -euo pipefail

# Ensure pip exists and install deps

python -m pip --version || true

python -m pip install --upgrade pip wheel setuptools || true

python -m pip install -r requirements.txt