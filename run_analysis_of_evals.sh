#!/bin/bash

uv run src/analyze_sweep.py outputs/CNNaug_data
uv run src/analyze_sweep.py outputs/CNNnoaug_data
uv run src/analyze_sweep.py outputs/pointwiseAug_data
uv run src/analyze_sweep.py outputs/pointwiseNoAug_data