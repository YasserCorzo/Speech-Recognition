#!/usr/bin/env bash
# THIS FILE IS GENERATED BY tools/setup_anaconda.sh
if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
. /swl/home/byan/speech_course_hw/11751_hw3/miniconda/etc/profile.d/conda.sh && conda deactivate && conda activate ctc-asr
