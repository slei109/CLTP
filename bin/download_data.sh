#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
DIR=$REPO/download/
mkdir -p $DIR

# download XNLI dataset
function download_xnli {
    OUTPATH=$DIR/xnli-tmp/
    if [ ! -d $OUTPATH/XNLI-MT-1.0 ]; then
      if [ ! -f $OUTPATH/XNLI-MT-1.0.zip ]; then
        wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/XNLI-MT-1.0.zip -d $OUTPATH
    fi
    if [ ! -d $OUTPATH/XNLI-1.0 ]; then
      if [ ! -f $OUTPATH/XNLI-1.0.zip ]; then
        wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/XNLI-1.0.zip -d $OUTPATH
    fi
    python $REPO/utils_preprocess.py \
      --data_dir $OUTPATH \
      --output_dir $DIR/xnli/ \
      --task xnli
    rm -rf $OUTPATH
    echo "Successfully downloaded data at $DIR/xnli" >> $DIR/download.log
}

download_xnli