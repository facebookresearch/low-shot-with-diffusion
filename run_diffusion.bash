#! /bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
set -ex

# replace with actual datadir 
DDIR=/checkpoint/matthijs/low-shot/
export DDIR


stage=$1

if [ -z "$stage" ]; then
    exit 1
fi

all_nbg="1000000 10000000"
# all experiments with 30 neightbors
k=30 

if [ $stage ==  1 ]; then

    # run on validation set to find optimal parameters
    mkdir -p $DDIR/logs/diffusion_val
    
    # nl = number of labeled images per class
    for nl in 1 2 5 10 20; do

        for nbg in $all_nbg; do

            for seed in 1 2 3 4 5; do 
                name=nl$nl.nbg$nbg.seed$seed
                # run with a large number of iterations to make sure we
                # don't miss the max
                python -u diffusion.py \
                       --mode val --nlabeled $nl --seed $seed \
                       --nbg $nbg --k $k \
                       --niter 40 |
                    tee $DDIR/logs/diffusion_val/$name.stdout
            done
                
        done
    done

fi



if [ $stage == 2 ]; then
    
    python parse_diffusion.py --mode val 

fi


if [ $stage == 3 ]; then
    # found by the previous script on the validation parameters
    # nl,nbg,k:niter
    validated_params="1,0,30:3 1,100000,30:0 1,1000000,30:3 1,10000000,30:4 1,100000000,30:7 2,0,30:3 2,100000,30:3 2,1000000,30:3 2,10000000,30:4 2,100000000,30:6 5,0,30:1 5,100000,30:0 5,1000000,30:3 5,10000000,30:4 5,100000000,30:5 10,0,30:1 10,100000,30:0 10,1000000,30:2 10,10000000,30:3 10,100000000,30:5 20,0,30:1 20,100000,30:0 20,1000000,30:2 20,10000000,30:3 20,100000000,30:4"

    # run on test
    mkdir -p $DDIR/logs/diffusion_test
   
    
    for nl in 1 2 5 10 20; do
        for nbg in $all_nbg; do 
            niter=''
            for vp in $validated_params; do
                if [ ${vp%:*} == $nl,$nbg,$k ]; then
                    niter=${vp#*:}
                    break
                fi
            done
            [ -z "$niter" ] && (
                echo could not find params for $nl,$nbg,$k; exit 1)

            for seed in {1..5}; do

                name=nl$nl.nbg$nbg.seed$seed
                
                # use --lslice 96 to reduce memory usage
                
                python -u diffusion.py \
                       --mode test --nlabeled $nl --seed $seed \
                       --nbg $nbg --k $k \
                       --niter $niter |
                    tee $DDIR/logs/diffusion_test/$name.stdout
            done
        done
    done


fi


if [ $stage == 4 ]; then

    python parse_diffusion.py --mode test

fi
