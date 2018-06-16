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


if [ $stage ==  1 ]; then

    # run on validation set to find optimal parameters
    mkdir -p $DDIR/logs/logreg_val
    
    # nl = number of labeled images per class
    for nl in 1 2 5 10 20; do
        # grid search on 3 parameters
        for lr in 0.01 0.001 0.0001; do
            for wd in 0 0.01 0.1 ; do
                for bs in 16 64 128; do
                    # seed = performance is averaged over 5 seed values
                    for seed in {1..5}; do
                        name=nl$nl.lr$lr.wd$wd.bs$bs.seed$seed

                        # it is useful to run this in parallel on a cluster
                        # (CPU only)
                        
                        python logreg.py \
                               --mode val --maxiter 75000 \
                               --lr $lr --wd $wd \--batchsize $bs \
                               --seed $seed --nlabeled $nl |
                            tee $DDIR/logs/logreg_val/$name.stdout
                        
                    done
                done
            done
        done
    done

fi



if [ $stage == 2 ]; then
    
    python parse_logreg.py --mode val 

fi


if [ $stage == 3 ]; then
    # found by the previous script on the validation parameters
    # nl:lr,wd,bs,maxiter
    validated_params="
        1:0.001,0.1,128,47000
        2:0.001,0.01,128,29500
        5:0.001,0.01,64,70500
        10:0.01,0,128,12000
        20:0.01,0,128,22000"

    # run on test
    mkdir -p $DDIR/logs/logreg_test

    for nl in 1 2 5 10 20; do
        params=''
        for vp in $validated_params; do
            if [ ${vp%:*} == $nl ]; then
                params=${vp#*:}
                break
            fi
        done
        [ -z "$params" ] && (echo could not find params for nl=$nl; exit 1)
        params=( ${params//,/ } )
        lr=${params[0]}
        wd=${params[1]}
        bs=${params[2]}
        maxiter=${params[3]}

        for seed in {1..5}; do

            name=nl$nl.seed$seed

            python logreg.py \
                   --mode test --maxiter $maxiter \
                   --lr $lr --wd $wd --batchsize $bs --seed $seed --nlabeled $nl \
                   --storemodel $DDIR/logs/logreg_test/model.$name.pt \
                   --storeL     $DDIR/logs/logreg_test/L.$name.npy |
                tee             $DDIR/logs/logreg_test/$name.stdout

            # and re-run validation, beacause we will need the L matrix
            python logreg.py \
                   --mode val --maxiter $maxiter \
                   --lr $lr --wd $wd --batchsize $bs --seed $seed --nlabeled $nl \
                   --storemodel $DDIR/logs/logreg_val/model.$name.pt \
                   --storeL     $DDIR/logs/logreg_val/L.$name.npy |
                tee             $DDIR/logs/logreg_val/$name.stdout
            
        done


    done


fi


if [ $stage == 4 ]; then

    python parse_logreg.py --mode test

fi
