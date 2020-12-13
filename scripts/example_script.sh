#!/bin/bash

# Script to launch all experiments

width_multiplier=128

for dataset in cifar10 
do
    for framework in avae vae gan bigan 
    do 
        echo ${framework} ${width_multiplier} ${dataset}
        # Training
        python3 main.py \
            --experiment_name="test_width=${width_multiplier}" \
            --width_multiplier=${width_multiplier} \
            --dataset=${dataset} \
            --framework=${framework} \
        
        # Testing
        python3 main.py \
            --experiment_name="test_width=${width_multiplier}" \
            --mode=test_fid \
            --width_multiplier=${width_multiplier} \
            --dataset=${dataset} \
            --framework=${framework} \

        python3 main.py \
            --experiment_name="test_width=${width_multiplier}" \
            --mode=test_mse_lpips \
            --width_multiplier=${width_multiplier} \
            --dataset=${dataset} \
            --framework=${framework} \

    done    
done

for dataset in bedroom cifar10 cifar100 svhn celeba
do
    for framework in vaegan
    do  
        echo ${framework} ${width_multiplier} ${dataset}
        if [ ${dataset} == celeba ]; then
            beta=5
        fi
        if [ ${dataset} == bedroom ]; then
            beta=4
        fi
        if [ ${dataset} == svhn ]; then
            beta=20
        fi
        if [ ${dataset} == cifar10 ]; then
            beta=10
        fi
        if [ ${dataset} == cifar100 ]; then
            beta=10
        fi
        
        # Training
        python3 main.py \
            --experiment_name="test_width=${width_multiplier}" \
            --width_multiplier=${width_multiplier} \
            --dataset=${dataset} \
            --framework=${framework} \
            --beta=${beta} \

        # Testing
        python3 main.py \
            --experiment_name="test_width=${width_multiplier}" \
            --mode=test_fid \
            --width_multiplier=${width_multiplier} \
            --dataset=${dataset} \
            --framework=${framework} \

        python3 main.py \
            --experiment_name="test_width=${width_multiplier}" \
            --mode=test_mse_lpips \
            --width_multiplier=${width_multiplier} \
            --dataset=${dataset} \
            --framework=${framework} \

    done    
done