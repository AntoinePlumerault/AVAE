for width_multiplier in 16
do
    for dataset in celeba
    do
        for framework in avae
        do 
            echo ${framework} ${width_multiplier} ${dataset} V
            python3 main.py \
                --experiment_name="sigma=1_width=${width_multiplier}_V" \
                --width_multiplier=${width_multiplier} \
                --dataset=${dataset} \
                --framework=${framework} \
                --final_step=1000 \
                --beta=1 \
                --eval_freq=1000
        done    
    done
done