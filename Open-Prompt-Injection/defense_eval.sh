#!/bin/bash
MODEL='base'
MEMORY_THRESHOLD=19000 # in MiB
ATTACK='combine' # Attack strategy
NUM_GPUS=1 # Number of GPUs needed for each task

cd Open-Prompt-Injection

run_task() {
    INJECTED=$1
    TARGET=$2
    GPU_IDS=$3

    echo "Running task on GPUs: $GPU_IDS"
    CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py \
        --model_config_path /home/<username>/PoisonedAlign/Open-Prompt-Injection/configs/model_configs/llama3_${MODEL}_config.json \
        --target_data_config_path /home/<username>/PoisonedAlign/Open-Prompt-Injection/configs/task_configs/${TARGET}_config.json \
        --injected_data_config_path /home/<username>/PoisonedAlign/Open-Prompt-Injection/configs/task_configs/${INJECTED}_config.json \
        --data_num 100 \
        --save_path RESULTS-combine-${MODEL}/${INJECTED}/${TARGET} \
        --attack_strategy ${ATTACK} \
        --defense no
}

# Function to get available GPUs, excluding occupied ones
get_available_gpus() {
    local occupied_gpus=("$@")
    local available_gpus=()
    
    # Suppose you have 8 gpus
    for gpu in {0..0}; do
        if [[ ! " ${occupied_gpus[@]} " =~ " ${gpu} " ]]; then
            AVAILABLE_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu)
            if [ $AVAILABLE_MEMORY -ge $MEMORY_THRESHOLD ]; then
                available_gpus+=($gpu)
            fi
        fi
    done

    echo "${available_gpus[@]}"
}

# Define tasks (unchanged)
declare -a tasks=(
    "mrpc mrpc"
    "mrpc jfleg"
    "mrpc hsol"
    "mrpc rte"
    "mrpc sst2"
    "mrpc sms_spam"
    "mrpc gigaword"
    "jfleg mrpc"
    "jfleg jfleg"
    "jfleg hsol"
    "jfleg rte"
    "jfleg sst2"
    "jfleg sms_spam"
    "jfleg gigaword"
    "hsol mrpc"
    "hsol jfleg"
    "hsol hsol"
    "hsol rte"
    "hsol sst2"
    "hsol sms_spam"
    "hsol gigaword"
    "rte mrpc"
    "rte jfleg"
    "rte hsol"
    "rte rte"
    "rte sst2"
    "rte sms_spam"
    "rte gigaword"
    "sst2 mrpc"
    "sst2 jfleg"
    "sst2 hsol"
    "sst2 rte"
    "sst2 sst2"
    "sst2 sms_spam"
    "sst2 gigaword"
    "sms_spam mrpc"
    "sms_spam jfleg"
    "sms_spam hsol"
    "sms_spam rte"
    "sms_spam sst2"
    "sms_spam sms_spam"
    "sms_spam gigaword"
    "gigaword mrpc"
    "gigaword jfleg"
    "gigaword hsol"
    "gigaword rte"
    "gigaword sst2"
    "gigaword sms_spam"
    "gigaword gigaword"
)

# List of occupied GPU jobs
declare -a occupied_jobs=()
declare -A job_to_gpus
occupied_gpus=()

# Run tasks
while [ ${#tasks[@]} -gt 0 ]; do
    echo "Occupied GPUs: ${occupied_gpus[*]}"
    available_gpus=($(get_available_gpus "${occupied_gpus[@]}"))

    if [ ${#available_gpus[@]} -ge $NUM_GPUS ]; then
        task=${tasks[0]}
        selected_gpus=("${available_gpus[@]:0:$NUM_GPUS}") # Select the needed number of GPUs
        run_task $task "$(IFS=','; echo "${selected_gpus[*]}")" &

        # Wait for the task to start and capture the job ID
        job_id=$! # Capture the job ID
        if [ -z "$job_id" ]; then
            echo "Error: Job ID is empty, task may not have started properly."
            continue
        fi

        echo "Started task: $task on GPUs: ${selected_gpus[*]} (Job ID: $job_id)"

        # Add selected GPUs and job ID to the occupied lists
        occupied_gpus+=("${selected_gpus[@]}")
        occupied_jobs+=($job_id)
        job_to_gpus[$job_id]="${selected_gpus[*]}"

        unset tasks[0]
        tasks=("${tasks[@]}") # Re-index the array
    else
        echo "Not enough GPUs available, sleeping for 30 seconds..."
        # sleep 30
    fi


    # Clean up finished tasks from the occupied jobs
    new_occupied_jobs=()
    for job in "${occupied_jobs[@]}"; do
        if [[ -z "$job" ]]; then
            echo "Warning: encountered an empty job ID in occupied_jobs."
            continue
        fi

        if kill -0 "$job" 2>/dev/null; then
            # Job is still running
            new_occupied_jobs+=("$job")
        else
            # Job has finished
            if [[ -n "${job_to_gpus[$job]}" ]]; then
                for gpu in ${job_to_gpus[$job]}; do
                    occupied_gpus=("${occupied_gpus[@]/$gpu}")
                done
                unset job_to_gpus[$job]  # Clean up the mapping
            else
                echo "Warning: job_to_gpus does not contain entry for job ID: $job"
            fi
        fi
    done
    occupied_jobs=("${new_occupied_jobs[@]}")

    # Remove duplicates and maintain order in occupied_gpus
    occupied_gpus=($(echo "${occupied_gpus[@]}" | tr ' ' '\n' | awk '!seen[$0]++' | tr '\n' ' '))
done

wait