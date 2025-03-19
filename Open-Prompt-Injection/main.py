import argparse
import os
import time
import numpy as np

import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Open Prompt Injection Experiments')
    parser.add_argument('--model_config_path', default='./configs/model_configs/palm2_config.json', type=str)
    parser.add_argument('--target_data_config_path', default='./configs/task_configs/rte_config.json', type=str)
    parser.add_argument('--injected_data_config_path', default='./configs/task_configs/hsol_config.json', type=str)
    parser.add_argument('--data_num', default=100, type=int)
    parser.add_argument('--save_path', default='./result', type=str)
    parser.add_argument('--attack_strategy', default='combine', type=str)
    parser.add_argument('--api_key_pos', default=0, type=int)
    parser.add_argument('--defense', default='no', type=str)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    
    target_task_responses_path = f'{args.save_path}/target_task_responses.npz'
    # injected_task_responses_path = f'{args.save_path}/injected_task_responses.npz'
    attack_responses_path = f'{args.save_path}/attack_responses.npz'
    # documents = [target_task_responses_path, injected_task_responses_path, attack_responses_path]
    documents = [target_task_responses_path, attack_responses_path]
    '''
    Preparation
    '''
    target_task = PI.create_task(open_config(config_path=args.target_data_config_path), args.data_num)
    print("LOADING MODEL")
    model_config = open_config(config_path=args.model_config_path)
    
    if False in [os.path.exists(file) for file in documents]:
        assert (0 <= args.api_key_pos < len(model_config["api_key_info"]["api_keys"]))
        model_config["api_key_info"]["api_key_use"] = args.api_key_pos
        model = PI.create_model(config=model_config)
        model.print_model_info()
        target_app = PI.create_app(target_task, model, log_path=f'{args.save_path}/query.log', defense=args.defense)
        
    inject_task = PI.create_task(open_config(config_path=args.injected_data_config_path), args.data_num, for_injection=True)
    attacker = PI.create_attacker(args.attack_strategy, inject_task)

    '''
    Experiments
    '''
    try:
        target_task_responses = np.load(target_task_responses_path, allow_pickle=True)['data']
    except:
        target_task_responses = []
        for i, (data_prompt, ground_truth_label) in enumerate(target_app):
            response = target_app.query(data_prompt, verbose=1, idx=i, total=len(target_app))
            target_task_responses.append(response)
            if i % 2 == 0:
                time.sleep(1)
        np.savez(target_task_responses_path, data=target_task_responses)

    if args.defense == 'no':
        try:
            injected_task_responses = np.load(injected_task_responses_path, allow_pickle=True)['data']
            # print(injected_task_responses)
        except:
            injected_task_responses = []
            for i, (data_prompt, ground_truth_label) in enumerate(attacker.task):
                response = model.query(attacker.task.get_instruction() + f'\nText: {data_prompt}')
                injected_task_responses.append(response)
                # print(i, response)
                if i % 2 == 0:
                    time.sleep(1)
            np.savez(injected_task_responses_path, data=injected_task_responses)
    else:
        injected_task_responses = None

    try:
        attack_responses = np.load(attack_responses_path, allow_pickle=True)['data']
    except:
        attack_responses = []

        for i, (data_prompt, ground_truth_label) in enumerate(target_app):
            data_prompt_after_attack = attacker.inject(data_prompt, i, target_task=target_task.task)
            response = target_app.query(data_prompt_after_attack, verbose=1, idx=i, total=len(target_app))
            attack_responses.append(response)
            if i % 2 == 0:
                time.sleep(1)
        np.savez(attack_responses_path, data=attack_responses)

    '''
    Evaluation
    '''
    evaluator = PI.create_evaluator(
        target_task_responses=target_task_responses, 
        target_task=target_task,
        injected_task_responses=injected_task_responses, 
        injected_task=attacker.task,
        attack_responses=attack_responses
    )

    # info = f"PNA-T = {evaluator.pna_t}\nPNA-I = {evaluator.pna_i}\nASV = {evaluator.asv}\nMR = {evaluator.mr}\nSOFT_ASV = {evaluator.soft_asv}\nSOFT_MR = {evaluator.soft_mr}\n"
    info = f"PNA-T = {evaluator.pna_t}\nASV = {evaluator.asv}\nSOFT_ASV = {evaluator.soft_asv}\n"
    
    print(info)
    with open(f'{args.save_path}/result.log', "w") as f:
        f.write(f'Model Base: {model_config["model_info"]["name"]}\n')
        f.write(f'Model Position: {model_config["model_info"]["path"]}\n')
        f.write(f'Attack: {args.attack_strategy}\n\n')
        f.write(info)
    print('[END]')