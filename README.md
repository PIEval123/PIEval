## GCG Attack on OpenPromptInjection and MMLU-PI  

### Directory: `./nanoGCG`  

#### Running the Attack:  
- **On OpenPromptInjection:**  
  ```bash
  python3 single_nanogcg.py --model <MODEL> --separator_loc <LOC> --adaptive <ADAPTIVE>
  ```  
- **On MMLU-PI:**  
  ```bash
  python3 single_nanogcg_mmlu.py --model <MODEL> --separator_loc <LOC> --adaptive <ADAPTIVE>
  ```  

#### Parameter Details:  
- **`model`**: Targeted LLM. Options:  
  - `llama3_secalign` (Llama-3-8B-SecAlign)  
  - `llama3_struq` (Llama-3-8B-StruQ)  
  - `llama3_base` (Llama-3-8B-undefended)  
  - `llama3_instruct` (Llama-3-8B-Instruct)  
- **`separator_loc`**: Location of separator. Options:  
  - `mid`  
- **`adaptive`**: Whether to perform an adaptive attack. Options:  
  - `0` (Existing GCG)  
  - `1` (Adaptive GCG)  

#### Output:  
- Results are saved in `./nanoGCG/results`.  
- You can calculate the final ASV using `./nanoGCG/generate_testing_data.ipynb`.
- The results presented in our paper are also stored in this directory.  

---  

## Adaptive Attack Against Attention Tracker  

### Directory: `./attention-tracker`  

#### Running the Attack:  
- **On OpenPromptInjection:**  
  ```bash
  python3 single_nanogcg_att_tracker.py --separator_loc <LOC>
  ```  
- **On MMLU-PI:**  
  ```bash
  python3 single_nanogcg_att_tracker_mmlu.py --separator_loc <LOC>
  ```  

#### Parameter Details:  
- **`separator_loc`**: Location of separator. Options:  
  - `fmid` (Separator only)  
  - `data` (Separator + Injected Instruction)  
  - `whole` (Separator + Injected Instruction + Injected Data)  

#### Output:  
- Results are saved in `./attention-tracker/results`.  
- You can calculate the final ASV using `./nanoGCG/generate_testing_data.ipynb`.
- The results presented in our paper are also stored in this directory.

## Effectiveness and Utility of Detection-Based Defense

### Attention Tracker
- Path: `cd attention-tracker`
- `python ./utility_check.py`

### PromptGuard
- Path: `cd prompt_guard`
- `python ./utility_check.py

## Relative Utility (Win Rate) for Prevention-Based Defense
- Path: `cd alpacaeval`
- `alpaca_eval --model_outputs alpacafarm_full_winrate_<evaluated_model>.json --reference_outputs alpacafarm_full_winrate_<reference_model>.json`


## Heuristic Attack on OpenPromptInjection and MMLU-PI

### OpenPromptInjection
- Path: `cd OpenPromptInjection`
- Parameters:
  - `MODEL`: The defended model evaluated. Refer to `model` parameter details in first section.
  - `ATTACK`: The heuristic attack method. Default setting is `combine` which is Combined Attack.
- `./defense_eval.sh`

### MMLU
- Path: `cd OpenPromptInjection`
- Parameters:
  - `model_config_path`: The defended model evaluated. Refer to `model` parameter details in first section.
  - `defend`: `True` for prevetion based defense defended models.
- `./python mmlu_eval.py`
