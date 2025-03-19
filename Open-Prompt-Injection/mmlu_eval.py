# %%
import re
import os
import time
import numpy as np
import json

import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config

def simple_extract_answer(
    response: str, 
    doc: dict,
    main_pattern: str = r"#### (\-?[0-9\.\,]+)",
    fallback: str = "[invalid]"
) -> str:
    """
    Extract a single multiple-choice answer from `response`, given `doc["choices"]`.
    It tries the main regex pattern first, then falls back to matching doc choices,
    and finally tries to detect references like 'A' or '(A)' in the text itself.
    If none match, returns `fallback`.
    
    Args:
        response (str): The model's response (e.g. "I think it's (B)")
        doc (dict):     A dictionary containing at least a key "choices",
                        e.g. {"choices": ["(A)", "(B)", "(C)", "(D)"]}.
        main_pattern (str): The primary regex pattern to attempt first.
        fallback (str): A default fallback string if no match is found.

    Returns:
        str: The extracted answer, typically "(A)", "(B)", "(C)", or "(D)".
    """

    def find_match(regex, text, convert_dict=None):
        """Helper: Return the first successful regex match (converted, if needed)."""
        convert_dict = convert_dict or {}
        matches = regex.findall(text)
        if matches:
            # By default, we just pick the first match
            match = matches[0]
            # If the match is a tuple of capture groups, pick the first nonempty one
            if isinstance(match, tuple):
                # e.g. if match = ('C', '') or ('', 'C')
                for m in match:
                    if m.strip():
                        match = m
                        break
            match = match.strip()
            # Optionally convert if there's a mapping dict
            return convert_dict.get(match, match)
        return None

    # 1) Try the "main_pattern" first
    main_regex = re.compile(main_pattern)
    answer = find_match(main_regex, response)
    if answer:
        return answer

    # 2) Build fallback patterns from doc["choices"], e.g. (A), (B), (C), (D)
    choice_to_alpha = {}
    fallback_regexes = []
    letters_only_regexes = []
    next_alpha_char = "A"

    for choice_text in doc["choices"]:
        # e.g. "(A)", "(B)", ...
        escaped_choice = re.escape(choice_text.strip())
        fallback_regexes.append(escaped_choice)

        # Map the original choice text to something like "(A)"
        choice_to_alpha[choice_text.strip()] = f"({next_alpha_char})"

        # Also keep track of the letter only (A, B, C, D)
        letters_only_regexes.append(next_alpha_char)

        # Move on to next letter
        next_alpha_char = chr(ord(next_alpha_char) + 1)

    # 2a) Fallback #1: match exact choice text (e.g. "(A)", "(B)", etc.)
    fallback_regex = re.compile("|".join(fallback_regexes))
    answer = find_match(fallback_regex, response, convert_dict=choice_to_alpha)
    if answer:
        return answer

    # 2b) Fallback #2: try to extract letter references from text
    #
    # We'll match scenarios like:
    #  - "The correct answer is (C)"
    #  - "The correct option would be C"
    #  - "C"
    #  - "(C)"
    # using an alternation in the regex. Note that we use two capture groups
    # so that whether the user wrote "C" or "(C)" we can unify it.
    #
    # Explanation of the pattern:
    #   - (?i) makes it case-insensitive
    #   - (?:correct\s+)?(?:answer|option)\s+(?:is|would\s+be)? is optional lead text
    #   - \(?([A-D])\)? captures (A) or A
    #   - The big alternation also tries to match a lone (A) or (B) or (C) or (D)
    #
    # If you find this pattern too aggressive, you can refine it further.
    letters_pattern = r"""(?ix)  # ignore case, verbose
        (?:                              # group for either:
           (?:correct\s+)?               # optional "correct "
           (?:answer|option)             # "answer" or "option"
           \s+(?:is|would\s+be)?         # optional "is" or "would be"
           \s*\(?([A-D])\)?              # capture letter with optional parens
         )
         |
         \(\s*([A-D])\s*\)              # OR a standalone "(C)"
         |
         \b([A-D])\b                    # OR just a standalone letter
    """
    # We will unify whichever group is nonempty.
    letter_regex = re.compile(letters_pattern)
    # We'll map raw "A" -> "(A)", raw "B" -> "(B)", etc.
    letter_to_paren = {}
    for letter in letters_only_regexes:
        letter_to_paren[letter] = f"({letter})"

    def convert_tuple_to_paren(match_tuple):
        """
        We'll receive up to 3 captures, e.g. ('C', '', '') or ('', 'C', '') or ('', '', 'C').
        Return the first nonempty one mapped to (X).
        """
        for group_val in match_tuple:
            g = group_val.strip()
            if g:
                # e.g. "C" => "(C)"
                return letter_to_paren.get(g.upper(), fallback)
        return None

    matches = letter_regex.findall(response)
    if matches:
        # The first match is a 3-tuple (some parts may be empty strings)
        paren = convert_tuple_to_paren(matches[0])
        if paren:
            return paren

    # 3) If all else fails, return the fallback
    return fallback

def calculate_accuracy(results):
    correct = 0
    for result in results:
        if result['expected'] == result['extracted']:
            correct += 1
    return correct / len(results)

# with open("/home/<username>/prompt-injection-eval/Open-Prompt-Injection/mmlu_sampled_200_base_secalign.json") as f:
#     results = json.load(f)

# print(f"Accuracy: {calculate_accuracy(results)}")
# assert False
# %%
model_config_path = "/home/<username>/prompt-injection-eval/Open-Prompt-Injection/configs/model_configs/gpt_ori_config.json"
model_config = open_config(config_path=model_config_path)
model = PI.create_model(config=model_config)

mmlu_sampled = json.load(open("/home/<username>/prompt-injection-eval/mmlu_eval/mmlu_sampled_200.json"))
# %%
def prepare_query(instruction, input, defend):
    if defend:
        return f'[MARK] [INST][COLN]\n{instruction}\n\n[MARK] [INPT][COLN]\n{input}\n\n[MARK] [RESP][COLN]\n'
    else:
        return f'{instruction}\n{input}\n'

results = []
for sample in mmlu_sampled:
    prompt = prepare_query(sample["instruction"], sample["input"], False)
    response = model.query(prompt)
    print(f"RESPONSE: {response}")
    print(f"EXPECTED: {sample['answer']}")
    print(f"EXTRACTED: {simple_extract_answer(response, sample)}")
    print()
    
    result = {
        'prompt': prompt,
        'response': response,
        'expected_encoded': sample['answer'],
        'expected': ['(A)', '(B)', '(C)', '(D)'][int(sample['answer'])],
        'extracted': simple_extract_answer(response, sample)
    }
    results.append(result)

print(f"Accuracy: {calculate_accuracy(results)}")

with open("mmlu_sampled_200_chatgpt4omini.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)