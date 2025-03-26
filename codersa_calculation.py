import json
import vllm
import re
import textwrap
from vllm import LLM, SamplingParams
from tqdm import tqdm

file_path = 'data/bootstrap_output.json'

try:
    with open(file_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"


llm = LLM(model=model_path, tensor_parallel_size=2, dtype='half')


description_generation_params = SamplingParams(n = 1,
                temperature=0.3,
                max_tokens = 128,)


sampling_params = SamplingParams(n = 1,
                temperature=1,
                max_tokens = 1,
                prompt_logprobs = 1)

def clean_code(block):
    # Remove comments
    block = re.sub(r'#.*$', '', block, flags=re.MULTILINE)

    # Remove docstrings
    block = re.sub(r'''"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'''', '', block)

    # Remove whitespace and empty lines
    block = re.sub(r'\n\s*\n', '\n', block)
    block = re.sub(r'^\s*\n', '', block, flags=re.MULTILINE)

    return block
#############################
def rename_function(text):
    pattern = r'def\s+([^\(]+)\('

    def replacer(match):
        return 'def f('

    new_text = re.sub(pattern, replacer, text)
    return new_text
#############################

def make_prompt_description(func):
  func_clean = clean_code(func)
  background = '''Follow the example, write a description for given python function. Just return description, not anything else.
Example function:
if isinstance(x,int) and isinstance(y,int) and isinstance(z,int): if (x+y==z) or (x+z==y) or (y+z==x): return True return False return False
Example description: Create a function that takes 3 numbers. Returns true if one of the numbers is equal to the sum of the other two, and all numbers are integers. Returns false in any other cases.
Function:
'''
  return background + rename_function(func_clean) + '''\nDescription:'''

def process_string(input_string):
    input_string = input_string.lstrip()

    if input_string.startswith('```'):
        start_index = 3 
        end_index = input_string.find('```', start_index)
        if end_index != -1:
            return input_string[start_index:end_index].strip()

    newline_index = input_string.find('\n')
    if newline_index != -1:
        return input_string[:newline_index]
    else:
        return input_string

for epoch in data:
  for question in epoch:
    for function in question['output']:
      generate_prompt = make_prompt_description(function['Function'])
      outputs = llm.generate(generate_prompt, description_generation_params)
      print(process_string(outputs[0].outputs[0].text))
      function['Generated_Description'] = process_string(outputs[0].outputs[0].text)

output_file = "data/CodeRSA_with_instructions.json"

with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)


#############
#CodeRSA#########
#############

import re
import numpy as np

def split_index(lst):

  first_index = None
  count = 0

  for i, item in enumerate(lst):
    if item == 'def':
      count += 1
      if count == 1:
          first_index = i
          break
  return first_index

def make_rsa_prompt(func, description):
  func_clean = clean_code(func)
  background = '''Follow the description, write a python function.
Description:'''
  return background + description + '''\nFunction:\n''' + func_clean

def extract_description(func_code):
    docstring_pattern = re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL)
    docstring_match = docstring_pattern.search(func_code)

    if docstring_match:
        docstring = docstring_match.group(0)
        # Split the docstring by 'Examples' or 'For example'
        split_pattern = re.compile(r'''Examples|For example|>>>|Example''', re.IGNORECASE)
        parts = split_pattern.split(docstring)
        if parts:
            description = parts[0]
            # Clean up the triple quotes and extra spaces/newlines
            description = description.strip('\'"')
            description = re.sub(r'\s+', ' ', description).strip()
            return description
    return ""

def normalize_log_probabilities(log_probs):
    log_probs = np.array(log_probs, dtype=float)  # Convert to numpy array for vectorized operations

    # Convert log probabilities to probabilities
    probs = np.exp(log_probs)

    total_sum = np.sum(probs)

    if total_sum == 0:
        return [0]

    normalized_probs = probs / total_sum
    return normalized_probs.tolist()

def get_start_index_rsa(outputs):
  start = 0
  for n, token in enumerate(outputs[0].prompt_logprobs):
    if token != None:
      if next(iter(token.values())).decoded_token == 'Function' and next(iter(outputs[0].prompt_logprobs[n+1].values())).decoded_token == ':\n':
        start = n+2
  return start

for epoch in tqdm(data):
  for question in epoch:
    for function in question['output']:
      rsa_prob = []
      func_clean = clean_code(function['Function'])
      ori_description = extract_description(question['input_prompt'])
      ori_prompt = make_rsa_prompt(func_clean, ori_description)
      outputs = llm.generate(ori_prompt, sampling_params)
      ori_start = get_start_index_rsa(outputs)
      rsa_prob.append(sum([next(iter(token.values())).logprob for token in outputs[0].prompt_logprobs[ori_start :]]))

      for item in question['output']:
        prompt = make_rsa_prompt(func_clean, item['Generated_Description'])
        outputs = llm.generate(prompt, sampling_params)
        start = get_start_index_rsa(outputs)
        rsa_prob.append(sum([next(iter(token.values())).logprob for token in outputs[0].prompt_logprobs[start :]]))
      
      function['RSA_Prob'] = rsa_prob
      function['RSA_Norm_Prob'] = normalize_log_probabilities(rsa_prob)



output_file = "data/CodeRSA_Result.json"

with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)