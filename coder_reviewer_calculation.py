import json
import vllm
import re
import textwrap
from vllm import LLM, SamplingParams

file_path = 'data/bootstrap_output.json'

try:
    with open(file_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    


def clean_code(block):
    # Remove comments
    block = re.sub(r'#.*$', '', block, flags=re.MULTILINE)

    # Remove docstrings (both triple double quotes and triple single quotes)
    block = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '', block, flags=re.MULTILINE)

    # Remove leading and trailing whitespace
    block = block.strip()

    # Remove empty lines and excessive whitespace
    block = re.sub(r'\n\s*\n', '\n', block)

    return block


def rename_function(text):
    pattern = r'def\s+([^\(]+)\('

    def replacer(match):
        return 'def f('

    new_text = re.sub(pattern, replacer, text)
    return new_text


def extract_docstring(code):
    pattern = r'(""".*?"""|\'\'\'.*?\'\'\')'
    match = re.search(pattern, code, re.DOTALL)
    if match:
        docstring = match.group(1)
        if docstring.startswith('"""') and docstring.endswith('"""'):
            docstring = docstring[3:-3]
        elif docstring.startswith("'''") and docstring.endswith("'''"):
            docstring = docstring[3:-3]
        return textwrap.dedent(docstring)
    else:
        return None

def get_start_index(outputs):
  start = 0
  for n, token in enumerate(outputs[0].prompt_logprobs):
    if token != None:
      if next(iter(token.values())).decoded_token == '""' and next(iter(outputs[0].prompt_logprobs[n+1].values())).decoded_token == '"':
        start = n+2
        break
  return start

def get_start_index_Coder(outputs):
  start = 0
  for n, token in enumerate(outputs[0].prompt_logprobs):
    if token != None:
      if next(iter(token.values())).decoded_token == '""' and next(iter(outputs[0].prompt_logprobs[n+1].values())).decoded_token == '"':
        start = n+2
  return start

def Format2LLAMA (f, input_prompt):
  ######
  clean_f = rename_function(clean_code(f))
  #clean_f = clean_code(f)
  ######
  comment = extract_docstring(input_prompt)
  extra_prompt = "\n# write a docstring for the  above function\n"

  return clean_f + extra_prompt + '""" ' + comment + '"""'

def Format2LLAMA_Coder (f, input_prompt):
  #########clean_f = rename_function(clean_code(f))
  clean_f = rename_function(clean_code(f))
  #########
  comment = extract_docstring(input_prompt)


  return '""" ' + comment + '"""' +'\n' +clean_f

# Load the model
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"


llm = LLM(model=model_path, tensor_parallel_size=2)


sampling_params = SamplingParams(n = 1, 
                                 temperature=1, 
                                 max_tokens = 1,
                                 prompt_logprobs = 1)


for epoch in data:
  for index, question in enumerate(epoch):
    for _ in question['output']:
      Reviewer_prob = 0
      func = _['Function']
      input_prompt = question['input_prompt']

      try:
        f2LLAMA2 = Format2LLAMA(func, input_prompt)
      except Exception as e:
        print(f"{index}：Error occurred: {e}")
      
      outputs = llm.generate(f2LLAMA2, sampling_params)
      start = get_start_index(outputs)
      reviewer_prob = [next(iter(token.values())).logprob for token in outputs[0].prompt_logprobs[start : -2]]
      _['Reviewer_prob'] = sum(reviewer_prob)

      try:
        f2LLAMA2_Coder = Format2LLAMA_Coder(func, input_prompt)
      except Exception as e:
        print(f"{index}：Error occurred: {e}")
      
      outputs = llm.generate(f2LLAMA2_Coder, sampling_params)
      start = get_start_index_Coder(outputs)
      coder_prob = [next(iter(token.values())).logprob for token in outputs[0].prompt_logprobs[start+1 :]]
      _['Coder_prob'] = sum(coder_prob)

      _['Coder_Reviewer_prob'] = _['Reviewer_prob'] + _['Coder_prob']

output_file = "data/CoderReviewer_result.json"

with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

