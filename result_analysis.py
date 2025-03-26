import numpy as np
import json
import random
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description='Specify file paths for CoderReviewer and CodeRSA JSON files.')
parser.add_argument('CoderReviewer', help='File path for CoderReviewer results (JSON)')
parser.add_argument('CodeRSA', help='File path for CodeRSA results (JSON)')

args = parser.parse_args()

with open(args.CoderReviewer, 'r', encoding='utf-8') as file:
    data_00 = json.load(file)

with open(args.CodeRSA, 'r', encoding='utf-8') as file:
    data_01 = json.load(file)

    
def sum_to_one_normalize(lst):
    total_sum = sum(lst)
    if total_sum == 0:
        return [0] * len(lst)
    return [x / total_sum for x in lst]



for index_epoch, epoch in enumerate(data_01):
  for index_question, question in enumerate(epoch):
    for index_answer, answer in enumerate(question['output']):
      answer['RSA_Norm_Prob'] = sum_to_one_normalize(answer['RSA_Prob'])

def first_element_rank(lst):
    if len(lst) == 0:
        return None  

    first_element = lst[0]  
    sorted_list = sorted(lst, reverse=True)  
    rank = sorted_list.index(first_element)  
    return rank

def first_element_rank_inverse(lst):
    if len(lst) == 0:
        return None  

    first_element = lst[0] 
    sorted_list = sorted(lst, reverse=False) 
    rank = sorted_list.index(first_element) 
    return rank


epoch_list = []
for index_epoch, epoch in enumerate(data_00):
  output_list= []
  for index_question, question in enumerate(epoch):
    coder_list = []
    reviewer_list = []
    coder_reviewer_list = []
    rsa_list = []
    pass_list = []
    for index_answer, answer in enumerate(question['output']):
      coder_list.append(answer['Coder_prob'])
      reviewer_list.append(answer['Reviewer_prob'])
      coder_reviewer_list.append(answer['Coder_Reviewer_prob'])
      rsa_list.append(data_01[index_epoch][index_question]['output'][index_answer]['RSA_Norm_Prob'][0])
      pass_list.append(answer['pass'])
    #print(rsa_list)
    output_list.append({
        'coder_index': coder_list.index(max(coder_list)),
        'reviewer_index': reviewer_list.index(max(reviewer_list)),
        'coder_reviewer_index': coder_reviewer_list.index(max(coder_reviewer_list)),
        'rsa_index': rsa_list.index(min(rsa_list)),

        'coder_pass': pass_list[coder_list.index(max(coder_list))],
        'reviewer_pass': pass_list[reviewer_list.index(max(reviewer_list))],
        'coder_reviewer_pass': pass_list[coder_reviewer_list.index(max(coder_reviewer_list))],
        'random_pass' : pass_list[random.randint(0, len(coder_list)-1)],
        'rsa_pass': pass_list[rsa_list.index(min(rsa_list))],
        'gold2coder': first_element_rank(coder_list),
        'gold2reviewer': first_element_rank(reviewer_list),
        'gold2coder_reviewer': first_element_rank(coder_reviewer_list),
        'gold2rsa': first_element_rank_inverse(rsa_list),
        'rsa_list': rsa_list,
    })
  epoch_list.append(output_list)
  
  
coder_reviewer_list = []
coder_list = []
rsa_list = []
reviewer_list = []
for epoch in epoch_list:
  for question in epoch:
    coder_reviewer_list.append(question['coder_reviewer_pass'])
    coder_list.append(question['coder_pass'])
    rsa_list.append(question['rsa_pass'])
    reviewer_list.append(question['reviewer_pass'])
    

acc_list = []
for epoch in epoch_list:
  coder_acc = 0
  reviewer_acc = 0
  coder_reviewer_acc = 0
  rsa_acc = 0
  random_acc = 0
  for question in epoch:
    if question['coder_pass'] == 100.0:
      coder_acc += 1
    if question['reviewer_pass'] == 100.0:
      reviewer_acc += 1
    if question['coder_reviewer_pass'] == 100.0:
      coder_reviewer_acc += 1
    if question['rsa_pass'] == 100.0:
      rsa_acc += 1
    if question['random_pass'] == 100.0:
      random_acc += 1
  acc_list.append([coder_acc/50, reviewer_acc/50, coder_reviewer_acc/50, rsa_acc/50, random_acc/50])
    

plt.figure(figsize=(8, 6))
data = [rsa_list, coder_list, coder_reviewer_list] # Care!
plt.violinplot(data, showmeans=True, showmedians=False)
plt.title('Violin plot of accuracy for different methods')
plt.xticks([1, 2, 3], ['rsa', 'coder', 'coder_reviewer'])
plt.ylabel('Accuracy')
plt.show()



# Creating a DataFrame for visualization
df = pd.DataFrame(acc_list, columns=[ 'Coder Accuracy','Reviewer Accuracy', 'Coder_Reviewer Accuracy','CodeRSA','random'])

# Creating the bar plot with the new data and updated color scheme
df.plot(kind='bar', figsize=(10, 6), color=['#6A5ACD', '#20B2AA', '#FFD700', '#FF6347', '#8FBC8F'])

# Setting labels and title
plt.xlabel('Epoch Index')
plt.ylabel('Accuracy')
plt.title('Accuracy of different methods over epochs')
plt.xticks(rotation=0)  # Rotate x labels for better readability

# Displaying the plot
plt.show()
