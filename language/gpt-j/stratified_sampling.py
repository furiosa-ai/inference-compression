import matplotlib.pyplot as plt
import numpy as np
import yaml
import numpy as np
import random
import json 

rouge = []
eval_data = []
acc_log = []
num_part = 4

for i in range(num_part):
    rouge_yaml_file_name = 'result_qlevel_4_cnn_eval_part_' + str(i) + '.yaml'
    json_file = open('data/cnn_eval_part_' + str(i) + '.json')
    log_file = open('build/logs/cnn_eval_part_' + str(i) + '_qlevel4_minmax' +  '/mlperf_log_accuracy.json')
    summary_file = 'summary_target_qlevel4_cnn_eval_part_' + str(i) + '.yaml'
    
    with open(rouge_yaml_file_name) as f:
        rouge_per_part = yaml.load(f, Loader=yaml.FullLoader)
    eval_data.append(json.load(json_file))
    acc_log.append(json.load(log_file))
    rouge.append(rouge_per_part)
    



total_rouge = {'rouge1': [],
               'rouge2': [],
               'rougeL': [],
               'rougeLsum': [],
               }

for i in range(len(rouge)):
    for key, value in rouge[i].items():
        total_rouge[key] += value
    
total_data_len = len(total_rouge['rouge2'])
data_per_part = int(total_data_len / num_part)


bins = np.linspace(0.0,1.0,5)

rouge_1 = total_rouge['rouge1']
rouge_2 = total_rouge['rouge2']
rouge_L = total_rouge['rougeL']


rouge_2_dict = {k:v for v, k in enumerate(rouge_2)}
rouge_2.sort()
    
hist_rouge_1, _ = np.histogram(rouge_1, bins)
hist_rouge_2, bin_edges = np.histogram(rouge_2, bins)
hist_rouge_L, _ = np.histogram(rouge_L, bins)
plt.plot(bins[:-1], hist_rouge_1)
plt.plot(bins[:-1], hist_rouge_2)
plt.plot(bins[:-1], hist_rouge_L)


percentage_per_beam  = hist_rouge_2/hist_rouge_2.sum()

sampled_data = []
index_sampled_data=[]
sample_size = 100 
plt.show()

for i in range(len(hist_rouge_2)):
    bin_samples = random.sample(rouge_2[0:hist_rouge_2[i]-1], int(percentage_per_beam[i]*sample_size))    
    # Append the samples to the overall list
    rouge_2 = rouge_2[hist_rouge_2[i]:]
    sampled_data.extend(bin_samples)
    index_sampled_data = index_sampled_data + [rouge_2_dict[num] for num in bin_samples]

stratified_eval_data = []
for selected_idx in index_sampled_data:
    qsl_idx = acc_log[selected_idx//data_per_part][selected_idx % data_per_part]['qsl_idx']
    stratified_eval_data.append(eval_data[selected_idx//data_per_part][qsl_idx])
    
    
with open('./data/stratified_eval_data.json', 'w') as json_file:
    json.dump(stratified_eval_data, json_file, indent='\t', separators=(',', ': '))


