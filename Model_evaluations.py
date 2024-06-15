
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import numpy as np
import random
import json
import re

file_path = "Your_prediction_JSON"
file_path2 = "Your_truth_JSON"


# # load from JSON
with open(file_path, 'r') as file:
    loaded_list = json.load(file)

with open(file_path2, 'r') as file:
    loaded_list2 = json.load(file)

# ### for viccuna predictions only!!!

# # Your input string
# for i in range(len(loaded_list)):
#     if len(loaded_list)==1:
#         input_str=loaded_list[i][0]
#         output_str = re.sub(r'\\', '', input_str)
#         loaded_list[i][0]=output_str
#     elif len(loaded_list)>1:
#         for j in range(len(loaded_list[i])):
#             input_str=loaded_list[i][j]
#             output_str = re.sub(r'\\', '', input_str)
#             loaded_list[i][j]=output_str
# # Remove backslashes using re.sub()


tasks_list = ['long_time_series_prediction', 'time_series_prediction', 'event_prediction', 'trajectory_completion', 'trajectory_prediction', 'time_series_anomaly_detection', 'time_series_imputation', 'arrval_time_estimation', 'trajectory_forecasting', 'map_mapping', 'recommendation', 'spatial_relationship_infer', 'bus_arrival', 'taxi_availability']

def string_list_to_sorted_indices(string_list, unique_list):
    sorted_unique_values = unique_list
    value_to_sorted_index = {value: idx for idx, value in enumerate(sorted_unique_values)}
    sorted_indices_list = [value_to_sorted_index[value] for value in string_list]
    return sorted_indices_list

prediction_label_one_hot =[]
for i in range(len(loaded_list)):
# for i in range(3):
    if len(loaded_list[i])==1:
        if loaded_list[i][0] in tasks_list:
            prediction_label_one_hot.append(string_list_to_sorted_indices(loaded_list[i], tasks_list))
        else:
            prediction_label_one_hot.append([15])
    elif len(loaded_list[i])==0:
        prediction_label_one_hot.append([15])
    else:
        temp_list=[]
        for j in range(len(loaded_list[i])):
            if loaded_list[i][j] in tasks_list:
                temp_list.append(string_list_to_sorted_indices([loaded_list[i][j]], tasks_list)[0])
                # print(string_list_to_sorted_indices([loaded_list[i][j]], tasks_list))
            else:
                temp_list.append(15)
        # print(temp_list)
        prediction_label_one_hot.append(temp_list)

truth_label_one_hot =[]
for i in range(len(loaded_list2)):
# for i in range(3):
    if len(loaded_list2[i])==1:
        if loaded_list2[i][0] in tasks_list:
            truth_label_one_hot.append(string_list_to_sorted_indices(loaded_list2[i], tasks_list))
        else:
            truth_label_one_hot.append([15])
    elif len(loaded_list2[i])==0:
        truth_label_one_hot.append([15])
    else:
        temp_list=[]
        for j in range(len(loaded_list2[i])):
            if loaded_list2[i][j] in tasks_list:
                temp_list.append(string_list_to_sorted_indices([loaded_list2[i][j]], tasks_list)[0])
                # print(string_list_to_sorted_indices([loaded_list[i][j]], tasks_list))
            else:
                temp_list.append(15)
        # print(temp_list)
        truth_label_one_hot.append(temp_list)

print(len(prediction_label_one_hot))
print(len(truth_label_one_hot))

def pad_label(y_true, y_pred):
    len_diff = len(y_true) - len(y_pred)
    if len_diff > 0:
        y_pred.extend([15]* len_diff)
    elif len_diff < 0:
        y_true.extend([15] * (-len_diff))
    return y_true, y_pred

precision_list=[]
recall_list=[]
f1_list=[]
count=0
for i in range(len(prediction_label_one_hot)):
    y_true_padded, y_pred_padded = pad_label(truth_label_one_hot[i], prediction_label_one_hot[i])
    precision_list.append(precision_score(y_true_padded, y_pred_padded, average='macro', zero_division=0))
    recall_list.append(recall_score(y_true_padded, y_pred_padded, average='macro', zero_division=0))
    f1_list.append(f1_score(y_true_padded, y_pred_padded,average='macro'))
    if truth_label_one_hot[i] == prediction_label_one_hot[i]:
            count+=1

print(sum(precision_list)/len(precision_list))
print(sum(recall_list)/len(recall_list))
print(sum(f1_list)/len(f1_list))
print(count/len(prediction_label_one_hot))
