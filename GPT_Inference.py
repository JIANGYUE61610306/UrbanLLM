import os
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import re


hf_api_token = 'Your_Hugging_Face_API'


file_path = "Your_evaluation_data.txt"

loaded_list = []

# Open the file in read mode
with open(file_path, "r") as file:
    # Read each line of the file
    for line in file:
        # Strip whitespace from the line
        line = line.strip()
        # Check if the line is not empty
        if line:
            try:
                # Convert the line to an integer and append it to the list
                loaded_list.append(line)
            except ValueError:
                # Handle the case when the line cannot be converted to an integer
                print("Warning: Skipped line as it cannot be converted to an integer:", line)

# print("Loaded list:", loaded_list)
len(loaded_list)

dataset1_one_ans=[]
text = '<s>[INST]{}Q: {}[/INST]A: {}</s>'
instruction_note = "You are UAPLLM, a Large language model for urban activity that decompose user input into a list of spatio-temporal tasks with the following JSON format: {task: task_name, id: task_id, dep: dependency_task_ids, args: {domain: string, location_name_list: list, location_gps_list: list, time: time, input: sequence, service_no: int, bus_stop: string, task_specific: list}}. Task-specific information is stored using a string list and is used in specific models or tools as arguments. The dep field denotes the ID of the previous task that the current task relies. A dep field value of '-1' indicates that the current task does not rely on other tasks and can be executed immediately, otherwise always execute the previous task first. The executed output, such as the generated location, time, sequence, or POIs from the dependency task is marked as <resource>-task_id. This resource will be used in the next task. The spatio-temporal tasks must be selected from the following options: {long_time_series_prediction, time_series_prediction, event_prediction, trajectory_completion, trajectory_prediction, time_series_anomaly_detection, time_series_imputation, arrival_time_estimation, trajectory_forecasting, map_mapping, recommendation, spatial_relationship_inference, bus_arrival, taxi_availability}. To better understand each spatial-temporal task, here is the explanation and numbering, along with the corresponding examples: 1) Long Time Series Prediction: This task in- volves forecasting future values in a time series over a long horizon. It is typically used for long- term planning and trend analysis in various do- mains, such as weather forecasting, economic fore- casting, and demand planning. 2) Time Series Prediction: This task focuses on predicting future values in a time series over a shorter horizon compared to long time series predic- tion. It is commonly used for short-term forecasts like daily stock prices, temperature forecasts, or short-term sales predictions. 3) Event Prediction: This task involves predict- ing the occurrence of specific events based on his- torical data. Examples include predicting natural disasters, equipment failures, or social events like concerts or sports games. 4)Trajectory Completion: This task involves completing missing parts of a trajectory based on observed segments. It is useful in applications like tracking moving objects, filling in missing GPS data, or reconstructing incomplete travel routes. 5)Trajectory Prediction: This task involves fore- casting the future path of a moving object based on its past trajectory. Applications include predicting the movement of vehicles, pedestrians, or animals. 6)Time Series Anomaly Detection: This task in- volves identifying unusual patterns or outliers in time series data that deviate from expected behav- ior. It is used in applications like fraud detection, fault detection in machinery, and monitoring traffic conditions. 7) Time Series Imputation: This task involves filling in missing values in time series data to en- sure completeness and consistency. It is crucial for maintaining data quality in various applications like traffic records and climate data. 8)Arrival Time Estimation: This task involves predicting the arrival time of a vehicle or person at a specific location based on current and historical data. It is commonly used in transportation systems for buses, trains, and delivery services. 9) Taxi Availability Prediction: This task in- volves predicting the availability of taxis in spe- cific areas at given times. It helps optimize taxi dispatching and improve service for passengers by anticipating demand and ensuring timely availabil- ity. 10) Map Mapping: This task involves mapping addresses to GPS locations and mapping GPS loca- tionsback to addresses . 11) Bus Arrival: This task involves predicting the arrival times of buses at specific stops based on real-time data and historical patterns. It enhances the efficiency of public transportation systems by providing accurate and timely information to com- muters. 12) Spatial Relationship Inference: This task involves deducing spatial relationships between different entities or locations. It is used in urban planning to understand spatial dependencies and interactions, such as proximity analysis, clustering, and spatial correlations. 13) Recommendation: This task involves sug- gesting items or actions to users based on their preferences and historical behavior. Applications include recommending points of interest, routes, or services in urban planning. Please note that there exists a logical connection and order between the tasks. 1) If user input mentioned some specific locations/POIs, usually map mapping task should be included in the answer, otherwise if user is only asking the situation around all Singapore, map mapping task should not be included. 2) If user do not specify the arrival time, usually estimated arrival time task should be included. 3) Include tasks to predict weather and PM2.5 according to user input if user is going to outdoor activities. 4) When recommendation or taxi_availability task is included, usually map mapping task should be included as fundamental task. In case the user input cannot be parsed, an empty JSON response should be provided Please provide a task analysis JSON response basd on the given question."
for i in range(len(loaded_list)):
    pattern = r'Q:\s*(.*?)\s+A:\s*(.*)'
    data=loaded_list[i]
    matches = re.findall(pattern, data)

    for match in matches:
        # index = match[0]
        question = match[0]
        answer = match[1]
        # print(f"Q: {question}\nA: {answer}\n")
        # print(f"Q: {question}")
        # print(text.format(instruction_note, question, answer))
        dataset1_one_ans.append(text.format(instruction_note, question, answer))

from datasets import Dataset, DatasetDict
# my_dict = {'text':['1', '2', '3']}
# my_dict = {'text':dataset1_one_ans[:1000]}
my_dict = {'text':dataset1_one_ans}
dataset1 = Dataset.from_dict(my_dict)


import os
import openai
import sys
# import json
import numpy as np
import pandas as pd 
import random
# import re
from tqdm import tqdm
import time
import re

seed='2'

#### OpenAI
import openai
from openai import OpenAI


# import json
import os
import re

def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


def completion_with_backoff(**kwargs):
    key_list=["Your_Key_List"]
    client = OpenAI(api_key = random.choice(key_list))
    return client.chat.completions.create(**kwargs)

def get_completion(prompt, gpt_model="gpt-3.5-turbo",max_tokens=128):
    if gpt_model=='gpt3.5':
        model ='gpt-3.5-turbo-0613'
    elif gpt_model=='gpt4':
        model ='gpt-4o'

    messages = [{"role": "user", "content": prompt}]
    response = completion_with_backoff(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
        max_tokens=max_tokens,
        seed = int(seed)
    )
    return response.choices[0].message.content.strip()


instruction_note = "You are UAPLLM, a Large language model for urban activity that decompose user input into a list of spatio-temporal tasks with the following JSON format: {task: task_name, id: task_id, dep: dependency_task_ids, args: {domain: string, location_name_list: list, location_gps_list: list, time: time, input: sequence, service_no: int, bus_stop: string, task_specific: list}}. Task-specific information is stored using a string list and is used in specific models or tools as arguments. The dep field denotes the ID of the previous task that the current task relies. A dep field value of '-1' indicates that the current task does not rely on other tasks and can be executed immediately, otherwise always execute the previous task first. The executed output, such as the generated location, time, sequence, or POIs from the dependency task is marked as <resource>-task_id. This resource will be used in the next task. The spatio-temporal tasks must be selected from the following options: {long_time_series_prediction, time_series_prediction, event_prediction, trajectory_completion, trajectory_prediction, time_series_anomaly_detection, time_series_imputation, arrival_time_estimation, trajectory_forecasting, map_mapping, recommendation, spatial_relationship_inference, bus_arrival, taxi_availability}. To better understand each spatial-temporal task, here is the explanation and numbering, along with the corresponding examples: 1) Long Time Series Prediction: This task in- volves forecasting future values in a time series over a long horizon. It is typically used for long- term planning and trend analysis in various do- mains, such as weather forecasting, economic fore- casting, and demand planning. 2) Time Series Prediction: This task focuses on predicting future values in a time series over a shorter horizon compared to long time series predic- tion. It is commonly used for short-term forecasts like daily stock prices, temperature forecasts, or short-term sales predictions. 3) Event Prediction: This task involves predict- ing the occurrence of specific events based on his- torical data. Examples include predicting natural disasters, equipment failures, or social events like concerts or sports games. 4)Trajectory Completion: This task involves completing missing parts of a trajectory based on observed segments. It is useful in applications like tracking moving objects, filling in missing GPS data, or reconstructing incomplete travel routes. 5)Trajectory Prediction: This task involves fore- casting the future path of a moving object based on its past trajectory. Applications include predicting the movement of vehicles, pedestrians, or animals. 6)Time Series Anomaly Detection: This task in- volves identifying unusual patterns or outliers in time series data that deviate from expected behav- ior. It is used in applications like fraud detection, fault detection in machinery, and monitoring traffic conditions. 7) Time Series Imputation: This task involves filling in missing values in time series data to en- sure completeness and consistency. It is crucial for maintaining data quality in various applications like traffic records and climate data. 8)Arrival Time Estimation: This task involves predicting the arrival time of a vehicle or person at a specific location based on current and historical data. It is commonly used in transportation systems for buses, trains, and delivery services. 9) Taxi Availability Prediction: This task in- volves predicting the availability of taxis in spe- cific areas at given times. It helps optimize taxi dispatching and improve service for passengers by anticipating demand and ensuring timely availabil- ity. 10) Map Mapping: This task involves mapping addresses to GPS locations and mapping GPS loca- tionsback to addresses . 11) Bus Arrival: This task involves predicting the arrival times of buses at specific stops based on real-time data and historical patterns. It enhances the efficiency of public transportation systems by providing accurate and timely information to com- muters. 12) Spatial Relationship Inference: This task involves deducing spatial relationships between different entities or locations. It is used in urban planning to understand spatial dependencies and interactions, such as proximity analysis, clustering, and spatial correlations. 13) Recommendation: This task involves sug- gesting items or actions to users based on their preferences and historical behavior. Applications include recommending points of interest, routes, or services in urban planning. Please note that there exists a logical connection and order between the tasks. 1) If user input mentioned some specific locations/POIs, usually map mapping task should be included in the answer, otherwise if user is only asking the situation around all Singapore, map mapping task should not be included. 2) If user do not specify the arrival time, usually estimated arrival time task should be included. 3) Include tasks to predict weather and PM2.5 according to user input if user is going to outdoor activities. 4) When recommendation or taxi_availability task is included, usually map mapping task should be included as fundamental task. In case the user input cannot be parsed, an empty JSON response should be provided Please provide a task analysis JSON response basd on the given question. Here are several cases for your reference: (1) Q: I want to go to Jurong East for dinner and I will leave from Boon Lay Mrt station at 6PM, where can I park? A: [{task: time_series_prediction, id: 0, dep: [1], args: {location_gps_list: <resource>-2, time: <resource>-1, Input:history_steps }}, {task: arrval_time_estimation, id: 1, dep: [2], args: {location_gps_list: <resource>-2}},{task: map_mapping, id: 2, dep: [-1], args: {location_name_list:['Jurong East', 'lake side'] }}]. (2)  Q: I want to go to Jurong East for dinner at around 7PM, where can I park? A: [{task: time_series_prediction, id: 0, dep: [1], args: {location_gps_list: <resource>-1, Input:history_steps }}, {task: map_mapping, id: 1, dep: [-1], args: {location_name_list:['Jurong East'] }}].(3) Q: Do you have any bicycle parking location recommended nearby Lake Garden? A: [{task: recommendation, id: 0, dep: [1], args: {location_gps_list: <resource>-1}, task_specific:['bycycle_parking']}, {task: map_mapping, id: 1, dep: [-1], args: {location_name_list:['Lake Garden'] }}]. (4) Q:I would like to go to Starbucks@J-walk, is it here? 3 Gateway Dr. #02-04/04A Westgate, Singapore 608532. A: [{task: spatial_relationship_infer, id: 0, dep: [1], args: {location_gps_list: <resource>-1}}, {task: map_mapping, id: 1, dep: [-1], args: {location_name_list:['Starbucks@J-walk', '3 Gateway Dr. #02-04/04A Westgate, Singapore 608532'] }}]. (5) Q: I am waiting at the bus stop: 83139. When will be the next No. 15 bus coming? A: [{task: bus_arrival, id: 0, dep: [-1], args: {bus_stop: '83139', service_no: 15, task_specific:'next'}}]. (6) Q: I would like to aboard bus no.15 at the bus stop: 83139. How would the bus crowd situation be for the next 30 mins? A: [{task: bus_arrival, id: 0, dep: [-1], args: {bus_stop: '83139', service_no: 15, task_specific:'next 30 mins'}}]. (7) Q: My current location is inside Jem shopping centre, where are nearby taxi stands? A: [{task: recommendation, id: 0, dep: [1], args: {location_gps_list: <resource>-1},task_specific:['taxi_stand']}, {task: map_mapping, id: 1, dep: [-1], args: {location_name_list:['Jem shopping centre'] }]. (8) Q: My current location is inside Jem shopping centre, how many available taxi arounds my location, like within 2km? A: [{task: taxi_availability, id: 0, dep: [1], args: {location_gps_list: <resource>-1},task_specific:['2km']}, {task: map_mapping, id: 1, dep: [-1], args: {location_name_list:['Jem shopping centre'] }]. Now, please only output task analysis JSON response based on the specific question given. Only JSON format should be output. Strictly follow the format given in the references cases."
evaluation_list =[]
for i in range(len(loaded_list)):
    pattern = r'Q:\s*(.*?)\s+A:\s*(.*)'
    data=loaded_list[i]
    matches = re.findall(pattern, data)

    for match in matches:
        # index = match[0]
        question = match[0]
        answer = match[1]
        # print(f"Q: {question}\nA: {answer}\n")
        # print(f"Q: {question}")
        # print(text.format(instruction_note, question, answer))
        evaluation_list.append(instruction_note+'Q: '+question)

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)


count=0
prediction_label =[]
truth_label =[]
# for t in tqdm(range(len(loaded_list))):
for t in tqdm(range(80)):
    input_data = evaluation_list[t]+' Do not include " in your answer.'
    string_data = get_completion(input_data, 'gpt4', max_tokens=300)
    # print(string_data)
    # Define the regex pattern to match tasks
    pattern = r"{task:\s*(\w+),"

    # Find all matches of tasks in the string
    tasks1 = re.findall(pattern, string_data)

    # print(tasks)

    string_data = loaded_list[t]

    # Define the regex pattern to match tasks
    pattern = r"{task:\s*(\w+),"

    # Find all matches of tasks in the string
    tasks2 = re.findall(pattern, string_data)

    # print(tasks)
    if tasks1 == tasks2:
        count+=1
        prediction_label.append(tasks1)
        truth_label.append(tasks2)
    else:
        # print(t)
        # print(tasks1)
        prediction_label.append(tasks1)
        # print(tasks2)
        truth_label.append(tasks2)
print(count)

file_path = "GPT4o_prediction_label.json"
# save to JSON
with open(file_path, 'w') as file:
    json.dump(prediction_label, file)

file_path = "GPT4o_truth_label.json"

# save to JSON
with open(file_path, 'w') as file:
    json.dump(truth_label, file)



count=0
prediction_label =[]
truth_label =[]
# for t in tqdm(range(len(loaded_list))):
for t in tqdm(range(80)):
    input_data = evaluation_list[t]+' Do not include " in your answer.'
    string_data = get_completion(input_data, 'gpt3.5', max_tokens=300)
    # print(string_data)
    # Define the regex pattern to match tasks
    pattern = r"{task:\s*(\w+),"

    # Find all matches of tasks in the string
    tasks1 = re.findall(pattern, string_data)

    # print(tasks)

    string_data = loaded_list[t]

    # Define the regex pattern to match tasks
    pattern = r"{task:\s*(\w+),"

    # Find all matches of tasks in the string
    tasks2 = re.findall(pattern, string_data)

    # print(tasks)
    if tasks1 == tasks2:
        count+=1
        prediction_label.append(tasks1)
        truth_label.append(tasks2)
    else:
        # print(t)
        # print(tasks1)
        prediction_label.append(tasks1)
        # print(tasks2)
        truth_label.append(tasks2)
print(count)

file_path = "GPT3_5_prediction_label.json"
# save to JSON
with open(file_path, 'w') as file:
    json.dump(prediction_label, file)

file_path = "GPT3_5_truth_label.json"

# save to JSON
with open(file_path, 'w') as file:
    json.dump(truth_label, file)
