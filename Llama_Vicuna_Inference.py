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

model='llama2'
if model == 'llama2':
### Llama 2
    model_name = "meta-llama/Llama-2-7b-chat-hf"
### Llama 3
elif model == 'llma3':
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
### Vicuna
elif model == 'Vicuna':
    model_name = "lmsys/vicuna-7b-v1.5-16k"
dataset_name = "mlabonne/guanaco-llama2-1k"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 5
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}



dataset = dataset1
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)


instruction_note = "You are UAPLLM, a Large language model for urban activity that decompose user input into a list of spatio-temporal tasks with the following JSON format: {task: task_name, id: task_id, dep: dependency_task_ids, args: {domain: string, location_name_list: list, location_gps_list: list, time: time, input: sequence, service_no: int, bus_stop: string, task_specific: list}}. Task-specific information is stored using a string list and is used in specific models or tools as arguments. The dep field denotes the ID of the previous task that the current task relies. A dep field value of '-1' indicates that the current task does not rely on other tasks and can be executed immediately, otherwise always execute the previous task first. The executed output, such as the generated location, time, sequence, or POIs from the dependency task is marked as <resource>-task_id. This resource will be used in the next task. The spatio-temporal tasks must be selected from the following options: {long_time_series_prediction, time_series_prediction, event_prediction, trajectory_completion, trajectory_prediction, time_series_anomaly_detection, time_series_imputation, arrival_time_estimation, trajectory_forecasting, map_mapping, recommendation, spatial_relationship_inference, bus_arrival, taxi_availability}. To better understand each spatial-temporal task, here is the explanation and numbering, along with the corresponding examples: 1) Long Time Series Prediction: This task in- volves forecasting future values in a time series over a long horizon. It is typically used for long- term planning and trend analysis in various do- mains, such as weather forecasting, economic fore- casting, and demand planning. 2) Time Series Prediction: This task focuses on predicting future values in a time series over a shorter horizon compared to long time series predic- tion. It is commonly used for short-term forecasts like daily stock prices, temperature forecasts, or short-term sales predictions. 3) Event Prediction: This task involves predict- ing the occurrence of specific events based on his- torical data. Examples include predicting natural disasters, equipment failures, or social events like concerts or sports games. 4)Trajectory Completion: This task involves completing missing parts of a trajectory based on observed segments. It is useful in applications like tracking moving objects, filling in missing GPS data, or reconstructing incomplete travel routes. 5)Trajectory Prediction: This task involves fore- casting the future path of a moving object based on its past trajectory. Applications include predicting the movement of vehicles, pedestrians, or animals. 6)Time Series Anomaly Detection: This task in- volves identifying unusual patterns or outliers in time series data that deviate from expected behav- ior. It is used in applications like fraud detection, fault detection in machinery, and monitoring traffic conditions. 7) Time Series Imputation: This task involves filling in missing values in time series data to en- sure completeness and consistency. It is crucial for maintaining data quality in various applications like traffic records and climate data. 8)Arrival Time Estimation: This task involves predicting the arrival time of a vehicle or person at a specific location based on current and historical data. It is commonly used in transportation systems for buses, trains, and delivery services. 9) Taxi Availability Prediction: This task in- volves predicting the availability of taxis in spe- cific areas at given times. It helps optimize taxi dispatching and improve service for passengers by anticipating demand and ensuring timely availabil- ity. 10) Map Mapping: This task involves mapping addresses to GPS locations and mapping GPS loca- tionsback to addresses . 11) Bus Arrival: This task involves predicting the arrival times of buses at specific stops based on real-time data and historical patterns. It enhances the efficiency of public transportation systems by providing accurate and timely information to com- muters. 12) Spatial Relationship Inference: This task involves deducing spatial relationships between different entities or locations. It is used in urban planning to understand spatial dependencies and interactions, such as proximity analysis, clustering, and spatial correlations. 13) Recommendation: This task involves sug- gesting items or actions to users based on their preferences and historical behavior. Applications include recommending points of interest, routes, or services in urban planning. Please note that there exists a logical connection and order between the tasks. 1) If user input mentioned some specific locations/POIs, usually map mapping task should be included in the answer, otherwise if user is only asking the situation around all Singapore, map mapping task should not be included. 2) If user do not specify the arrival time, usually estimated arrival time task should be included. 3) Include tasks to predict weather and PM2.5 according to user input if user is going to outdoor activities. 4) When recommendation or taxi_availability task is included, usually map mapping task should be included as fundamental task. In case the user input cannot be parsed, an empty JSON response should be provided Please provide a task analysis JSON response basd on the given question."
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
for t in tqdm(range(len(loaded_list))):
    prompt = evaluation_list[t]+' Do not include " in your answer.'
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
    result = pipe(f"<s>[INST] {prompt} [/INST]")

    string_data = result[0]['generated_text']

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




file_path = "{}_prediction_label.json".format(model)
# save to JSON
with open(file_path, 'w') as file:
    json.dump(prediction_label, file)

file_path = "{}_truth_label.json".format(model)

# save to JSON
with open(file_path, 'w') as file:
    json.dump(truth_label, file)

# # # load from JSON
# with open(file_path, 'r') as file:
#     loaded_list = json.load(file)

