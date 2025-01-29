# UrbanLLM

 This is the official release of Urban: UrbanLLM: Autonomous Urban Activity Planning and Management with Large Language Models
# Requirements
```bash
pip install -r requirements.txt
```
## Data

Dataset for training, inference, and evaluation is released.

## Train Model
python3 run UrbanLearning.py with training_sample.txt dataset.

## Inference Results
python3 run UrbanLLM_Inference.py GPT_Inference.py Llama_Vicuna_Inference.py
Use evaluation_sample.txt as inference dataset, you will obtain two JSON files for evaluation.

## Evaluate Model
python3 run github/GPT4API/Model_evaluations.py
Use the obtained two JSON files from Inference.py code for evaluation. Or you can directly use llama2_truth_label.json and llama2_prediction_label.json to see model performance
