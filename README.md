# iterPrompt
This is the source code for paper: iterPrompt: An Iterative Prompt-tuning Method for Nested
Relation Extraction, submitted to SIGIR 2023. 
## Requirements
1. python==3.6
2. h5py==3.1.0
3. matplotlib==3.3.4
4. nltk==3.6.7
5. pandas==1.1.5
6. prompt-toolkit==3.0.19
7. scipy==1.5.4
8. torch==torch-1.8.0
## Description
* config.py: model parameter configuration file
* data_ utils.py: data processing code file
* utils.py: indicator calculation code file
* models.py: various relationship extraction model code files
* tiny_ models.py: various basic sub-model code files
* fewshot.py: few-shot experiment code file
* prompt_ policy.py: prompt learning prompt model implementation and experimental code file

## Data sets
*  Data sets 5-1, 5-2, 5-3, 5-4, 5-5 are used for nested relation extraction

## Run
python prompt_policy.py
