from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, Dataset
import torch
import pandas as pd
import re
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score
import os
import ast
from langchain import PromptTemplate, HuggingFacePipeline
import torch.nn as nn
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
import streamlit as st
import tempfile

# Define the BitsAndBytesConfig for quantization
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",  
)


# Load model
def load_model(model_name="meta-llama/Llama-2-7b-chat-hf",cache_dir=""):
   tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,cache_dir=cache_dir)
   model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,cache_dir=cache_dir)
   print("Model Loaded Successfully")
   return model, tokenizer

def load_gpt_model(model_name="gpt2-large"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("GPT2 Model Loaded Successfully!")
    return model, tokenizer

def load_model_without_cache(model_name="meta-llama/Llama-2-7b-chat-hf"):
   tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
   model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)
   print("Model Loaded Successfully")
   return model, tokenizer

# Load quantized model
def load_quantized_model(model_name="meta-llama/Llama-2-70b-chat-hf",cache_dir="/scratch/miislam/cache"):
   tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,cache_dir=cache_dir)
   model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,cache_dir=cache_dir,quantization_config=nf4_config)
   print("Model Loaded Successfully")
   return model, tokenizer

# Load data
def load_data(data_path="dataset/examples.csv"):
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df)
    print(dataset)
    print("Dataset Loaded Successfully")
    return dataset

# Load data with encoding
def load_data_with_encoding(data_path="dataset/examples.csv"):
    df = pd.read_csv(data_path,encoding="ISO-8859-1")
    dataset = Dataset.from_pandas(df)
    print(dataset)
    print("Dataset Loaded Successfully")
    return dataset
    
# Load gpu   
def load_gpu():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loaded to GPU")
    return device

def get_llama3_3_70b_reponse(prompt, max_new_tokens=2000):
    device = load_gpu()
    model, tokenizer = load_quantized_model("meta-llama/Llama-3.3-70B-Instruct", cache_dir="/scratch/miislam/cache")
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device) 
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def extract_list_from_response(response):
    match = re.search(r"\[(.*?)\]", response)
    if match:
        try:
            return ast.literal_eval(f"[{match.group(1)}]")
        except (SyntaxError, ValueError):
            return None
    return None

def extract_python_list_from_prompt(response):
    match = re.search(r"\[\s*['\"].*?['\"]\s*\]", response, re.DOTALL)
    
    if match:
        try:
            return ast.literal_eval(match.group(0).strip())  
        except (SyntaxError, ValueError):
            return None  
    return None

def extract_last_python_list(response):
    matches = re.findall(r"\[[^\]]+\]", response, re.DOTALL)
    
    if matches:
        try:
            return ast.literal_eval(matches[-1].strip()) 
        except (SyntaxError, ValueError):
            return None  
    return None

def extract_python_list(llm_output):
    match = re.findall(r"\[.*?\]", llm_output)
    
    if match:
        try:
            return ast.literal_eval(match[-1]) 
        except (SyntaxError, ValueError):
            print(f"Error: Unable to parse extracted list from output: {match[-1]}")
            return [] 
    return []  
    
def clean_list(list):
    return list.strip("[]").replace('"', '').replace(",",";").replace("`","")

def to_lowercase(lst):
    return [item.lower() for item in lst]

def process_list(lst_str):
    lst_str = lst_str.strip("[]")  
    
    return ";".join([item.strip().lower() for item in lst_str.split(";")])

def extract_last_code(predicted_output):
    last_code_index = predicted_output.rfind("code:")
    if last_code_index == -1:
        return "No code found"
    return predicted_output[last_code_index + len("code:"):].strip()

def exact_match(gt, pred):
    return gt.strip() == pred.strip()

def bleu_score(gt, pred):
    return sentence_bleu([gt.split()], pred.split())

def edit_distance(gt, pred):
    return Levenshtein.distance(gt, pred)

def bert_score(gt, pred, model_type="bert-base-uncased"):
    if isinstance(gt, str):
        gt = [gt]
    if isinstance(pred, str):
        pred = [pred]

    P, R, F1 = score(pred, gt, lang="en", model_type=model_type)

    return {
        F1.mean().item()
    }
def compute_rouge(gt, pred):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gt, pred)
    return {
        scores['rougeL'].fmeasure
    }

def check_cosine_similarity(df, pred, gt, output_file):
    # df = df.dropna(subset=[pred, gt]).reset_index(drop=True)
    pred = df[pred].astype(str).tolist()
    gt = df[gt].astype(str).tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(pred + gt)
    generated_vectors = tfidf_matrix[:len(pred)]
    ground_truth_vectors = tfidf_matrix[len(pred):]
    cosine_similarities = [
        cosine_similarity(gv, gt)[0][0] for gv, gt in zip(generated_vectors, ground_truth_vectors)
    ]
    df["dist"] = cosine_similarities
    df.to_csv(output_file, index=False)
    print(f"Cosine similarity calculated and saved to {output_file}!")

# Import Json file

def load_json_file(path):
    with open(path,'r') as file:
        data = json.load(file)
    return data

def find_fields(d, key_target, results=[]):
    if isinstance(d, dict):
        for key, value in d.items():
            if key == key_target:
                results.append(value)
            elif isinstance(value, (dict, list)):
                find_fields(value,key_target, results)
    elif isinstance(d, list):
        for item in d:
            find_fields(item, key_target, results)
    return results

def copy_column(source_file, target_file, column_name, new_column_name, output_file):
    source_df = pd.read_csv(source_file)
    print(f" Columns: {source_df.columns}")
    target_df = pd.read_csv(target_file)

    if column_name not in source_df.columns:
        print(f"Column '{column_name}' not found in {source_file}")
        return

    target_df[new_column_name] = source_df[column_name]
    target_df.to_csv(output_file, index=False)
    print(f"Column '{column_name}' copied successfully to {output_file}")
    return target_df



def print_keys_values(data):
    for key, value in data.items():
        print(f"Key: {key}, Value: {value}")

def extract_value(data, outer_key, inner_key):
    inner_data = data.get(outer_key, {}).get(inner_key, {})
    print(f"Inner data: {inner_data}")
    
    return inner_data

def cleaning_data(data_path="dataset/sboms/bomps.csv"):
    df = pd.read_csv(data_path)
    if 'survey_response' not in df.columns:
       raise ValueError("Column 'survey_response' not found in dataset.")

    df['survey_response'] = df['survey_response'].astype(str).str.replace("\n", "").str.replace("-", "").str.replace('"','').str.replace("'","")
    df['survey_response'] = df['survey_response'].str.replace(r'\d+', '', regex=True)
    return df

def saving_to_csv(df,output_file):
    return df.to_csv(output_file,index=False)


def extract_decision(llm_response):
    cleaned_response = llm_response.strip().replace("```", "").strip().replace("\n","")
    matches = re.findall(r"answer:\s*(yes|no)", cleaned_response, re.IGNORECASE)
    return matches[1].lower() if len(matches)>=2 else "no"

def extract_and_save_values(data_path, outer_key, fields, output_csv):
    df = load_json_file(data_path)
    data = pd.DataFrame(columns=["code", "definition"])

    for inner_key in fields:
        extracted_values = extract_value(df, outer_key, inner_key)
        
        if isinstance(extracted_values, dict):
            for key, value in extracted_values.items():
                data.loc[len(data)] = [key, value]

    data.to_csv(output_csv, index=False)
    print(f"CSV file saved at {output_csv}!")

def check_match(text):
    match = re.search(r'"answer":\s*"(\w+)"', text)
    if match:
        answer = match.group(1)
    else:
        answer = "no"
    return answer
 # The following presents a qualitative coding of answers from a video game research study. The answers explain why a participant experienced a game as art. The codes summarize the given reasons as compactly as possible, preferably with just one or a few words. If an answer lists multiple reasons, the corresponding codes are separated by semicolons.	