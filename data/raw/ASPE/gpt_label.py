import os
import openai
import random
import pandas as pd
from tqdm import tqdm
from time import sleep
from converter import read_line_examples_from_dataset_file

template = '''I will provide you with two cases related to the aspect-based sentiment analysis task. Place your input and answer between the symbols "####". Please analyze whether these two cases are similar in terms of domain, sentiment, syntax, and lexical choice, considering the following criteria:

- Domain Relevance: Do both sentences likely belong to the same domain?
- Sentiment Similarity: Do the sentences express similar sentiments? Note that a sentence might contain multiple aspect-sentiment pairs (positive, neutral, negative). If multiple sentiments are present, are their quantities and combinations roughly equivalent? (Please compare the two answers.)
- Syntactic Similarity: Are the grammatical structures of the two sentences similar?
- Lexical Similarity: Do the sentences employ similar words, phrases, or constructions?

Sentence 1: “{case1}”
Sentence 2: “{case2}”

Based on the above criteria, please analyze and provide reasons. Output a 0 or 1 for each criterion (0 indicates dissimilarity, 1 indicates similarity). The output *format should be as follows*:
- Domain Relevance: 0 or 1
- Sentiment Similarity: 0 or 1
- Syntactic Similarity: 0 or 1
- Lexical Similarity: 0 or 1
'''


openai.api_key = "sk-dPLhPmRUfMhXGux956F4E2A1832f475fA981A91dA087B221"
openai.api_base = "https://one-api.glm.ai/v1"

def get_response(prompt):
    max_try=10000000
    try_times=0
    while True:
        try_times+=1
        if try_times<=max_try:
            try:
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                # model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant"},
                    {"role": "user", "content": prompt},
                ]
                )
                if hasattr(response["choices"][0]["message"], "content"):
                    return response['choices'][0]['message']['content']
            #打印错误
            except Exception as e:
                sleep(60)
                print(e)
        else:
            return None

# TODO 随机采样不均衡，改为用 sbert 采样 top-

train_file = 'train_merge.txt'
csv_file = 'pairs_gptlabeled.tsv'
with open(train_file, 'r') as f:
    lines = f.readlines()

    df = pd.read_csv(csv_file, sep='\t')
    df = df.values.tolist()
    for line in tqdm(random.sample(lines, 1000)):
        # 针对每个句子随机采样 10 个样本
        sample_lines = random.sample(lines, 10)
        
        if line in sample_lines:
            sample_lines.remove(line)
        
        for sline in sample_lines:
            prompt = template.format(case1=line, case2=sline)
            response = get_response(prompt)
            print("\n==============")
            print(prompt)
            print("=====")
            print(response)
            print("=============\n")
            df.append([line, sline, response])


        save_df = pd.DataFrame(df, columns=['case1', 'case2', 'response'])
        save_df.to_csv(csv_file, index=False, sep='\t')