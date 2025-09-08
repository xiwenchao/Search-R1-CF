
import os
import torch
import re
import json
from sentence_transformers import SentenceTransformer



def extract_title(solution_str):
    pattern = r'(?s)<answer>(.*?)</answer>'
    match = re.search(pattern, solution_str)
    return match.group(1) if match else None

def read_json(json_file:str) -> dict:
    f = open(json_file, 'r')
    return json.load(f)

def similarity_match(solution_str, ground_truth):
    title = extract_title(solution_str)

    if title:
        match = re.search(r'"([^"]*)', title)
        if match:
            text = match.group(1)
        else:
            text = solution_str.split('\n', 1)[0]
        
        # Identify your sentence-embedding model
        model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
        embeddings = torch.load(f"./data/amazon_data/CDs_and_Vinyl/embeddings.pt")
        name2id = read_json(f"./data/amazon_data/CDs_and_Vinyl/name2id.json")
        embeddings = torch.tensor(embeddings).cuda()
        
        predict_embedding = torch.tensor(model.encode(text))
        dist = torch.cdist(predict_embedding, embeddings, p=2)
        rank = dist.argsort()

        target_name = ground_truth.strip().strip('"')
        if target_name in name2id:
            target_id = name2id[target_name]
        else:
            target_id = 0

        rankId = rank[target_id]
        if rankId == 1:
            match = 1.0
        elif rankId <= 5:
            match = 0.8
        elif rankId <= 10:
            match = 0.5
        else:
            match = 0.0
    else:
        match = 0.0
    return match


def compute_score_ranking(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    match_score = similarity_match(solution_str, ground_truth)
    return match_score
