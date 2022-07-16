import pandas as pd
from transformers import pipeline
import argparse
import time
import plotly.express as px
import numpy as np

def get_fuzzy_category(data_frame, search_phrase, target_column = 'Title', threshold = 0.95, do_plot = True):
    class_labels = [search_phrase, 'other']
    data_frame[search_phrase] = data_frame[target_column].apply(lambda x: assign_fuzzy_category(x, class_labels, threshold)).astype(int)
    if(do_plot):
        plot_fuzzy_category(data_frame, search_phrase, target_column)
    return data_frame
    
def plot_fuzzy_category(data_frame, search_phrase, target_column):
    fig = px.scatter(
        data_frame, x='Column1', y='Column2', 
        color = search_phrase, hover_data = ['Authors', 'Title']
    )
    fig.write_html("../data/{}_{}.html".format(search_phrase, target_column))
    
def assign_fuzzy_category(test_string, class_labels, threshold):
    result = intent_classifier.classify(test_string.rstrip(), class_labels, len(class_labels) > 2)
    return make_decision(result, threshold)

def make_decision(result, threshold):
    if(result['labels'][0] == 'other' and result['scores'][1] < threshold):
        return False
    else:
        return True

class IntentClassifier():
    def __init__(self, model_name):
        self.dn = 0
        # list of feasible models: https://huggingface.co/models?search=nli
        self.model = pipeline("zero-shot-classification", model=model_name, tokenizer=model_name, device=self.dn)
    
    def classify(self, sequence, candidate_labels, multi_label=False, hypothesis_template="The intent of this statement is {}."):
        # play around with 'hypothesis_template' for better results
        result = self.model(sequence, candidate_labels, multi_label=multi_label, hypothesis_template=hypothesis_template)
        result["scores"] = [round(i, 4) for i in result["scores"]]
        return result

    
if __name__ == "__main__":
    df_embeddings_proj = pd.read_csv(r'../data/IVA_combined_embeddings_tsne.csv')
    df_embeddings_proj.set_index('Index')

    intent_classifier = IntentClassifier('facebook/bart-large-mnli')

    df_embeddings_proj["Title"].replace('', np.nan, inplace=True)
    df_embeddings_proj.dropna(subset=["Title"], inplace=True)

    df_embeddings_proj["Abstract"].replace('', np.nan, inplace=True)
    df_embeddings_proj.dropna(subset=["Abstract"], inplace=True)

    search_phrases = ['soldier', 'virtual human']
    target_columns = ['Title', 'Abstract']

    for search_phrase in search_phrases:
        for target_column in target_columns:
            temp_df = get_fuzzy_category(df_embeddings_proj, search_phrase, target_column)