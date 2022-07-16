from transformers import pipeline
import argparse
import time

class IntentClassifier():
    def __init__(self, model_name):
        self.dn = 0
        # list of feasible models: https://huggingface.co/models?search=nli
        self.model = pipeline("zero-shot-classification", model=model_name, tokenizer=model_name, device=self.dn)
    
    def classify(self, sequence, candidate_labels, multi_class=False, hypothesis_template="The intent of this statement is {}."):
        # play around with 'hypothesis_template' for better results
        result = self.model(sequence, candidate_labels, multi_class=multi_class, hypothesis_template=hypothesis_template)
        result["scores"] = [round(i, 4) for i in result["scores"]]
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='facebook/bart-large-mnli', type=str, help='Choise of NLI model')
    parser.add_argument('--sequence', type=str, help='Input sequence', required=True)
    parser.add_argument('--candidate_labels', nargs='+', help='Input labels', required=True)
    args = parser.parse_args()
    
    ic = IntentClassifier(args.model_name)
    result = ic.classify(args.sequence, args.candidate_labels, len(args.candidate_labels) > 2)
    print(result)



    

