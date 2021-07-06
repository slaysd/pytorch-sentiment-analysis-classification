import torch
import torch.nn.functional as F
import spacy
import argparse

from dataset import MyDataset
nlp = spacy.load('en_core_web_sm')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sentence', type=str)
    parser.add_argument('--device', type=str, default="cpu")
    args = parser.parse_args()

    #TODO: add model demo

    dataset = MyDataset(batch_size=1, use_vector=True)
    tokenized = [tok.text for tok in nlp.tokenizer(args.sentence)]
    indexed = [dataset.TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(torch.device(args.device))
    tensor = tensor.unsqueeze(1)
    prediction = F.sigmoid(model(tensor))
    
    print(prediction.item())
