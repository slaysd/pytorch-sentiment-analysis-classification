import torch
import spacy
import argparse

from dataset import MyDataset
nlp = spacy.load('en')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sentence', type=str, required=True)

    dataset = MyDataset(batch_size=1, use_vector=True)
    tokenized = [tok.text for tok in nlp.tokenizer(args.sentence)]
    indexed = [dataset.TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = F.sigmoid(model(tensor))
    return prediction.item()
