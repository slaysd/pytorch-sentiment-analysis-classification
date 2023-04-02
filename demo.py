import torch
import torch.nn.functional as F
import spacy
import argparse
import re

from dataset import MyDataset
nlp = spacy.load('en_core_web_sm')

def get_models(ckpt_path, device, dataset):
    from train import build_model
    info = torch.load(ckpt_path)
    args = info['args']
    pth = info['pth']

    model = build_model(args, device, dataset)
    model.load_state_dict(pth)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('input', type=str)
    args = parser.parse_args()


    device = torch.device("cpu")
    dataset = MyDataset(batch_size=1, use_vector=True)
    model = get_models(args.ckpt_path, device, dataset)

    tokenized = [tok.text for tok in nlp.tokenizer(args.input)]
    indexed = [dataset.TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    tensor = torch.cat(2*[tensor], dim=1)
    preds = model(tensor)
    max_preds = preds.argmax(dim=1)
    print(max_preds[0].item())
