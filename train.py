import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from dataset import MyDataset
from model import RNN


def parse_args():
    parser = argparse.ArgumentParser(description="Toy Project")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--cuda', type=bool, default=True,
                        help="Use cuda")
    parser.add_argument('--ed', type=int, default=100,
                        help="Embedding dimensions")
    parser.add_argument('--hd', type=int, default=256,
                        help="Hidden dimensions")

    args = parser.parse_args()

    return args


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    # convert into float for division
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    dataset = MyDataset()

    # Hyper parameters
    INPUT_DIM = len(dataset.TEXT.vocab)
    EMBEDDING_DIM = args.ed
    HIDDEN_DIM = args.hd
    OUTPUT_DIM = 1

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(10):
        train_loss, train_acc = train(model, dataset.dataloader['train'],
                                      optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, dataset.dataloader['dev'],
                                         criterion)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc * 100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc * 100:.2f}%')
    torch.save(model.state_dict(), "./checkpoints/final.pth")


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        batch.text = batch.text.permute(1, 0)
        pred = model(batch.text).squeeze(1)
        loss = criterion(pred, batch.label)
        acc = binary_accuracy(pred, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    for batch in iterator:
        batch.text = batch.text.permute(1, 0)
        pred = model(batch.text).squeeze(1)
        loss = criterion(pred, batch.label)
        acc = binary_accuracy(pred, batch.label)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == '__main__':
    args = parse_args()
    main(args)
