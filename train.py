import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from tensorboardX import SummaryWriter
from glob import glob

from dataset import MyDataset
from model import RNN, LSTM, CNN, LSTM_with_Attention


def generate_experiment_name():
    dirs = glob('runs/*')
    indices = [int(val[-3:]) for val in dirs if val[-3:].isdigit()]
    if indices:
        last_idx = max(max(indices), 0)
    else:
        last_idx = 0

    return f'runs/experiments{last_idx+1:03d}'


writer = SummaryWriter(generate_experiment_name())
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Toy Project", add_help=False)
    parser.add_argument('--optim', type=str, choices=['sgd', 'adam'],
                        default='sgd', help="Choose optimizer")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--epoch', type=int, default=15,
                        help="Epoch")
    parser.add_argument('--cuda', action='store_true',
                        help="Use cuda")
    parser.add_argument('--ed', type=int, default=300,
                        help="Embedding dimensions")
    parser.add_argument('--word_vector', type=str2bool, nargs='?',
                        default=True, help="Use word vector like gloVe")
    parser.add_argument('--model', type=str, choices=['rnn', 'lstm', 'cnn','lstm_attn'],
                        required=True,
                        help="Choose model")

    parser_args, _ = parser.parse_known_args()

    target_parser = argparse.ArgumentParser(parents=[parser])
    if parser_args.model in 'cnn':
        target_parser.add_argument('--filter', type=int, default=100,
                                   help="Filter number")
        target_parser.add_argument('--filter_size', type=list, nargs='+',
                                   default=[3, 4, 5], help="Filter size")
        target_parser.add_argument('--dropout', type=str2bool, nargs='?',
                                   default=True, help="Use dropout")
    elif parser_args.model in 'rnn':
        target_parser.add_argument('--hd', type=int, default=512,
                                   help="Hidden dimensions")
    elif parser_args.model in ['lstm', 'lstm_attn']:
        target_parser.add_argument('--hd', type=int, default=512,
                                   help="Hidden dimensions")
        target_parser.add_argument('--layer', type=int, default=2,
                                   help="Layer number")
        target_parser.add_argument('--bidirectional', type=str2bool, nargs='?',
                                   default=True, help="use bidirectional")
        target_parser.add_argument('--dropout', type=str2bool, nargs='?',
                                   default=True, help="Use dropout")

    args = target_parser.parse_args()
    writer.add_text('Hyperparameters', str(args))

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
    dataset = MyDataset(batch_size=args.batch_size, use_vector=args.word_vector)

    # Hyper parameters
    INPUT_DIM = len(dataset.TEXT.vocab)
    OUTPUT_DIM = 1

    # TODO: SVM

    if args.model == 'rnn':
        print("Model: Vanila RNN")
        model = RNN(INPUT_DIM, args.ed, args.hd, OUTPUT_DIM).to(device)
    elif args.model == 'lstm':
        print("Model: LSTM")
        model = LSTM(
            INPUT_DIM, args.ed, args.hd, OUTPUT_DIM,
            n_layers=args.layer, use_bidirectional=args.bidirectional,
            use_dropout=args.dropout).to(device)
    elif args.model == 'lstm_attn':
        print("Model: LSTM with Attension")
        model = LSTM_with_Attention(
            INPUT_DIM, args.ed, args.hd, OUTPUT_DIM,
            n_layers=args.layer, use_bidirectional=args.bidirectional,
            use_dropout=args.dropout).to(device)
    elif args.model == 'cnn':
        print("Model: CNN")
        model = CNN(
            INPUT_DIM, args.ed, args.filter, args.filter_size, OUTPUT_DIM,
            args.dropout).to(device)

    if args.word_vector:
        model.embedding.weight.data.copy_(dataset.TEXT.vocab.vectors)

    if args.optim == 'sgd':
        print("Optim: SGD")
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        print("Optim: Adam")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(size_average=True).to(device)

    best_acc = 0
    for epoch in range(args.epoch):
        train_loss, train_acc = train(model, dataset.dataloader['train'],
                                      optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, dataset.dataloader['dev'],
                                         criterion)
        test_loss, test_acc = evaluate(model, dataset.dataloader['test'],
                                       criterion)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc * 100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc * 100:.2f}%, Test Loss: {test_loss:.3f}, Test Acc: {test_acc * 100:.2f}%')

        writer.add_scalars('data/loss', {
            'train': train_loss,
            'val': valid_loss,
            'test': test_loss,
        }, epoch + 1)
        writer.add_scalars('data/acc', {
            'train': train_acc,
            'val': valid_acc,
            'test': test_acc,
        }, epoch + 1)

        if best_acc <= valid_acc:
            best_acc = valid_acc
            acc_result = test_acc
            pth = model.state_dict()
            if args.model in 'cnn':
                filename = "checkpoints/{}_{}_bs{}_filter{}_acc{:.03f}.pth".format(
                    args.model, args.optim, args.batch_size, 100, test_acc
                )
            elif args.model in ['lstm', 'rnn', 'lstm_attn']:
                filename = "checkpoints/{}_{}_bs{}_hd{}_acc{:.03f}.pth".format(
                    args.model, args.optim, args.batch_size, args.hd, test_acc
                )
    writer.add_text('Test acc', str(acc_result))
    torch.save(pth, filename)


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
    print(args)
    main(args)
