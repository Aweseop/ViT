import torch
import torch.nn as nn
import torch.optim as optim
from model import VIT
from dataset import load_MNIST
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

def train(args, model, train_set, val_set) :
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs) :
        print('*** Epoch {} ***'.format(epoch))

        # Training
        model.train()  
        running_loss, running_acc = 0.0, 0.0
        for idx, (inputs, labels) in enumerate(train_set) :
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True) :
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.shape[0]
                running_acc += torch.sum(preds == labels.data)
        running_acc /= (idx+1) * args.batch_size
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', running_loss, running_acc))
            
        # Validation
        model.eval()  
        running_acc = 0.0
            
        for idx, (inputs, labels) in enumerate(val_set) :
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False) :
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # statistics
                running_acc += torch.sum(preds == labels.data)
        running_acc /= (idx+1) * args.batch_size
        print('{} Acc: {:.4f}\n'.format('valid', running_acc))


def test(args, model, test_set) :
    model.eval()  
    running_acc = 0.0

    for idx, (inputs, labels) in enumerate(test_set) :
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False) :
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_acc += torch.sum(preds == labels.data)
    running_acc /= (idx+1) * args.batch_size
    print('{} Acc: {:.4f}\n'.format('test', running_acc))


def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of training iterations')
    parser.add_argument('--in_channel', type=int, default=1, help='number of channels')
    parser.add_argument('--img_size', type=int, default=28, help='input image size')
    parser.add_argument('--patch_size', type=int, default=4, help='patch size')
    parser.add_argument('--emb_dim', type=int, default=4*4, help='Encoder embedding dimension')
    parser.add_argument('--n_layers', type=float, default=6, help='number of encoder layers')
    parser.add_argument('--num_heads', type=int, default=2, help='number of multi-head attention heads')
    parser.add_argument('--forward_dim', type=int, default=4, help='MLP block embedding dimension')
    parser.add_argument('--dropout_ratio', type=int, default=0.1, help='dropout ratio')
    parser.add_argument('--n_classes', type=int, default=10, help='number of data classes')

    args = parser.parse_args()

    model = VIT(args.in_channel, args.img_size, args.patch_size, args.emb_dim, args.n_layers, args.num_heads, args.forward_dim, args.dropout_ratio, args.n_classes).to(device)
    train_set, val_set, test_set = load_MNIST(args.batch_size, n_fold=5)

    train(args, model, train_set, val_set)
    test(args, model, test_set)


if __name__ == "__main__" :
    main()
