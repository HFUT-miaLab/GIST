import os
import time
import torch
import argparse
import torch.utils.data as data
import numpy as np
from sklearn.metrics import roc_auc_score
from TransMIL.models.TransMIL import TransMIL
from loader import TrainBagLoader
from TransMIL.MyOptimizer.radam import RAdam
from TransMIL.MyOptimizer.lookahead import Lookahead
from utils.utils import compute_TP_FP_TN_FN, compute_sensitivity, compute_specificity

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 RNN aggregator training script')
parser.add_argument('--data_path', type=str, default=r'',
                    help='features path')
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size (default: 1)')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--save_path', type=str, default='checkpoint',
                    help='path to trained model checkpoint')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--reg', type=str, default=1e-5, help='weight_decay')
parser.add_argument('--n_classes', type=int, default=5, help='number of classes')


def main(args):
    init_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dst = TrainBagLoader(args.data_path, 'data/gist_train.csv', args.n_classes)
    train_loader = data.DataLoader(dataset=train_dst, batch_size=args.batch_size,
                                   sampler=torch.utils.data.sampler.WeightedRandomSampler(
                                       train_dst.get_weights(), len(train_dst), replacement=True
                                   ),num_workers=args.workers,pin_memory=True)
    model = TransMIL(n_classes=args.n_classes)
    model = model.to(device)

    optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    optimizer = Lookahead(optimizer)
    loss_function = torch.nn.CrossEntropyLoss()

    epochs = args.nepochs
    start_time = time.time()

    for epoch in range(epochs):
        train_single(model, train_loader, optimizer, loss_function, device, epoch)

        state = {'net': model.state_dict(), 'optimizer': optimizer, 'epoch': epoch + 1}
        torch.save(state, os.path.join(args.save_path, 'model{:d}.pth'.format(epoch + 1)))
    end_time = time.time()
    print("TransMIL cost time: ", end_time - start_time, "s")


def train_single(model, train_loader, optimizer, loss_function, device, epoch):
    running_loss = 0
    model.train()
    for index, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        result_dict = model(data=features)
        loss = loss_function(result_dict['logits'], labels[0])
        loss.backward()
        optimizer.step()
        print("\rEpoch:{:d} train slide batch {:d}:{:d} loss={:.3f}".format(epoch + 1, index + 1, len(train_loader),
                                                                            loss.item()), end="")


def init_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
