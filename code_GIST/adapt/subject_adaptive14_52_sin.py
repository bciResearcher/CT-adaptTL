'''''
subject adaptive 
'''''

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from os.path import join as pjoin
from SHALLOW import deep
import time

# GIST data path
dfile = h5py.File('../process/DATA52/ku_mi_smt.h5', 'r')
# pretrained model path
outpath = '../pretrain/pretrain14_52'
device = torch.device('cuda')


def evaluate(model, x, y):
    pred = []
    label = []
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=100, shuffle=True)
    model.eval()
    num = 0
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            test_input, test_lab = d
            label.append(test_lab.detach().cpu().numpy())
            out = model(test_input)
            _, indices = torch.max(out, dim=1)
            pred.append(indices.detach().cpu().numpy())
            correct = torch.sum(indices == test_lab)
            num += correct.cpu().numpy()
    return num * 1.0 / x.shape[0], pred, label


def get_data(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)


# storage of prediction labels, true labels and classification accuracy
pred_52 = []
label_52 = []
acc=np.zeros([52])
# computation time for each subject
# time_52 = np.zeros([52])
for n in range(52):
    # start = time.time()
    X, Y = get_data(n+1)
    # print(X.shape, Y.shape)

    T_X_train, T_Y_train = X[:100], Y[:100]
    T_X_test, T_Y_test = X[100:], Y[100:]

    T_X_train = T_X_train.transpose([0, 2, 1])
    T_X_test = T_X_test.transpose([0, 2, 1])

    T_X_train = torch.tensor(np.expand_dims(T_X_train, axis=1), dtype=torch.float32)
    T_X_test = torch.tensor(np.expand_dims(T_X_test, axis=1), dtype=torch.float32)
    T_Y_train = torch.tensor(T_Y_train, dtype=torch.long)
    T_Y_test = torch.tensor(T_Y_test, dtype=torch.long)

    T_X_train, T_Y_train = T_X_train.to(device), T_Y_train.to(device)
    T_X_test, T_Y_test = T_X_test.to(device), T_Y_test.to(device)

    target_set = TensorDataset(T_X_train, T_Y_train)
    target_set2 = TensorDataset(T_X_test, T_Y_test)
    target_loader = DataLoader(dataset=target_set, batch_size=40, shuffle=True)
    target_loader2 = DataLoader(dataset=target_set2, batch_size=40, shuffle=True)

    model = deep()
    # load pretrained model
    checkpoint = torch.load(pjoin(outpath, 'model_cv{}.pt'.format(3)))   # select and use the optimal model
    model.load_state_dict(checkpoint)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.4)

    for epoch in tqdm(range(200)):
        model.train()
        t = 0
        for i, (train_fea, train_lab) in enumerate(target_loader):
            out = model(train_fea)
            _, pred = torch.max(out, dim=1)
            t += (pred == train_lab).sum().cpu().numpy()
            loss = F.nll_loss(out, train_lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print('\n Train, epoch: {}, i: {}, loss: {}'.format(epoch, i, loss))
        print('acc',t/100)
    acc[n], pred_, label_ = evaluate(model, T_X_test, T_Y_test)
    pred_52.append(pred_)
    label_52.append(label_)
    # save fine-tuned model for each subject
    path = pjoin('./model_52_', 'model_cv{}.pt'.format(n))
    torch.save(model.state_dict(),path)
    print(acc)
    # end = time.time()
    # times = end - start
    # time_52[n] = times
    # print('time computation',times)

# np.save('52acc_fin', acc)
# np.save('pred_52', pred_52)
# np.save('label_52', label_52)
# np.save('time_single_52',time_52)
print(np.mean(acc))
