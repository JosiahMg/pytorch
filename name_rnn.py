import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import gzip
import csv
import time
import matplotlib.pyplot as plt
import math

HIDDEN_SIZE = 100
BATCH_SIZE = 64
N_LAYER = 2
N_EPOCHS = 20
N_CHARS = 128
USE_GPU = True


torch.cuda.set_device(0)

def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

# 统计运行时间
def time_since(since):
    s = time.time() - since
    m = math.floor(s/60)
    s -= m*60
    return "%dm %ds" % (m, s)


def plot_acc_list(acc_list):
    epoch = np.arange(1, len(acc_list)+1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

"""
X:  name
    type: str
y: country
    number
"""
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'data/names_train.csv.gz' if is_train_set else 'data/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, item):
        # 返回名字为字符串(str)， 国家为索引值(int)
        return self.names[item], self.country_dict[self.countries[item]]

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = dict()
        # 遍历国家列表，从0开始制作字典
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num


trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = trainset.getCountriesNum()

def name2list(name):
    # 将名字转换成ASSIC的列表
    arr = [ord(c) for c in name]
    return arr, len(arr)



def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
    countries = countries.long()

    # make padding, append 0 to name_sequences
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]
    # seq_tensor.shape:[batchSize, seqLen]
    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        # The input of Embedding Layer with shape:(seqLen, batchSize)
        # The output of Embedding layer with shape:(seqLen, batchSize, hiddenSize)
        self.embedding = torch.nn.Embedding(input_size, hidden_size)

        """
        The inputs of GRU layer with shape:
            input:(seqLen, batchSize, hiddenSize)
            output:(nLayers*nDirections, batchSize, hiddenSize)
        The outpus of GRU layer with shape:
            output:(seqLen, batchSize, hiddenSize*nDirections)
            hidden:(nLayers*nDirections, batchSize, hiddenSize)
        """
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size*self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        # input shape: (batch, seqLen)->(seqLen, batch)
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        # input.shape=(seqLen, batchSize) ->
        # embedding.shape=(seqLen, batchSize, hiddenSize)
        embedding = self.embedding(input)

        # 对每个sequence中有效的字符从多到小进行排序
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output


def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader):
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{(i+1)*len(inputs)}/{len(trainset)}]', end='')
            print(f'loss={total_loss/((i+1)*len(inputs))}')

    return total_loss


def testModel():
    correct = 0
    total = len(testset)
    print('evaluating trained model ...')
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100*correct/total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct/total


if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS+1):
        trainModel()
        acc = testModel()
        acc_list.append(acc)

    print('Training time ', time_since(start))

    plot_acc_list(acc_list)




