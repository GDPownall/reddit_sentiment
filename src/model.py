#!/usr/bin/env python

from torch import nn
from transformers import AdamW, BertModel
import torch
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AITAClassifier(nn.Module):

    def __init__(self, data):
        super(AITAClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(data.pre_trained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.lin  = nn.Linear(self.bert.config.hidden_size, data.n_classes())
        self.out  = nn.Softmax()


    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask
                ).pooler_output
        dropped = self.drop(pooled_output)
        output  = self.lin(dropped)
        return self.out(output)


def train(model, data, n_epochs=3, batch_size = 32):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias = False)
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(data.one_hot_weights()).to(device))

    data.train_test_split()
    train = data.split_data[0]
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        print('Epoch number:',epoch)
        for b in range(0, len(train), batch_size):
            input_ids      = train[b:b+batch_size]['input_ids'].to(device)
            attention_mask = train[b:b+batch_size]['attention_mask'].to(device)
            targets        = torch.FloatTensor(train[b:b+batch_size]['targets']).to(device)

            outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                    )
            _, preds = torch.max(outputs, dim=1)
            _, targets_idx = torch.max(targets, dim=1)
            loss = loss_fn(outputs, targets_idx)

            epoch_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch number:',epoch,', loss:',epoch_loss)

def evaluate(model, data):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    val  = data.split_data[1]
    test = data.split_data[2]

    with torch.no_grad():
        for dataset in [val,test]:
            input_ids      = dataset[:]['input_ids'].to(device)
            attention_mask = dataset[:]['attention_mask'].to(device)
            targets        = torch.FloatTensor(dataset[:]['targets']).to(device)

            outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                    )

            _, preds = torch.max(outputs, dim=1)
            _, targets_idx = torch.max(targets, dim=1)

            loss = loss_fn(outputs, targets_idx)

            loss_val = loss.item()
            print('loss:',loss_val)
            print('Confusion matrix:')
            targets_idx = targets_idx.cpu()
            preds = preds.cpu()
            print(confusion_matrix(targets_idx,preds))
