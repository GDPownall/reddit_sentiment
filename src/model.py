#!/usr/bin/env python

from torch import nn
from transformers import AdamW, BertModel

class AITAClassifier(nn.Module):

    def __init__(self, data):
        super(AITAClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(data.pre_trained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.lin  = nn.Linear(self.bert.config.hidden_size, data.n_classes())
        self.out  = nn.Softmax()


    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask
                )

        dropped = self.drop(pooled_output)
        output  = self.lin(dropped)
        return self.out(output)


def train(model, data, n_epochs=3, batch_size = 32):
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias = False)
    loss_fn = nn.CrossEntropyLoss()

    data.train_test_split()
    train = data.split_data[0]
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        print('Epoch number:',epoch)
        for b in range(0, len(train), batch_size):
            input_ids      = train[b:b+batch_size]['input_ids']
            attention_mask = train[b:b+batch_size]['attention_mask']
            targets        = train[b:b+batch_size]['targets']

            print(input_ids.shape)
            outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                    )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

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
            input_ids      = dataset[:]['input_ids']
            attention_mask = dataset[:]['attention_mask']
            targets        = dataset[:]['targets']

            outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                    )

            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            loss_val = loss.item()
            print('loss:',loss_val)
