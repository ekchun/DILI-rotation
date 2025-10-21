# Load model directly
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from utils import Classifier, SmilesDataset
from copy import deepcopy

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR", output_hidden_states=True)

# load dataset
df = pd.read_csv("dataset/CellCount_FINAL.csv")
sequences = df['smiles'].to_list()
split_df = pd.read_csv(
    "dataset/CellCount_FINAL_train_valid_test_scaffold_2c4e2e77-5019-4db8-954a-e55ac70d33a8.csv")
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

outputs = model(**tokens)
# print(outputs)
del tokenizer
del model

print(outputs.hidden_states[-1].size())

embeddings = outputs.hidden_states[-1][:, 0, :].detach() # only take the [CLS] token
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings.size())

feat_dict = {}
for idx, row in df.iterrows():
    feat_dict[row.spid] = embeddings[idx]

feat_size = embeddings.size(1)
classifier = Classifier(feat_size, 256, 1)

train_dataset = SmilesDataset(df, feat_dict, split_df, 'train')
valid_dataset = SmilesDataset(df, feat_dict, split_df, 'valid')
test_dataset = SmilesDataset(df, feat_dict, split_df, 'test')

trainloader = torch.utils.data.DataLoader(train_dataset, 16, shuffle=True, num_workers=0)
validloader = torch.utils.data.DataLoader(valid_dataset, 8, shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, 8, shuffle=False, num_workers=0)

print(len(trainloader), len(validloader), len(testloader))

classifier = classifier.cuda()
classifier.train()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.302, 4.314])).cuda()
optim = torch.optim.Adam(classifier.parameters(), lr=0.0001, weight_decay=1e-5)

epochs = 500
best_auc = 0
best_prec = 0
best_rec = 0
best_classifier = None

for epoch in range(epochs):
    total_train_loss = 0
    classifier.train()

    for idx, batch in enumerate(trainloader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        output = classifier(x).squeeze(-1)
        loss = criterion(output, y.float())
        
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optim.step()

        total_train_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {total_train_loss / len(trainloader)}")
    
    # Evaluation every 5 epochs
    if (epoch + 1) % 5 == 0:
        classifier.eval()
        num_corrects = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for idx, batch in enumerate(validloader):
                x, y = batch
                x = x.cuda()
                y = y.cuda()

                output = F.sigmoid(classifier(x))
                # predictions = output >= 0.5
                # output = torch.argmax(F.softmax(classifier(x)))
                
                # corrects = (output == y).sum().item()
                # num_corrects += corrects

                all_preds.extend(output.cpu().detach().tolist())
                all_labels.extend(y.cpu().detach().tolist())
        # accuracy = 100*(num_corrects / len(valid_dataset))
        # print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy}")

        all_preds = torch.tensor(all_preds).numpy()
        all_labels = torch.tensor(all_labels).numpy()

        # Convert predictions to binary class labels using a threshold (e.g., 0.5)
        predicted_classes = (all_preds >= 0.5)

        # Calculate AUC
        auc = roc_auc_score(all_labels, all_preds)
        print(f"AUC: {auc}")

        # Calculate Precision
        precision = precision_score(all_labels, predicted_classes)
        print(f"Precision: {precision}")

        # Calculate Recall
        recall = recall_score(all_labels, predicted_classes)
        print(f"Recall: {recall}")

        if auc >= best_auc:
            best_auc = auc
            best_prec = precision
            best_rec = recall
            best_classifier = deepcopy(classifier)

print(f"Validation\nAUC: {best_auc}")
print(f"Precision: {best_prec}")
print(f"Recall: {best_rec}")

best_classifier.eval()
num_corrects = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for idx, batch in enumerate(testloader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        # output = F.sigmoid(classifier(x)) >= 0.5
        # output = torch.argmax(F.softmax(best_classifier(x)))
        output = F.sigmoid(classifier(x))
        all_preds.extend(output.cpu().detach().tolist())
        all_labels.extend(y.cpu().detach().tolist())
        # corrects = (output == y).sum().item()
        # num_corrects += corrects
# accuracy = 100*(num_corrects / len(test_dataset))
# print(f"Inference - Accuracy: {accuracy}")

all_preds = torch.tensor(all_preds).numpy()
all_labels = torch.tensor(all_labels).numpy()

# Convert predictions to binary class labels using a threshold (e.g., 0.5)
predicted_classes = (all_preds >= 0.5)

# Calculate AUC
auc = roc_auc_score(all_labels, all_preds)
# Calculate Precision
precision = precision_score(all_labels, predicted_classes)
# Calculate Recall
recall = recall_score(all_labels, predicted_classes)

print(f"Inference\nAUC: {auc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")