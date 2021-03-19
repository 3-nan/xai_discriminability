import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torchnet.meter import mAPMeter
# from pytorch_lightning.metrics.classification import AveragePrecision
from sklearn.metrics import average_precision_score

from .dataloading.custom import get_dataset
from .dataloading.dataloader import DataLoader


def compute_ap_score(y_predicted, y_true):

    scores = 0.0

    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true = y_true[i], y_score=y_predicted[i])

    return scores


def train_model(model, trainloader, testloader, criterion, optimizer, device, num_epochs=25):

    print("Device in usage is {}".format(device))

    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = trainloader
            else:
                model.eval()
                dataloader = testloader

            running_loss = 0.0
            # running_ap = 0.0

            mAP = mAPMeter()
            mAP.reset()
            # ap = AveragePrecision()
            i = 0

            for inputs, labels in dataloader:
                # inputs = torch.FloatTensor([sample.image for sample in data])
                inputs = torch.FloatTensor(inputs)
                inputs = inputs.permute(0, 3, 1, 2).to(device)
                # labels = torch.BoolTensor([sample.one_hot_label for sample in data]).to(device)
                # hingelabels = labels.copy()
                # hingelabels[labels == 0] = -1

                labels = torch.FloatTensor(labels).to(device)
                # hingelabels = torch.LongTensor(hingelabels).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    sigmoid = nn.Sigmoid()
                    # tanh = nn.Tanh()
                    # outputs = tanh(outputs)
                    loss = criterion(outputs, labels)

                    outputs = sigmoid(outputs)

                    # _, preds = torch.max(outputs, 1)
                    # sigmoid = nn.Sigmoid()
                    # preds = sigmoid(outputs)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += mAPMeter(outputs, labels.data)  # torch.sum(preds == labels.data)
                with torch.no_grad():
                    mAP.add(outputs, labels)
                    # running_ap += compute_ap_score(outputs.cpu().numpy(), labels.cpu().numpy())
                i += 1

            epoch_loss = running_loss / len(dataloader.dataset)
            # epoch_acc = running_ap / len(dataloader.dataset)   # mAP.value()     # running_corrects.double() / len(dataloader)
            epoch_acc = mAP.value()

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "intermediate_model.pt")

    # end of epoch here
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


num_classes = 20
num_epochs = 30
batch_size = 50
feature_extract = True

datapath = "../data/VOC2012/"

# set up model
model = vgg16(pretrained=True)
print(model)

set_parameter_requires_grad(model, feature_extract)
for param in model.classifier.parameters():
    param.requires_grad = True
model.classifier[6] = nn.Linear(4096, num_classes)

# load data
datasetclass = get_dataset("VOC2012Dataset")
dataset = datasetclass(datapath, "train")
dataset.set_mode("preprocessed")
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

testset = datasetclass(datapath, "val")
testset.set_mode("preprocessed")
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

params_to_update = model.parameters()
if feature_extract:
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)

# define optimizer and criterion
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# criterion = nn.CrossEntropyLoss()
# criterion = nn.HingeEmbeddingLoss()
criterion = nn.BCEWithLogitsLoss()

# Train and evaluate
model = train_model(model, trainloader, testloader, criterion, optimizer, device, num_epochs=num_epochs)

# save model
torch.save(model.state_dict(), "model.pt")
