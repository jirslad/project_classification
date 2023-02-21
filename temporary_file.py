''' Script for testing purposes. '''


################################################################################################
# TEST LOSS FUNCTIONS

# import torch

# # correct class idx / correct target tensor
# y = torch.tensor(0)
# y_tensor_CE = torch.tensor([1, 0, 0], dtype=torch.float32) # output of Softmax should ideally look like this
# print(f"correct class index: {y}, correct target tensor CE: {y_tensor_CE}")

# # raw output of the neural network
# y_logits = torch.tensor([3.2, 5.1, -1.7])
# print(f"logits: {y_logits}")

# # softmax
# y_softmax = y_logits.softmax(dim=0)
# print(f"after softmax: {y_softmax}")

# # log (base e) applied on softmax
# y_log = y_softmax.log()
# logsm = torch.nn.LogSoftmax(dim=0)
# y_log_layer = logsm(y_logits)
# print(f"log likelihood: {y_log}, with LogSoftmax: {y_log_layer}")

# # loss functions definition
# loss_fn_NLL = torch.nn.NLLLoss()
# loss_fn_CE = torch.nn.CrossEntropyLoss()

# # loss NLL
# loss_NLL = loss_fn_NLL(y_log, y)
# print(f"loss NLL with class index: {loss_NLL}")

# # loss CE
# loss_CE = loss_fn_CE(y_logits, y)
# loss_CE_tensor = loss_fn_CE(y_logits, y_tensor_CE.float())
# print(f"loss CE with class index: {loss_CE}, loss CE with tensor: {loss_CE_tensor}")

# #####
# print("======= MULTI-LABEL CLASSIFICATION =======")

# y = torch.tensor([0., 1., 0., 1.])
# logits = torch.tensor([-10., 10., -10., 10.])
# print(f"target: {y}, logits: {logits}")

# sigmoided = torch.sigmoid(logits)
# print(f"after sigmoid: {sigmoided}")

# losses = - (y * torch.log(sigmoided) + (1 - y) * torch.log(1 - sigmoided) )
# print(f"losses: {losses}")

# loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
# loss = loss_fn(logits, y)
# print(f"losses: {loss}, total: {loss.sum()}")

################################################################################################
# TEST SPLITTING DATA

import torch
from torchvision import datasets
from torch.utils.data import random_split

dataset = datasets.DTD(root="datasets",
                       split="test",
                       transform=None,
                       download=True)

print(len(dataset))

train, val, test = random_split(dataset, [0.5, 0.2, 0.3], generator=torch.Generator().manual_seed(42))

print(len(train), len(val), len(test))

################################################################################################



aa = 1