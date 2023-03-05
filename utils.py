import torch

### ACCURACY FUNCTIONS ###

# TODO:
# implement Hamming Loss for multi-label accuracy meassure,
# or use "So given a single example where you predict classes A, G, E and the test case has E, A, H, P as the correct ones you end up with Accuracy = Intersection{(A,G,E), (E,A,H,P)} / Union{(A,G,E), (E,A,H,P)} = 2 / 5"
# or use TorchMetrics https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html#multilabel-accuracy


def multiclass_accuracy(y, y_class_idxs):
    '''Returns accuracy in range [0, 1]'''
    return (y == y_class_idxs).sum().item() / len(y)

def multilabel_accuracy(y, y_logits):
    return 999.99

