'''
Utility functions (loss, regularization, etc.)
'''
from typing import List
from autograd.engine import Scalar
from autograd import nn

import numpy as np

def l2_regularization(model: nn.Module, alpha=1e-4) -> float:
    return alpha * sum((p*p for p in model.parameters()))

def svm_max_margin_loss(outputs: List[Scalar], labels: List[int]) -> List[Scalar]:
    return [(1 + -y_i[0]*output_i).relu() for y_i, output_i in zip(labels, outputs)]


def softmax_loss(outputs: List[Scalar], labels: List[int]) -> Scalar:

    outputs_value = [[x[0].value, x[1].value] for x in outputs]
    #labels = [[x[0].value, x[1].value] for x in labels]

    outputs_value = np.array(outputs_value)
    #labels = np.array(labels)

    exp_a = np.exp(outputs_value - np.max(outputs_value))
    softmax = exp_a/exp_a.sum(axis=0, keepdims=True)

    loss = labels - np.log(softmax)
    #loss /= len(outputs)
    loss = [Scalar(l, o) for l, o in zip(loss, outputs)]

    return loss





    # def loss_func(a, x, y):
    
    # N, D = x.shape
    # exp_a = np.exp(a - np.max(a))
    # softmax =  exp_a / exp_a.sum(axis=0, keepdims=True)
        
    
    # loss = np.sum(y - np.log(softmax))
    # loss /= N
    # return loss