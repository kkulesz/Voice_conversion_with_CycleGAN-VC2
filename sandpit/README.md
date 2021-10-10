todo:
1. autograd
2. torch.optim
3. ...

bug todo:
1. https://pytorch.org/docs/stable/generated/torch.tensor.html
```
torch.tensor() always copies data. If you have a Tensor data and want to avoid a copy, use torch.Tensor.requires_grad_() or torch.Tensor.detach(). If you have a NumPy ndarray and want to avoid a copy, use torch.as_tensor().
```

2. https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
```
Returned Tensor shares the same storage with the original one. In-place modifications on either of them will be seen, and may trigger errors in correctness checks. IMPORTANT NOTE: Previously, in-place size / stride / storage changes (such as resize_ / resize_as_ / set_ / transpose_) to the returned tensor also update the original tensor. Now, these in-place changes will not update the original tensor anymore, and will instead trigger an error. For sparse tensors: In-place indices / values changes (such as zero_ / copy_ / add_) to the returned tensor will not update the original tensor anymore, and will instead trigger an error.
```

3. https://towardsdatascience.com/in-place-operations-in-pytorch-f91d493e970e
4. check network and losses with torchviz