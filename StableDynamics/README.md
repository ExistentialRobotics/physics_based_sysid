# This contains the code for Example ...

# Dependencies
Jupyter: https://jupyter.org/install

Python packages: matplotlib, scipy, sympy, torch

# Demo 
Follow the steps in StableDynamics.ipynb to run the example

# Error
Block [34] complains about this error. TODO: Investigate
```
TypeError                                 Traceback (most recent call last)
Cell In[48], line 8
      6 for data in train_dataloader:
      7   optimizer.zero_grad()
----> 8   loss, _ = runbatch(model_simple, loss_simple, data)
      9   loss_parts.append(np.array([l.cpu().item() for l in [loss]]))
     11   optim_loss = loss[0] if isinstance(loss, (tuple, list)) else loss

Cell In[24], line 16
     13 Yactual = to_variable(Yactual, cuda=False)#torch.cuda.is_available())
     15 Ypred = model(X)
---> 16 return loss(model, Ypred, Yactual, X, no_proj), Ypred

TypeError: <lambda>() takes 3 positional arguments but 5 were given
```