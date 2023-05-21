from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Variable


# fonction réalisant l'inversion variationelle
def model_inverse(entry: torch.Tensor, target: torch.Tensor, model, alpha: float = 0.005) -> tuple:
	"""
	Performs model inverse optimization to modify the given input array to approximate the target array.

	Parameters
	----------
	entry : torch.Tensor
		Input array to be modified.
	target : torch.Tensor
		Target array to be approximated.
	model : object
		The model used for prediction.
	alpha : float, optional
		Threshold value for the loss, indicating the desired approximation level. Default is 0.005.

	Returns
	-------
	tuple
		Tuple containing the modified input array and the predicted array after optimization.
	"""

	criterion = nn.MSELoss()

	X = Variable(entry.clone().detach(), requires_grad=True)
	optimizer = torch.optim.Adam([X], lr=0.0001)

	for i in range(100_000):
		current = model(X)
		loss = criterion(current.float(), target.float()) + 0.01 * criterion(X.float(), entry.float())

		if i % 1000 == 0:
			print(
				f"Iteration {i}:\n\tloss exit {criterion(current, target)}\n\tloss entry {criterion(entry, X)}"
			)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if criterion(current, target) < alpha:
			print(i)
			break

	return X, current
