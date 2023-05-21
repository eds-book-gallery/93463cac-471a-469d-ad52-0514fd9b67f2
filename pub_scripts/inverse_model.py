import torch
import torch.nn as nn
from torch.autograd import Variable


# fonction réalisant l'inversion variationelle
def model_inverse(entry, cible, model, alpha: float = 0.005):
	criterion = nn.MSELoss()

	X = Variable(entry.clone().detach(), requires_grad=True)
	optimizer = torch.optim.Adam([X], lr=0.0001)

	for i in range(100000):
		current = model(X)
		loss = criterion(current.float(), cible.float()) + 0.01 * criterion(X.float(), entry.float())

		if i % 1000 == 0:
			print(
				f"Iteration {i}:\n\tloss exit {criterion(current, cible)}\n\tloss entry {criterion(entry, X)}"
			)

		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

		if (criterion(current, cible) < alpha):
			print(i)

			break

	return X, current
