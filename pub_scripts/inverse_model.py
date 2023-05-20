import torch
import torch.nn as nn
from torch.autograd import Variable


# fonction réalisant l'inversion variationelle
def model_inverse(entre, cible, model, alpha=0.005):
	criterion = nn.MSELoss()

	X = Variable(entre.clone().detach(), requires_grad=True)
	optimizer = torch.optim.Adam([X], lr=0.0001)

	for i in range(100000):
		current = model(X)
		loss = criterion(current.float(), cible.float()) + 0.01 * criterion(X.float(), entre.float())

		if i % 1000 == 0:
			print(
				f"Iteration {i}:   loss exit {criterion(current, cible)} loss entry {criterion(entre, X)}"
			)

		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

		if (criterion(current, cible) < alpha):
			print(i)

			break

	return X, current
