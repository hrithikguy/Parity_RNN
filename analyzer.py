import numpy as np
import datetime
import os
import cPickle as pickle
import matplotlib.pyplot as plt



train_accuracies = pickle.load(open("train_acc", "rb"))
test_accuracies = pickle.load(open("test_acc", "rb"))
train_losses = pickle.load(open("train_loss", "rb"))
test_losses = pickle.load(open("test_loss", "rb"))
weights = pickle.load(open("weights", "rb"))
biases = pickle.load(open("biases", "rb"))
cells = pickle.load(open("cells", "rb"))
hiddens = pickle.load(open("hiddens", "rb"))



x_axis = np.arange(0, len(train_accuracies))

plt.figure(1)
plt.plot(x_axis, train_accuracies, 'r')
plt.plot(x_axis, test_accuracies, 'g')

plt.show()

plt.figure(1)
plt.plot(x_axis, train_losses, 'r')
plt.plot(x_axis, test_losses, 'g')

plt.show()

plt.figure(1)


weight_graph = []
for i in weights:
	weight_graph.append(i[0][0])


cell_graph = []
for i,j in enumerate(cells):
	if (i % 2 == 0):
		cell_graph.append(j[1])

plt.plot(x_axis, weight_graph, 'r')

plt.show()

plt.figure(1)
plt.plot(x_axis, cell_graph, 'g')
plt.show()

