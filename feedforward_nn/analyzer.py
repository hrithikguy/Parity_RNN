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


# print cells[0]
# print len(cells[0])
# print cells[0][0]
# print len(cells[0][0])


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
		cell_graph.append(j[0])

hidden_graph = []
for i,j in enumerate(hiddens):
	if (i % 2) == 1:
		hidden_graph.append(j[0])


# print len(cell_graph)
# print len(cell_graph[0])

import sklearn.manifold
tsne = sklearn.manifold.TSNE(n_components = 2)
X_reduced = tsne.fit_transform(cell_graph)

# print X_reduced

# tsnes = []
# tsnes2 = []
# for i in X_reduced:
# 	tsnes.append(i[0])
# 	tsnes2.append(i[1])


# plt.plot(x_axis, weight_graph, 'r')

# plt.show()

# print tsnes

# plt.figure(1)
# plt.plot(x_axis, tsnes, 'g')
# plt.plot(x_axis, tsnes2, 'r')

# plt.show()



# for i in range(len(cell_graph[0])):
# 	cur = []
# 	for j in cell_graph:
# 		cur.append(j[i])

# 	plt.figure(1)
# 	plt.plot(x_axis, cur, 'g')
# 	plt.show()


hidden_state_vectors = []

for i in range(len(hidden_graph[0])):
	cur = []
	for j in hidden_graph:
		cur.append(j[i])

	hidden_state_vectors.append(cur)

	# plt.figure(1)
	# plt.plot(x_axis, cur, 'g')
	# plt.show()


cell_state_vectors = []

for i in range(len(cell_graph[0])):
	cur = []
	for j in cell_graph:
		cur.append(j[i])

	cell_state_vectors.append(cur)

average = []

for i in range(len(hidden_state_vectors[0])):
	cur = 0
	for j in hidden_state_vectors:
		cur += j[i]

	cur = cur/len(hidden_state_vectors)

	average.append(cur)



plt.figure(1)
plt.plot(x_axis, average, 'r')

plt.show()



average = []

for i in range(len(cell_state_vectors[0])):
	cur = 0
	for j in cell_state_vectors:
		cur += j[i]

	cur = cur/len(cell_state_vectors)

	average.append(cur)



plt.figure(1)
plt.plot(x_axis, average, 'r')

plt.show()



# hidden_state_vectors = np.asmatrix(hidden_state_vectors)
# print hidden_state_vectors





# cell_state_vectors = []

# for i in range(len(cell_graph[0])):
# 	cur = []
# 	for j in cell_graph:
# 		cur.append(j[i])

# 	cell_state_vectors.append(cur)

	# plt.figure(1)
	# plt.plot(x_axis, cur, 'g')
	# plt.show()





