
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from numpy import binary_repr


sample_lengths = 15



numbers = list(np.array(np.random.choice(range(pow(2, sample_lengths)), min(10000, pow(2, sample_lengths)), replace=False)))
# print numbers



train_numbers = []
test_numbers = []

for i,j in enumerate(numbers):
    split_index = int(0.8 * len(numbers))
    if i < split_index:
        train_numbers.append(j)
    else:
        test_numbers.append(j)

# print train_numbers
# print len(train_numbers)
# print test_numbers
# print len(test_numbers)



train_x = []
train_y = []
test_x = []
test_y = []

for i in train_numbers:
    train_x.append(map(int, list(binary_repr(i, sample_lengths))))

for i in train_x:
    if (sum(i) % 2) == 0:
        train_y.append([1, 0])
    else:
        train_y.append([0, 1])    


for i in test_numbers:
    test_x.append(map(int, list(binary_repr(i, sample_lengths))))

for i in test_x:
    if (sum(i) % 2) == 0:
        test_y.append([1, 0])
    else:
        test_y.append([0, 1])    


# for i,j in enumerate(train_x):
#     print j, train_y[i]


# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 5
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = sample_lengths # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()


train_accuracies = []
train_losses = []

test_accuracies = []
test_losses = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        print epoch
        avg_cost = 0.
        total_batch = int(len(train_x)/100)
        # Loop over all batches
        for i in range(total_batch):
            
            batch_indices = np.random.choice(len(train_x), 100)
            batch_x = []
            batch_y = []
            for i in batch_indices:
                batch_x.append(train_x[i])
                batch_y.append(train_y[i])

            # print batch_x
            # print batch_y
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        #train_losses.append(cost.eval({x: train_x, y: train_y}))
        #print("Epoch:", '%04d' % (epoch+1), "cost=", \
        #    "{:.9f}".format(avg_cost))
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))
        #test_accuracies.append(accuracy.eval({x: test_x, y: test_y}))

        #print("Train Accuracy")
        #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #print("Accuracy:", accuracy.eval({x: train_x, y: train_y}))
        train_accuracies.append(accuracy.eval({x: train_x, y: train_y}))
        test_losses.append(cost.eval({x: test_x, y: test_y}))
        test_accuracies.append(accuracy.eval({x: test_x, y: test_y}))
        train_losses.append(cost.eval({x: train_x, y: train_y}))


    print("Final Train Accuracy")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: train_x, y: train_y}))


    print("Final Test Accuracy")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))


print train_accuracies
print test_accuracies
print train_losses
print test_losses


x_axis = np.arange(0, len(train_accuracies))

plt.figure(1)
plt.plot(x_axis, train_accuracies, 'r')
plt.plot(x_axis, test_accuracies, 'g')

plt.show()

plt.figure(1)
plt.plot(x_axis, train_losses, 'r')
plt.plot(x_axis, test_losses, 'g')

plt.show()
