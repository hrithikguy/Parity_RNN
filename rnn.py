
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import cPickle as pickle
from datetime import datetime

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


train_x_final = []
train_y_final = []
test_x_final = []
test_y_final = []

for i in train_x:
    cur = []
    for j in i:
        cur.append([j])
    train_x_final.append(cur)



for i in test_x:    
    cur = []
    for j in i:
        cur.append([j])
    test_x_final.append(cur)

train_x = train_x_final
test_x = test_x_final

print train_x
print test_x

# for i,j in enumerate(train_x):
#     print j, train_y[i]


# Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 10
display_step = 1

# Network Parameters
n_input = 1
n_steps = sample_lengths
n_neurons = 200 # 1st layer number of features
n_layers = 2 # 2nd layer number of features
# n_input = sample_lengths # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)
restore_path = "" # load model from restore_path

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
# def rnn(x):
    # Hidden layer with RELU activation

# print x

    # layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # layer_1 = tf.nn.relu(layer_1)
    # # Hidden layer with RELU activation
    # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # layer_2 = tf.nn.relu(layer_2)
    # # Output layer with linear activation
    # out_layer = tf.matmul(layer_2, weights['out']) + biases['out']



    # return out_layer

# Store layers weight & bias
weights = {
    # 'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_neurons, n_classes]))
}
biases = {
    # 'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

x_unstacked = tf.unstack(x, n_steps, 1)
# cell = tf.contrib.rnn.BasicLSTMCell(n_neurons)
# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_neurons) for _ in range(n_layers)])
cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_neurons),  tf.contrib.rnn.BasicLSTMCell(n_neurons)])


outputs,states = tf.contrib.rnn.static_rnn(cell, x_unstacked, dtype=tf.float32)

pred = tf.matmul(outputs[-1], weights['out']) + biases['out']


# # Construct model

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


train_accuracies = []
train_losses = []

test_accuracies = []
test_losses = []


weight_matrix = []
bias_matrix = []
cell_matrix = []
hidden_matrix = []

perfect_accuracy_count = 0

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    if restore_path != "":
        # Restore variables from disk.
        saver.restore(sess, restore_path)
        print("Model restored.")

    # Training cycle
    for epoch in range(training_epochs):

        # time that current epoch started
        epoch_start_time = datetime.now()

        print epoch
        if perfect_accuracy_count > 5:
            break
        avg_cost = 0.
        total_batch = int(len(train_x)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            
            batch_indices = np.random.choice(len(train_x), batch_size)
            batch_x = []
            batch_y = []
            for i in batch_indices:
                batch_x.append(train_x[i])
                batch_y.append(train_y[i])

            # batch_x = tf.convert_to_tensor(batch_x)
            # batch_y = tf.convert_to_tensor(batch_y)
            # print batch_x
            # print batch_y
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, cur_epoch_states = sess.run([optimizer, cost, states], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # time that current epoch finished
        epoch_finish_time = datetime.now()
        # print duration of epoch
        print("epoch="+str(epoch)+" took time = "+str(epoch_time))

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
        # Calculate accuracy
        print("Accuracy:", train_accuracies[-1])

        if train_accuracies[-1] >= 0.999 or test_accuracies[-1] >= 0.999:
            perfect_accuracy_count += 1


        print("Final Test Accuracy")
        # Calculate accuracy
        print("Accuracy:", test_accuracies[-1])

        weight_matrix.append(weights['out'].eval())
        bias_matrix.append(biases['out'].eval())

        for i in cur_epoch_states:
            cell_matrix.append(i[0])
            hidden_matrix.append(i[1])


        # np.savez("weights3_" + str(epoch), states = states[-1])

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

    # Save the variables to disk.
    # model_checkpoint_name based on final test accuracy
    # can improve to include file with all hyperparam info in same dir
    model_checkpoint_name = "/tmp/model"+str(accuracy)+".ckpt"
    save_path = saver.save(sess, model_checkpoint_name)
    print "Model saved in file: %s" % save_path


print train_accuracies
print test_accuracies
print train_losses
print test_losses

dir_name = str(datetime.datetime.now().isoformat())
os.makedirs(dir_name)

pickle.dump(weight_matrix, open(dir_name + "/weights", "wb"))
pickle.dump(bias_matrix, open(dir_name + "/biases", "wb"))
pickle.dump(cell_matrix, open(dir_name + "/cells", "wb"))
pickle.dump(hidden_matrix, open(dir_name + "/hiddens", "wb"))
pickle.dump(train_accuracies, open(dir_name + "/train_acc", "wb"))
pickle.dump(test_accuracies, open(dir_name + "/test_acc", "wb"))
pickle.dump(train_losses, open(dir_name + "/train_loss", "wb"))
pickle.dump(test_losses, open(dir_name + "/test_loss", "wb"))



x_axis = np.arange(0, len(train_accuracies))

plt.figure(1)
plt.plot(x_axis, train_accuracies, 'r')
plt.plot(x_axis, test_accuracies, 'g')

plt.show()

plt.figure(1)
plt.plot(x_axis, train_losses, 'r')
plt.plot(x_axis, test_losses, 'g')

plt.show()
