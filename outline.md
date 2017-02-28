# Plain outline of this notebook
## **preprocessing data**
- **download** and **unzip** tar.gz
- **extract features and labels** from 1 out of 5 training batches
- **display stats** of this batch and **print out an image**
- **normalize features** down to range 0 to 1
- **one-hot-encode** labels
- **randomize data** (done by cifar-10 maker)
- **save preprocessed data into a pickle file**, and load this pickle file directly for convenience of training

## **building models**
- build a feature tensor, a label tensor, a dropout tensor waiting to be fed
- build a conv layer 
    - filter_weight = [width, height, in_channels, out_channels)
    - strides = [1, width, height, 1]
    - with `tf.nn.conv2d(input_tensor, filter_weights, filter_strides, padding)`
    - add bias to conv layer with `tf.nn.bias_add`
    - apply non-linear activation to conv layer with `tf.nn.relu`
- build max-pool layer: 
    - pool filter_size = [1, width, height, 1]
    - pool strides = [1, width, height, 1]
    - pool layer by `tf.nn.max_pool(input_tensor, filter, strides, padding)`
- build a flatten layer
    - get input layer's shape: `t.get_shape()`
    - calc num of elements from shape: `tensorShape.num_elements()`
    - flatten_layer = tf.reshape(x_tensor, [-1, num_features])
- build a fc_layer
    - get weights and bias shape right
    - layer_fc= tf.add(tf.matmul(x_tensor, weights), biases)
- build output_layer: similar to fc_layer

**Build CNN model function**
- set parameters: 
    - conv_out_channels = 20,
    - conv_filter_shape = (5,5) or [5, 5, 3, 20]
    - conv_strides_shape = (2,2) or [1, 2,2, 1]
    - pool_filter_shape = (2,2) or [1, 2,2, 1]
    - pool_strides = (2,2) or [1,2,2,1] 
- build a conv_layer and add a max_pool layer
- add a flatten layer 
- add a fc_layer
- apply dropout to fc_layer 
- add output layer
- return tensor

**Build Neural Net**
- reset all variables of default graph
- create tensors for features, labels and dropout
- run the CNN model function, return logits tensor
- name logits for loading from disk
- build cost tensor using softmax + cross_entropy + take_average on logits and true labels
- build optimizer tensor with AdamOpt and cost
- build a tensor for correct_pred
- build a tensor for accuracy

**Build a NN training function**
- do a session run
- feed features, labels, keep_prob data
- run optimizer tensor 

**print stats function**
- do a session run
- feed data like above, except keep_prob = 1
- run cost tensor compute to get loss
- feed data with valid_features, valid_label, and 1
- run accuracy tensor to compute valid_acc on validation set
- print loss and valid_acc

**Set hyperparameter**
- epochs
- batch_size
- keep_prob

**run a single large batch to get loss and valid_acc**
- open a session
- initialize all variables of default graph
- for each epoch, and for each small batch of the large batch
- train network to optimize or update weights
- at end of each epoch, print loss and valid_acc on validation set

**run all 5 large batches to get loss and valid_acc**
- open a session
- initialize all variables of default graph
- for each epoch, 
- and for each small batch of every large batches (5)
- train network to optimize or update weights
- at end of each large batch, print loss and valid_acc on validation set
- save the model or the session into a file

**Test Model**
- make sure batch_size is set as previous or 64
- set model_file path
- build a test model function: 
- load test set (test_features, test_labels) from pickle fine
- assign a name to default graph
- open a session with graph name
- import the saved model and restore all variables of the model onto the current default graph
- get all variables of saved model by name and assign them a new variable name
- get test_features and test_labels into many small batches
- compute accuracy tensor with each small batch
- add up each small batch's accuracy and count up num of accuracies
- print and compute the final accuracy
- print out 4 random sample images
    - randomly sample 4 images
    - compute a tensor to get top 3 predictions of an image
    - print out images and predictions
