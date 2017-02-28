# Plain outline of this notebook
## **funcs to preprocess data**
Build the following functions to do the tasks described for each function: 
- **download** and **unzip** tar.gz
- **extract features and labels** from 1 out of 5 training batches
- **display stats** of this batch and **print out an image**
- **normalize features** down to range 0 to 1
- **one-hot-encode** labels
- **randomize data** (done by cifar-10 maker)
- **transform and save in pickle**
    - helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
    - apply normalization, one-hot-encode to entire dataset
    - save them in pickle file

## **funcs to build input tensors and layers**
Build functions to create tensors for each input and each layer
- **func to build input tensor**: build a feature tensor, a label tensor, a dropout tensor waiting to be fed
- **func to build a conv layer tensor**
    - filter_weight = [width, height, in_channels, out_channels)
    - strides = [1, width, height, 1]
    - with `tf.nn.conv2d(input_tensor, filter_weights, filter_strides, padding)`
    - add bias to conv layer with `tf.nn.bias_add`
    - apply non-linear activation to conv layer with `tf.nn.relu`
- **func to build max-pool layer tensor** 
    - pool filter_size = [1, width, height, 1]
    - pool strides = [1, width, height, 1]
    - pool layer by `tf.nn.max_pool(input_tensor, filter, strides, padding)`
- **func to build a flatten layer tensor**
    - get input layer's shape: `t.get_shape()`
    - calc num of elements from shape: `tensorShape.num_elements()`
    - flatten_layer = tf.reshape(x_tensor, [-1, num_features])
- **func to build a fc_layer tensor**
    - get weights and bias shape right
    - layer_fc= tf.add(tf.matmul(x_tensor, weights), biases)
- **func to build output_layer**: similar to fc_layer

## **func to add layers into a CNN model**
> Create a function to run all tensor functions above with parameters inputs to build a specific model (forward pass)
- **set parameters**: 
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

## **Build A Forward Pass**
> run all funcs created above to build a network from input tensors to layers, then to accuracy tensor
- reset all variables of default graph
- run funcs to create tensors for features, labels and dropout
- run CNN model function to get logits tensor through a forward pass
- name logits for loading from disk later
- build a cost tensor using softmax + cross_entropy + take_average on logits and true labels
- build a optimizer tensor with AdamOpt and cost
- build a tensor for correct_pred
- build a tensor for accuracy

## more functions 
**Build Backward Pass/Optimization function**
- do a session run
- feed features, labels, keep_prob data
- run optimizer tensor 

** print stats function**
> this func help to print loss and accuracy after each epoch and the last iteration
- do a session run
- feed data like above, except keep_prob = 1
- run cost tensor compute to get loss
- feed data with valid_features, valid_label, and 1
- run accuracy tensor to compute valid_acc on validation set
- print loss and valid_acc

## Training and optimizing or forward-backward pass looping
**Set hyperparameter**
- epochs
- batch_size
- keep_prob

**run a single large batch to update weights and print loss and valid_acc**
- open a session
- initialize all variables of default graph
- for each epoch, and for each small batch of the large batch
- training or optimizing: forward pass = get loss, backward pass = update weights
- at end of each epoch, print loss and valid_acc on validation set

**run all 5 large batches to update weights and print loss and valid_acc**
- open a session
- initialize all variables of default graph
- for each epoch, 
- and for each small batch of every large batches (5)
- training or optimizing: forward pass = get loss, backward pass = update weights
- at end of each large batch, print loss and valid_acc on validation set
- save the model or the session into a file

**Test Model**
- make sure batch_size is set as previous or 64
- set model_file path
- load test set (test_features, test_labels) from pickle fine
- assign a name to default graph
- open a session with graph name
- import the saved model and restore all variables of the model onto the current default graph
- get all variables of saved model by name and assign them new names
- get test_features and test_labels into many small batches
- compute accuracy tensor with each small batch
- add up each small batch's accuracy and count up num of accuracies
- print and compute the final accuracy
- print out 4 random sample images
    - randomly sample 4 images
    - compute a tensor to get top 3 predictions of an image
    - print out images and predictions
