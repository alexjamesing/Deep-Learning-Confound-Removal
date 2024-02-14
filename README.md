# Confound Orthogonalization Layer for Neural Networks

This Python module introduces a custom TensorFlow/Keras layer designed to orthogonalize input data with respect to a set of specified confounds. The layer ensures that the output is statistically independent of these confounds, which is particularly useful in contexts where it's crucial to control for certain variables to obtain valid and generalizable inferences from neural network models.

## Features

- **Custom Keras Layer**: Seamlessly integrates with existing TensorFlow/Keras models.
- **Batch Orthogonalization**: Allows for batch-wise orthogonalization of inputs with respect to confounds, enabling its use in standard mini-batch gradient descent training procedures.
- **Configurable Parameters**: The layer includes several parameters to control the orthogonalization process, including momentum and epsilon for numerical stability.

## Installation

This module requires TensorFlow to be installed in your environment. It can be installed using pip:

```bash
pip install tensorflow
```

Once TensorFlow is installed, you can add this script to your project and import the custom layer as follows:


~~~

from confound_layer import ConfoundLayer

~~~

## Usage

To use the ConfoundLayer in your neural network model, simply instantiate it and add it to your model like any other Keras layer:

```

confound_layer = ConfoundLayer(tot_num=total_samples_in_dataset)
# Add it to your model
model.add(confound_layer)

```

The ConfoundLayer takes two inputs:

- The data input that you want to orthogonalize.

- The input confounds with respect to which the data input will be orthogonalized.

## Parameters

tot_num: Total number of samples.
epsilon: Offset for batch normalization (default 1e-4).
momentum: Momentum for covariance matrices (default 0.95).
diag_offset: Offset added to the diagonal of the covariance matrix (default 1e-3).

## Example
Here is a brief example of how to integrate the ConfoundLayer into a Keras model:

~~~

# Define your model inputs
data_input = keras.Input(shape=(data_dim,))
confound_input = keras.Input(shape=(confound_dim,))

# Add the ConfoundLayer
confound_layer = ConfoundLayer(tot_num=1000)
orthogonalized_output = confound_layer([data_input, confound_input])

# Continue building your model
# ...

# Create the model
model = keras.Model(inputs=[data_input, confound_input], outputs=orthogonalized_output)

~~~

## Contributing

Contributions to the development and enhancement of this custom layer are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

Author
Alex Ing, 2023

License
This project is open-sourced under the MIT license.
