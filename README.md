<p align="center">
<img src="confound_removal_logo.webp" width="300" height="300">
</p>

**About Package**

This Python module introduces a custom TensorFlow/Keras layer designed to orthogonalize input data with respect to a set of specified confounds. The layer ensures that the output is statistically independent of these confounds, which is particularly useful in contexts where it's crucial to control for certain variables to obtain valid and generalizable inferences from neural network models.

**Features**

- Custom Keras Layer: Seamlessly integrates with existing TensorFlow/Keras models.
- Batch Orthogonalization: Allows for batch-wise orthogonalization of inputs with respect to confounds, enabling its use in standard mini-batch gradient descent training procedures.
- Configurable Parameters: The layer includes several parameters to control the orthogonalization process, including momentum and epsilon for numerical stability.

**Simple Example**

Once the layer is imported, it can be added to a model in the normal way:

~~~

import tensorflow as tf
from deepcleaner.ConfoundLayer import ConfoundLayer 

n_samples = 100

# Define the model
input_layer = tf.keras.Input(shape=(1000,))
conv_input_layer = tf.keras.Input(shape=(10,))

# Dense layer with 100 units and ReLU activation function
x = tf.keras.layers.Dense(100, activation='relu')(input_layer)
x = tf.keras.layers.Dense(100, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = ConfoundLayer(n_samples,momentum=0.99)([x, conv_input_layer])

output_layer = tf.keras.layers.Dense(5)(x)

# Create the model
model = tf.keras.Model(inputs=[input_layer,conv_input_layer], outputs=output_layer)

# Model summary
model.summary()

~~~

The layer has one required input denoted here by n_samples: This should be the total number of samples in the training set during model training. This parameter is used internally to help estimate covariance matrices required for orthogonalisation.

Note that it was necessary to use the functional API, as ConfoundLayer takes multiple inputs.

**Installation**

The easiest way to install this package is using pip, which will install all necessary dependencies. This can be achieved by cloning the repository, and typing:

~~~

pip install .

~~~

Note that tensorflow currently supports python 3.9-3.11. If you have a newer version of python installed, it will not be possible to install tensorflow, which is a necessary dependency of this package.


**Tutorial**

In this tutorial, we show how the confound removal layer can be used to remove the effect of nuissance covariates in deep learning models, enhancing their accuracy, reliability, and genralisability. This layer uses the Moore-Penrose pseudo inverse to pathe confounding effects of a set of nuissance covariates. This procedure decorrelates the confounds from the output of the neural network.

**Dataset Description**

First, we download a small dataset for use in the model.

Histological images are microscopic images of tumor tissue that are ubiquitously available in the field of oncology. Expression of the hormone estrogen defines a major clinical subtype in breast cancer. Nevertheless, defining this subtype requires additional testing. Predicting estrogen expression from histological images would therefore be very useful. Nevertheless, confounding effects relating to the imaging acquisition site (mediated through the effects of staining etc) are known to confound deep learning analyses of this kind. Other confounds such as subject age and tumor purity can also have a confounding effect on analyses of this kind.

The datset we use here comes from the TCGA project. Histological images are extremely large, often on the order of several Gigabytes per image. To create this dataset, we broke each image up into small tissue sections called tiles, passed them through the efficientnetB0 convolutional architecture and took the mean over the last layer for all tiles in each image. The histological data sample therefore represents a highly compressed set of features from each processed image.

Genetic data was also downloaded from the TCGA website. We extracted RNASeq data for the ESR1 (estrogen receptor) gene. We then applied global normalization and an inverse normal transformation to gene counts.

~~~

import tensorflow as tf
import pandas as pd
import os as os
import numpy as np
import pkg_resources
from tensorflow.keras import regularizers
from deepcleaner.ConfoundLayer import ConfoundLayer 

def load_csv_as_tensor(file_name):
    # Construct the full path to the CSV file within the package
    file_path = pkg_resources.resource_filename('deepcleaner', os.path.join('data', file_name))
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Convert the pandas DataFrame to a TensorFlow tensor
    tensor = tf.convert_to_tensor(df.values, dtype=tf.float32)
    
    return tensor

# Names of the CSV files
histo_train_file = 'histopathology_train.csv'
histo_test_file = 'histopathology_test.csv'
confound_train_file = 'confound_variables_train.csv'
confound_test_file = 'confound_variables_test.csv'
esr1_train_file = 'ESR1_expression_train.csv'
esr1_test_file = 'ESR1_expression_test.csv'

# Load each CSV file as a separate tensor
histo_train_tensor = load_csv_as_tensor(histo_train_file)
histo_test_tensor = load_csv_as_tensor(histo_test_file)
confound_train_tensor = load_csv_as_tensor(confound_train_file)
confound_test_tensor = load_csv_as_tensor(confound_test_file)
esr1_train_tensor = load_csv_as_tensor(esr1_train_file)
esr1_test_tensor = load_csv_as_tensor(esr1_test_file)

~~~

We first define and train a model that does NOT utilse confound removal:

~~~

## Here, we define a model using the keras/tensorflow functional API
input_layer = tf.keras.Input(shape=(histo_train_tensor.shape[1],))
x = tf.keras.layers.Dense(100,kernel_regularizer = regularizers.L1L2(l1=1e-2, l2=1e-2))(input_layer)
x = tf.keras.layers.Dense(100,kernel_regularizer = regularizers.L1L2(l1=1e-2, l2=1e-2))(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.BatchNormalization()(x)
# Output layer with a single unit (for regression tasks)
output_layer = tf.keras.layers.Dense(1,kernel_regularizer = regularizers.L1L2(l1=1e-2, l2=1e-2))(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mean_squared_error')

# Train the model directly with the NumPy arrays
model.fit(x=histo_train_tensor, y=esr1_train_tensor, epochs=100, batch_size=128, verbose=False)

~~~

We can then evaluate the performance of the model on the training set, and the testing set.

~~~

## evaluate model performance on train set
print(model.evaluate(x=histo_train_tensor, y=esr1_train_tensor))


## evaluate model performance on test set
print(model.evaluate(x=histo_test_tensor, y=esr1_test_tensor))


~~~

we then define a simple function to calculate pearsons pairwise correlation coefficient between the predicted output and the confounds:

~~~

def calculate_correlations(vector, array_2d):
    """
    Calculate the correlation coefficients between a vector and each column of a 2D array.

    Parameters:
    - vector: A 1D NumPy array.
    - array_2d: A 2D NumPy array.

    Returns:
    - A list of correlation coefficients between the vector and each column of the 2D array.
    """
    # Initialize an empty list to store correlation coefficients
    correlations = []

    # Loop through each column in array_2d
    for i in range(array_2d.shape[1]):
        # Calculate correlation coefficient between the vector and the current column
        corr_coeff = np.corrcoef(tf.squeeze(vector), array_2d[:, i])[0,1]
        
        # Append the correlation coefficient to the list
        correlations.append(corr_coeff)

    return correlations

~~~

We can then examine associations between the model outputs and the confounds in both the training set, and the testing set:

~~~

## evaluate assoiciation with confounds on training set
histopathology_prediction = model.predict(histo_train_tensor)
correlations = calculate_correlations(histopathology_prediction,confound_train_tensor)
# Print the correlation coefficients
for i, corr in enumerate(correlations):
    print(f"Correlation with confound column {i}: {corr}")

## evaluate assoiciation with confounds on testing set
histopathology_prediction = model.predict(histo_test_tensor)
correlations = calculate_correlations(histopathology_prediction,confound_test_tensor)
# Print the correlation coefficients
for i, corr in enumerate(correlations):
    print(f"Correlation with confound column {i}: {corr}")

~~~

We can see from these results that the model output correlates highly with come of the confounds


We now define a new neural network model with the addition of the confound removal layer:

~~~

# Input layer specifying the shape of input data
input_layer = tf.keras.Input(shape=(histo_train_tensor.shape[1],))
conv_input_layer = tf.keras.Input(shape=(confound_train_tensor.shape[1],))

# Dense layer with 10 units and ReLU activation function
x = tf.keras.layers.Dense(100, activation='relu',kernel_regularizer = regularizers.L1L2(l1=1e-2, l2=1e-2))(input_layer)
x = tf.keras.layers.Dense(100, activation='relu',kernel_regularizer = regularizers.L1L2(l1=1e-2, l2=1e-2))(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = ConfoundLayer(confound_train_tensor.shape[0],momentum=0.95)([x, conv_input_layer])
# Output layer with a single unit (for regression tasks)
output_layer = tf.keras.layers.Dense(1,kernel_regularizer = regularizers.L1L2(l1=1e-2, l2=1e-2))(x)

# Create the model
model = tf.keras.Model(inputs=[input_layer,conv_input_layer], outputs=output_layer)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mean_squared_error')

~~~


This layer has one required input: The total number of samples the model is trained on. This parameter is needed for the estimation of covariance matrices required for removing confounding effects.

We now train this model:

~~~

# Train the model directly with the NumPy arrays
model.fit(x=[histo_train_tensor,confound_train_tensor], y=esr1_train_tensor, epochs=100, batch_size=128, verbose=False)

~~~

We can then evaluate model performance on the training and testing datasets.

~~~

## evaluate model performance on train set
print(model.evaluate(x=[histo_train_tensor, confound_train_tensor], y=esr1_train_tensor))

## evaluate model performance on test set
print(model.evaluate(x=[histo_test_tensor, confound_test_tensor], y=esr1_test_tensor))

~~~

More importantly in this context, we can see that the model outputs are no longer confounded by the nuissance covariates in either the training set or the testing set:

~~~

## evaluate assoiciation with confounds on training set
histopathology_prediction = model.predict([histo_train_tensor,confound_train_tensor])
correlations = calculate_correlations(histopathology_prediction, confound_train_tensor)
# Print the correlation coefficients
for i, corr in enumerate(correlations):
    print(f"Correlation with confound column {i}: {corr}")

## evaluate assoiciation with confounds on testing set
histopathology_prediction = model.predict([histo_test_tensor,confound_test_tensor])
correlations = calculate_correlations(histopathology_prediction, confound_test_tensor)
# Print the correlation coefficients
for i, corr in enumerate(correlations):
    print(f"Correlation with confound column {i}: {corr}")

~~~

We can see that the model outputs no longer correlate highly with the nuissance covariates we are using to train the model.


It is necessary to use the new .keras format while saving. This is because this format deals with the serialization and deserialization of custom models and layers most stably. This format will save both the model architecture and its weights.

~~~

model.save('/path/to/model/save/model.keras')

~~~


