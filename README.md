# WEATHER-FORECAST

# IMPORTING THE NECESSARY LIBRARIES

CODING
IN[1]:

import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

from torch import nn,optim

import torch.nn.functional as F

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

sns.set(style= 'whitegrid',palette='muted',font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE","#FFDD00","#FF7D00","#FF006D","#93D30C","#8F00FF"]

#HAPPY_COLORS_PALETTE = ["blue","yellow","orange","pink","green","purple"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize']=12,8

RANDOM_SEED=42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)


#notes for IN[1]:
#The code you provided appears to be written in Python
and is importing various libraries including torch, os,
numpy, pandas, tqdm, seaborn, matplotlib, and sklearn.
It seems to be setting up the environment for data
analysis and machine learning tasks.

Here's a breakdown of what each imported library does:

torch: PyTorch is a popular open-source machine learning
framework for building and training neural networks.
os: This library provides functions for interacting with
the operating system, such as file operations and
environment variables.
numpy: A powerful library for numerical computing in 
Python, providing support for large, multi-dimensional
arrays and matrices.
pandas: Pandas is a library for data manipulation and
analysis. It provides data structures like DataFrame,
which allows for efficient handling and processing of
structured data.
tqdm: A library for creating progress bars to track the
progress of iterations or tasks.
seaborn: Seaborn is a Python data visualization library
based on matplotlib. It provides a high-level interface 
for creating informative and attractive statistical graphics.
pylab: A module in matplotlib that combines functionality
from pyplot and numpy into a single namespace.
matplotlib: A plotting library for creating static, animated,
and interactive visualizations in Python.
sklearn: Scikit-learn is a popular machine learning library 
that provides various algorithms and tools for data preprocessing,
model selection, and evaluation.
The code then sets some configurations for the plotting styles,
random number generator seeds, and other parameters related to 
the visualization and machine learning tasks.




BUILDING THE DIGRAPH CODING
IN[2]:
def ann_viz(model,view=True,filename="network.gv"):

    from graphviz import Digraph

    input_layer = 0
    hidden_layers_nr = 0
    layer_types = []
    hidden_layers = []
    output_layer=0
    layers=[layer for layer in model.modules() if type(layer)== torch.nn.Linear]


    for layer in layers:
        if layer == layers[0]:
            input_layer=layer.in_features
            hidden_layers_nr += 1
            if type(layer)==torch.nn.Linear:
                hidden_layers.append(layer.out_features)
                layer_types.append("Dense")
            else:
                raise Exception("Input error")
        else:
            if layer ==layers[-1]:
                output_layer=layer.out_features
            else:
                hidden_layers_nr +=1
                if type(layer)==torch.nn.Linear:
                    hidden_layers.append(layer.out_features)
                    layer_types.append("Dense")
                else:
                    raise Exception("Hidden error")

        last_layer_nodes=input_layer
        nodes_up = input_layer

    g=Digraph("g",filename=filename)

    n=0
    g.graph_attr.update(splines="false",nodesep="0.5",ranksep="0",rankdir='LR')

    #Input layer
    with g.subgraph(name="cluster_input") as c:
        if type(layers[0])==torch.nn.Linear:
            the_label="Input Layer"
            if layers[0].in_features>10:
                the_label+="(+"+str(layers[0].in_features-10)+")"
                input_layer=10
            c.attr(color="white")
            for i in range(0,input_layer):
                n+=1
                c.node(str(n))
                c.attr(labeljust="1")
                c.attr(label=the_label,labelloc="bottom")
                c.attr(rank="same")
                c.node_attr.update(
                    width="0.65",
                    style="filled",
                    shape="circle",
                    color=HAPPY_COLORS_PALETTE[3],
                    fontcolor=HAPPY_COLORS_PALETTE[3],
                )
    for i in range(0,hidden_layers_nr):
        with g.subgraph(name="cluster_"+str(i+1)) as c:
            if layer_types[i]=="Dense":
                c.attr(color="white")
                c.attr(rank="same")
                the_label=f'Hidden Layer{i+1}'
                if layers[i].out_features>10:
                    the_label+="(+"+str(layers[i].out_features-10)+")"
                    hidden_layers[i]=10
                c.attr(labeljust="right",labelloc="b",label=the_label)
                for j in range(0,hidden_layers[i]):
                    n+=1
                    c.node(
                        str(n),
                        width="0.65",
                        shape="circle",
                        style="filled",
                        color=HAPPY_COLORS_PALETTE[0],
                        fontcolor=HAPPY_COLORS_PALETTE[0],

                    )
                    for h in range(nodes_up-last_layer_nodes+1,nodes_up+1):
                        g.edge(str(h),str(n))
                last_layer_nodes=hidden_layers[i]
                nodes_up+=hidden_layers[i]
            else:
                raise Exception("Hidden layer type not supported")

    with g.subgraph(name="cluster_output") as c:
        if type(layers[-1])==torch.nn.Linear:
            c.attr(color="white")
            c.attr(rank="same")
            c.attr(labeljust="1")
            for i in range(1,output_layer+1):
                n+=1
                c.node(
                    str(n),
                    width="0.65",
                    shape="circle",
                    style="filled",
                    color=HAPPY_COLORS_PALETTE[4],
                    fontcolor=HAPPY_COLORS_PALETTE[4],
                )
                for h in range(nodes_up-last_layer_nodes+1,nodes_up+1):
                    g.edge(str(h),str(n))
            c.attr(label="Output Layer",labelloc="bottom")
            c.node_attr.update(
                color="#2ecc71",style="filled",fontcolor="#2ecc71",shape="circle"
                #color="green",style="filled,fontcolor="green",shape="circle"

            )
    g.attr(arrowshape="none")
    g.edge_attr.update(arrowhead="none",color="#707070",penwidth="2")
    #g.edge_attr.update(arrowhead="none",color="gray",penwidth="2")
    if view is True:
        g.view()
    return g
    
    
  NOTES FOR IN[2]:
 The ann_viz function takes a neural network model as input
 and visualizes its architecture using the Graphviz library.
 Here's a breakdown of how the function works:

The function starts by importing the necessary modules, 
including Digraph from Graphviz.
It initializes variables for the input layer, number of
hidden layers, layer types, hidden layers, and output layer.
The function then extracts the linear layers from the
model and populates the variables accordingly, identifying
the input layer, hidden layers, and output layer.
It sets up the Digraph object g with the specified filename.
The function iterates over the layers to create the
visualization. It starts with the input layer, creating a
subgraph for it and adding nodes representing each input feature.
Next, it iterates over the hidden layers, creating a subgraph 
for each hidden layer and adding nodes for each neuron in the layer.
Finally, it creates a subgraph for the output layer and adds
nodes for each output neuron.
The function connects the nodes between layers using edges.
It sets the visual attributes for the graph and edges.
If the view parameter is set to True, it calls g.view() to 
display the graph.
Finally, it returns the graph object.
To use this function, you need to have the Graphviz library
installed. You can pass your neural network model as an
argument to the ann_viz function to visualize its architecture.

Note: The function relies on the HAPPY_COLORS_PALETTE
variable, which should be defined prior to calling the ann_viz function.



IN[3]:
df=pd.read_csv('/content/drive/MyDrive/weather_aus.csv')
df.head()


NOTES FOR [3]:
The code snippet you provided reads a CSV file named "weather_aus.csv"
from Google Drive using the pandas library and assigns it to a
DataFrame variable called df. It then displays the first few rows
of the DataFrame using the head() function.

Here's a breakdown of what the code does:

The pandas library is imported with the alias pd.
The pd.read_csv() function is called to read the CSV file
located at '/content/drive/MyDrive/weather_aus.csv' in Google
Drive. The contents of the CSV file are loaded into the DataFrame df.
The head() function is called on the DataFrame df to display
the first five rows by default.
By executing this code, you are reading the contents of the 
'weather_aus.csv' file from your Google Drive into a DataFrame,
allowing you to work with the data and perform various operations
such as data cleaning, exploration, and analysis.


IN[4]:
df.shape


NOTES[4]:
The df.shape attribute of a DataFrame returns a tuple 
representing the dimensions of the DataFrame. The shape
tuple contains two elements: the number of rows and the
number of columns, respectively.

By executing df.shape, you will get the dimensions of 
the DataFrame df, indicating the number of rows and 
columns in the dataset. It can be useful to quickly
assess the size and structure of the data.

For example, if the output of df.shape is (5000, 10), 
it means that the DataFrame df has 5000 rows and 10
columns.




IN[5]:
cols=['Rainfall','Humidity3pm','Pressure9am','RainToday','RainTomorrow']

df = df[cols]



NOTES FOR IN[5]:
The code snippet you provided selects a subset of columns 
from the DataFrame df and assigns the result to a new
DataFrame df with the same variable name.

Here's a breakdown of what the code does:

The cols list contains the names of the columns you want
to select from the DataFrame.
The square brackets [] are used to index the DataFrame df
with the cols list. This operation selects only the columns
specified in the cols list.
The resulting subset of columns is assigned to a new DataFrame
df, which overwrites the original df variable.
By executing this code, you are creating a new DataFrame df
that contains only the columns specified in the cols list.
The original DataFrame is modified to include only the 
selected columns, and any other columns are excluded.

This can be useful when you want to work with a specific 
subset of columns from a larger DataFrame and focus on
analyzing or manipulating only those columns.



IN[6]:
df['RainToday'].replace({'No':0,'Yes':1},inplace=True)
df['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)

NOTES[6]:
The code snippet you provided replaces the values in the
'RainToday' and 'RainTomorrow' columns of the DataFrame df
with numerical values. Specifically, it replaces the
string values 'No' with 0 and 'Yes' with 1.

Here's a breakdown of what the code does:

df['RainToday'] selects the 'RainToday' column from 
the DataFrame df.
The replace() function is called on the selected column.
Within the replace() function, a dictionary is provided
as an argument to map the old values to new values. In 
this case, it maps 'No' to 0 and 'Yes' to 1.
The inplace=True parameter is set to modify the 'RainToday'
column in-place, meaning the changes will be applied
directly to the DataFrame df without creating a new DataFrame.
The same process is repeated for the 'RainTomorrow' column.
By executing this code, the string values 'No' and 'Yes' 
in the 'RainToday' and 'RainTomorrow' columns of the
DataFrame df will be replaced with the corresponding
numerical values 0 and 1, respectively. This can be
helpful when converting categorical variables into numerical 
representations for certain machine learning algorithms or
analysis tasks.




IN[7]:
df=df.dropna(how='any')
df.head()


NOTES[7]:
The code snippet you provided drops rows containing missing values
(NaN) from the DataFrame df and assigns the result back to the same
variable df. It then displays the first few rows of the modified
DataFrame using the head() function.

Here's a breakdown of what the code does:

The df.dropna() function is called on the DataFrame df.
The how='any' parameter is used to specify that any row 
containing at least one missing value should be dropped.
Alternatively, you can use how='all' to drop rows only if
all values are missing.
The rows with missing values are dropped, and the modified
DataFrame is assigned back to the variable df, overwriting
the original df.
The head() function is called on the modified DataFrame df
to display the first few rows.
By executing this code, any rows in the DataFrame df that
contain missing values will be removed. The resulting 
DataFrame will only contain rows with complete data. 
This is often done to ensure that the data used for
analysis or modeling does not contain missing values,
as some algorithms may not handle missing values well. 
The head() function call then displays the first few
rows of the modified DataFrame for inspection.



IN[8]:
#sns.countplot(df.RainTomorrow);
import seaborn as sns
import matplotlib.pyplot as plt




fig, ax = plt.subplots()


sns.countplot(data=df, x='RainTomorrow', ax=ax)

ax.set_title('Rain Tomorrow Count')
ax.set_xlabel('Rain Tomorrow')
ax.set_ylabel('Count')


plt.show()



NOTES FOR IN[8]:
The code snippet you provided uses the seaborn and matplotlib 
libraries to create a count plot of the 'RainTomorrow' column
in the DataFrame df and displays the plot.

Here's a breakdown of what the code does:

The seaborn library is imported with the alias sns,
and the matplotlib.pyplot library is imported with the
alias plt.
fig, ax = plt.subplots() creates a new figure and axes
object for the plot. The ax variable represents the axes
on which the plot will be drawn.
sns.countplot() is called to create the count plot. It 
takes the DataFrame df as the data source and 'RainTomorrow'
as the variable to be plotted on the x-axis.
The various ax.set_*() functions are used to set the title,
x-label, and y-label of the plot.
plt.show() is called to display the plot.
By executing this code, you will generate a count plot that
shows the distribution of the 'RainTomorrow' variable in the
DataFrame df. The x-axis represents the 'RainTomorrow' values,
and the y-axis represents the count of each value. This type 
of plot is useful for visualizing the balance or imbalance of
a categorical variable's values in a dataset.


IN[9]:
df.RainTomorrow.value_counts()/df.shape[0]

NOTES FOR IN[9]:
The code df.RainTomorrow.value_counts()/df.shape[0] calculates 
the relative frequency or proportion of each unique value in the
'RainTomorrow' column of the DataFrame df.
It provides the ratio of the count of each value to the total
number of rows in the DataFrame.

Here's a breakdown of what the code does:

df.RainTomorrow selects the 'RainTomorrow' column from the
DataFrame df.
The value_counts() function is called on the selected column
to count the occurrences of each unique value.
/ df.shape[0] divides the count of each unique value by the
total number of rows in the DataFrame, which is given by df.shape[0].
The result is a series object with the unique values of 'RainTomorrow'
as the index and the relative frequencies as the values.
By executing this code, you will obtain a series that shows the
proportion or relative frequency of each unique value in the 
'RainTomorrow' column. This can be useful to understand the
distribution of the target variable or to analyze the class 
imbalance in a classification problem.


IN[10]:
x = df[['Rainfall','Humidity3pm','RainToday','Pressure9am']]
y=df[['RainTomorrow']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=RANDOM_SEED)

NOTES FOR IN[10]:
The code snippet you provided splits the DataFrame df into
input features (x) and the target variable (y). It then
performs train-test split on the data using the train_test_split
function from scikit-learn.

Here's a breakdown of what the code does:

x = df[['Rainfall','Humidity3pm','RainToday','Pressure9am']]
selects a subset of columns from the DataFrame df and assigns
it to the variable x. This subset contains the input features 
or independent variables.
y = df[['RainTomorrow']] selects the 'RainTomorrow' column 
from the DataFrame df and assigns it to the variable y. This
represents the target variable or dependent variable.
The train_test_split() function is called with the following
parameters:
x and y are the input features and target variable, respectively.
test_size=0.2 specifies that 20% of the data will be used for
testing, while the remaining 80% will be used for training.
random_state=RANDOM_SEED sets the random seed for reproducibility
of the train-test split. The value of RANDOM_SEED is previously
set as 42.
The train-test split is performed, and the resulting data is
assigned to x_train, x_test, y_train, and y_test, representing
the training and testing sets for the input features and target
variable, respectively.
By executing this code, you will have the input features (x) 
and target variable (y) separated into training and testing sets
(x_train, x_test, y_train, y_test). This allows you to train a 
machine learning model on the training set and evaluate its 
performance on the testing set.



IN[11]:
x_train = torch.from_numpy(x_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())

x_test = torch.from_numpy(x_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)




NOTES FOR IN[11]:
The code snippet you provided converts the training and testing data
from NumPy arrays to PyTorch tensors and prints the shapes of the converted tensors.

Here's a breakdown of what the code does:

x_train = torch.from_numpy(x_train.to_numpy()).float() converts
the training data (x_train) from a NumPy array to a PyTorch tensor
using the torch.from_numpy() function. The .float() method is called 
to ensure that the tensor has a floating-point data type.
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float()) 
converts the training labels (y_train) from a NumPy array to a PyTorch tensor.
Similar to the previous step, it uses torch.from_numpy() and .float() to convert
the data type. The torch.squeeze() function is called to remove any extra
dimensions from the tensor.
x_test = torch.from_numpy(x_test.to_numpy()).float() converts the testing data
(x_test) from a NumPy array to a PyTorch tensor.
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float()) converts
the testing labels (y_test) from a NumPy array to a PyTorch tensor. It uses 
the same steps as for y_train.
print(x_train.shape, y_train.shape) and print(x_test.shape, y_test.shape) are
used to print the shapes of the converted tensors, showing the number of samples
(rows) and the number of input features (columns) in the training and testing sets.
By executing this code, you convert the training and testing data from NumPy
arrays to PyTorch tensors, which can be used as inputs for training and 
evaluating PyTorch models. The printed shapes provide information about 
the dimensions of the tensors, confirming the number of samples and input features in each set.


#Now_i_am_going_to_create_neural_network

IN[12]:
class Net(nn.Module):
  def __init__(self,n_features):
    super(Net,self).__init__()
    self.fc1=nn.Linear(n_features,5)
    self.fc2=nn.Linear(5,3)
    self.fc3=nn.Linear(3,1)

  def forward(self,x):
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))
    
    
NOTES FOR IN[12]:
The code snippet you provided defines a neural network model 
using the nn.Module class from PyTorch. The network consists 
of three fully connected layers (nn.Linear modules) with
ReLU activation functions in the hidden layers and a sigmoid activation function in the output layer.

Here's a breakdown of what the code does:

The Net class is defined, which inherits from the nn.Module class.
The __init__ method is overridden to define the network architecture.
It takes n_features as a parameter, representing the number of input features.
In the __init__ method, three fully connected layers are defined:
self.fc1 = nn.Linear(n_features, 5) defines the first fully connected layer
with n_features input neurons and 5 output neurons.
self.fc2 = nn.Linear(5, 3) defines the second fully connected layer with 5
input neurons (matching the number of output neurons from the previous layer)
and 3 output neurons.
self.fc3 = nn.Linear(3, 1) defines the third fully connected layer with 3 input 
neurons and 1 output neuron.
The forward method is defined to specify the forward pass of the network. 
It takes the input tensor x as a parameter.
In the forward method, the input tensor x is passed through each layer with
a ReLU activation function applied after each hidden layer.
The output of the last layer is passed through a sigmoid activation function
using torch.sigmoid before being returned as the final output of the network.
By defining this neural network model, you have created a basic architecture
that can be used for tasks such as binary classification.




IN[13]:
net = Net(x_train.shape[1])
ann_viz(net,view=False)


NOTES FOR IN[13]:
The code snippet you provided creates an instance of the
Net class with the number of input features (x_train.shape[1]) as the argument.
It then calls the ann_viz function to visualize the neural network architecture using the net model.

Here's a breakdown of what the code does:

net = Net(x_train.shape[1]) creates an instance of the Net class by
passing x_train.shape[1] as the argument. This determines the number
of input features of the network based on the shape of the training data.
ann_viz(net, view=False) calls the ann_viz function to visualize the
neural network architecture using the net model. The view=False parameter
is passed to prevent the visualization from being displayed immediately.
The ann_viz function takes the model as an input and generates a visualization
of the neural network using the Graphviz library. However, since view=False is 
specified, the visualization will not be displayed immediately, and you would
need to use the appropriate method to view or save the visualization.

To view the visualization, you can call ann_viz(net, view=True) instead.
Alternatively, you can save the visualization to a file using
ann_viz(net, view=False, filename="network.gv") and then open the file to view the visualization.



#BUILDING THE RELU GRAPH
IN[14]:
ax = plt.gca()
plt.plot(
  np.linspace(-1,1,5),
  F.relu(torch.linspace(-1,1,steps=5)).numpy()
)
ax.set_ylim([-0.1,1.1]);  


NOTES FOR [14]:
The code snippet you provided plots the output of the 
ReLU activation function using PyTorch's F.relu() function.
It generates a plot of the ReLU function over a specified range.

Here's a breakdown of what the code does:

ax = plt.gca() retrieves the current axes object for the plot.
np.linspace(-1,1,5) generates an array of 5 evenly spaced values
from -1 to 1, inclusive.
F.relu(torch.linspace(-1,1,steps=5)).numpy() applies the
ReLU activation function (F.relu()) to the PyTorch tensor
created by torch.linspace(). The tensor contains the same
5 evenly spaced values from -1 to 1. The .numpy() method is
called to convert the PyTorch tensor to a NumPy array.
plt.plot(...) plots the values generated in the previous step, 
representing the x-values and y-values of the plot.
ax.set_ylim([-0.1,1.1]) sets the y-axis limits of the plot to
range from -0.1 to 1.1.
The resulting plot displays the ReLU function over the specified
range, showing the nonlinear activation behavior where negative
values are set to zero and positive values remain unchanged.
By executing this code, you will generate a plot that visualizes 
the ReLU activation function over the specified range.


#BUILDING THE SIGMOID GRAPH:
IN[15]:
ax=plt.gca()
plt.plot(
  np.linspace(-10,10,100),
  torch.sigmoid(torch.linspace(-10,10,steps=100)).numpy()
)
ax.set_ylim([-0.1,1.1]);


NOTES FOR IN[15]:
The updated code snippet you provided plots the output
of the sigmoid activation function using PyTorch's
torch.sigmoid() function. It generates a plot of the
sigmoid function over a specified range.

Here's a breakdown of what the code does:

ax = plt.gca() retrieves the current axes object for
the plot.
np.linspace(-10,10,100) generates an array of 100 evenly
spaced values from -10 to 10, inclusive.
torch.sigmoid(torch.linspace(-10,10,steps=100)).numpy()
applies the sigmoid activation function (torch.sigmoid()) 
to the PyTorch tensor created by torch.linspace(). The tensor 
contains the same 100 evenly spaced values from -10 to 10.
The .numpy() method is called to convert the PyTorch tensor to a NumPy array.
plt.plot(...) plots the values generated in the previous step,
representing the x-values and y-values of the plot.
ax.set_ylim([-0.1,1.1]) sets the y-axis limits of the plot to
range from -0.1 to 1.1.
The resulting plot displays the sigmoid function over the specified range, 
showing the S-shaped curve where the output is squeezed between 0 and 1.
By executing this code, you will generate a plot that visualizes the sigmoid activation function over the specified range.



#CREATING THE BINARY CROSS ENTROPY
IN[16]:
criterion=nn.BCELoss()


NOTES FOR IN[16]:
The code snippet criterion = nn.BCELoss() defines the binary
cross-entropy loss function (BCELoss) from the torch.nn module.

Here's a breakdown of what the code does:

nn.BCELoss() creates an instance of the binary cross-entropy 
loss function (BCELoss), which is commonly used for binary
classification problems. BCE stands for Binary Cross-Entropy.
The criterion variable is assigned the instance of BCELoss for 
later use in the training process.
By using the binary cross-entropy loss function, you can compute
the loss between the predicted outputs of your model and the true
binary labels. The loss value indicates how well the model is performing,
and during training, the goal is to minimize this loss by adjusting the
model's parameters through gradient descent or other optimization algorithms.


IN[17]:
optimizer = optim.Adam(net.parameters(),lr=0.001)


NOTES FOR IN[17]:
he code snippet optimizer = optim.Adam(net.parameters(), lr=0.001) 
creates an instance of the Adam optimizer from the torch.optim module 
and associates it with the parameters of the net model.

Here's a breakdown of what the code does:

optim.Adam(...) creates an instance of the Adam optimizer. Adam is an
optimization algorithm commonly used for training neural networks.
net.parameters() returns an iterable of the model's parameters.
It provides the parameters that need to be updated during the optimization process.




IN[18]:
device = torch.device("cuda:0"if torch.cuda.is_available()else"cpu")

NOTES FOR [18]:
The code snippet device = torch.device("cuda:0" if torch.cuda.is_available()
else "cpu") assigns a device to be used for tensor computations in PyTorch.
It checks if a CUDA-enabled GPU is available and if so, assigns the device
as "cuda:0". Otherwise, it assigns the device as "cpu".

Here's a breakdown of what the code does:

torch.cuda.is_available() checks if a CUDA-enabled GPU is available for computation.
If torch.cuda.is_available() returns True, indicating that a GPU is available, 
the device is set as "cuda:0". The 0 represents the index of the GPU if multiple GPUs are available.
If torch.cuda.is_available() returns False, indicating that a GPU is not available
or CUDA is not installed, the device is set as "cpu".
The device variable is assigned the selected device to be used for tensor computations.
By using the device variable, you can explicitly specify whether to use the CPU or GPU
for tensor operations in your PyTorch code. This allows you to take advantage of 
GPU acceleration if a compatible GPU is available, improving the performance of your computations.
The parameters returned by net.parameters() are passed as an argument to the
Adam optimizer, indicating which parameters should be optimized.
lr=0.001 sets the learning rate of the optimizer to 0.001. 
The learning rate determines the step size at each iteration during the optimization process.
By using the Adam optimizer, you can update the model's parameters based on the
gradients computed during backpropagation. The optimizer uses the gradients and 
the learning rate to adjust the model's parameters, aiming to minimize the loss function
and improve the model's performance during training.


IN[19]:
x_train=x_train.to(device)
y_train=y_train.to(device)

x_test=x_test.to(device)
y_test=y_test.to(device)


NOTES FOR[19]:
The code snippet x_train = x_train.to(device), y_train = y_train.to(device), 
x_test = x_test.to(device), and y_test = y_test.to(device) moves the 
training and testing data tensors to the specified device.

Here's a breakdown of what the code does:

x_train.to(device) moves the x_train tensor to the specified device (cuda:0 GPU or cpu).
y_train.to(device) moves the y_train tensor to the same device.
x_test.to(device) moves the x_test tensor to the same device.
y_test.to(device) moves the y_test tensor to the same device.
By moving the tensors to the device, you ensure that the tensor
computations, such as forward and backward passes during training
and evaluation, are performed on the specified device (GPU or CPU).
This is necessary when working with GPU-accelerated computations to take 
advantage of the GPU's parallel processing capabilities.



IN[20]:
net = net.to(device)
criterion=criterion.to(device)

NOTES FOR IN[20]:
The code snippet net = net.to(device) and 
criterion = criterion.to(device) move the 
neural network model (net) and the loss criterion (criterion) to the specified device.

Here's a breakdown of what the code does:

net.to(device) moves the neural network model
(net) to the specified device (cuda:0 GPU or cpu).
criterion.to(device) moves the loss criterion
(criterion) to the same device.
By moving the model and the loss criterion to the device,
you ensure that the computations and operations involving
them are performed on the specified device. This is necessary for 
the model to utilize the GPU's computational power if a GPU is available
or to ensure compatibility and consistency between the model and the criterion when running computations on the CPU.


#Now we are calculating the accuracy
IN[21]:
def calculate_accuracy(y_true,y_pred):
  predicted=y_pred.ge(.5).view(-1)
  return(y_true==predicted).sum().float()/len(y_true)
  
  
NOTES[21]:
The function calculate_accuracy(y_true, y_pred) calculates
the accuracy of the predicted values (y_pred) compared to the true values (y_true).

Here's a breakdown of what the code does:

y_pred.ge(.5) compares each element in y_pred with the
threshold of 0.5, resulting in a tensor of Boolean values indicating whether each element is greater than or equal to 0.5.
.view(-1) reshapes the tensor to have a single dimension, as required for comparison with y_true.
predicted stores the reshaped tensor of Boolean values.
(y_true == predicted).sum().float() calculates the number of
correct predictions by comparing the elements of y_true with predicted
and summing the occurrences where they are equal. The result is cast to a float.
The number of correct predictions is divided by the length of y_true to compute the accuracy.
The accuracy value is returned.
This function is useful for evaluating the performance of a
binary classification model by comparing the predicted values to the true values 
and calculating the percentage of correct predictions.



NOTES[22]:
def round_tensor(t,decimal_places=3):
  return round(t.item(),decimal_places)

for epoch in range(1000):
    y_pred = net(x_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred,y_train)

    if epoch% 100 == 0:
      train_acc = calculate_accuracy(y_train,y_pred)

      y_test_pred=net(x_test)
      y_test_pred = torch.squeeze(y_test_pred)

      test_loss = criterion(y_test_pred,y_test)
      test_acc = calculate_accuracy(y_test,y_test_pred)

      print(


f'''epoch{epoch}
Train set - loss:{round_tensor(train_loss)},accuracy:{round_tensor(train_acc)}
Train set - loss:{round_tensor(test_loss)},accuracy:{round_tensor(test_acc)}
''')
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()  
    
    
    
    
NOTES FOR IN[22]:
The code snippet provided is a training loop for the neural network model. Here's a breakdown of what the code does:

The loop runs for 1000 epochs.
Inside each epoch:
y_pred is obtained by passing the training input x_train through the neural network (net).
y_pred is reshaped using torch.squeeze() to remove any dimensions of size 1.
The training loss is calculated using the specified loss criterion (criterion) 
and the predicted values (y_pred) and true values (y_train).
If the current epoch is a multiple of 100, the following steps are performed for evaluation purposes:
The accuracy of the model on the training set is calculated using the
calculate_accuracy() function and the predicted and true values.
y_test_pred is obtained by passing the test input x_test through the neural network.
y_test_pred is reshaped using torch.squeeze().
The test loss is calculated using the specified loss criterion (criterion) and 
the predicted values (y_test_pred) and true values (y_test).
The accuracy of the model on the test set is calculated using the calculate_accuracy()
function and the predicted and true values.
The epoch number, training loss, training accuracy, test loss, and test accuracy are printed.
The optimizer's gradients are reset to zero using optimizer.zero_grad().
The gradients of the training loss are computed using train_loss.backward().
The optimizer updates the model's parameters using optimizer.step().
This training loop iterates over the data, calculates the loss and accuracy,
performs backpropagation, and updates the model's parameters for a specified number of epochs.


SAVING THE MODEL ( YOU CAN SEE IN FILE SECTION 'model.pth' file is displayed and you can download)
IN[23]:
MODEL_PATH = 'model.path'

torch.save(net,MODEL_PATH)


NOTES FOR IN[23]:
The code torch.save(net, MODEL_PATH) saves the neural network model (net) to the specified file path (MODEL_PATH).

Here's what the code does:

torch.save() is a function provided by PyTorch for saving models, tensors, and other objects to a file.
net is the model object that you want to save.
MODEL_PATH is a string specifying the file path where you want to save the model.
The torch.save() function serializes the model object and saves it to the specified file path.
After executing this code, the model will be saved to the specified file path and can be
loaded later using torch.load() to resume training or make predictions.





LOAD THE MODEL 
IN[24]:
net = torch.load(MODEL_PATH)

NOTES FOR IN[24]:
The code net = torch.load(MODEL_PATH) loads the
saved neural network model from the specified file path (MODEL_PATH) and assigns it to the variable net.

Here's what the code does:

torch.load() is a function provided by PyTorch for loading saved models, tensors, and other objects from a file.
MODEL_PATH is a string specifying the file path from where you want to load the model.
The torch.load() function reads the serialized model object from the specified file path and returns it.
The returned model object is then assigned to the variable net, allowing you to use it for further computations or predictions.
After executing this code, the model saved at the specified file path will be loaded and ready to be used for any desired tasks.
    

IN[25]:
classes = ['No rain','Raining']

y_pred = net(x_test)
y_pred = y_pred.ge(.5).view(-1).cpu()
y_test = y_test.cpu()
print(classification_report(y_test,y_pred,target_names=classes))



NOTES FOR IN[25]:
There seems to be a typo in your code. 
The line print(classification_report(y_test, y-pred, target_names=classes))
should be corrected to print(classification_report(y_test, y_pred, target_names=classes))
(replacing y-pred with y_pred). Here's the corrected code:

The code performs the following steps:

classes is a list containing the class labels for the classification problem.
y_pred contains the predicted values obtained by passing the test input x_test
through the loaded neural network model (net). The ge(.5) function is used to convert 
the predicted probabilities to binary values based on a threshold of 0.5. view(-1) reshapes the tensor to a 1-dimensional array.
Both y_pred and y_test tensors are moved to the CPU for compatibility with the classification_report() function.
classification_report() generates a classification report that includes precision,
recall, F1-score, and support for each class based on the predicted and true labels.
The target_names argument specifies the names of the classes.
The classification report is printed using print().
Make sure to fix the typo and execute the corrected code to obtain the classification report for your model's predictions.





CREATING THE CONFUSION MATRIX
IN[26]:
cm=confusion_matrix(y_test,y_pred)

df_cm = pd.DataFrame(cm,index=classes,columns=classes)

hmap=sns.heatmap(df_cm,annot=True,fmt="d")

hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(),rotation=0,ha='right')

hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(),rotation=30,ha='right')

plt.ylabel('True label')

plt.xlabel('Predicted label')


NOTES FOR [26]:
The code performs the following steps:

cm calculates the confusion matrix using the predicted and true labels.
df_cm creates a DataFrame from the confusion matrix with class labels as row and column indices.
hmap creates a heatmap using seaborn's heatmap function. The annot=True argument
adds the numerical values to the heatmap cells, and fmt="d" formats the values
as integers.
hmap.yaxis.set_ticklabels() and hmap.xaxis.set_ticklabels() set the tick labels 
on the y-axis and x-axis, respectively, with appropriate rotation and alignment.
plt.ylabel() and plt.xlabel() set the labels for the y-axis and x-axis, respectively.
plt.show() displays the generated heatmap.
Execute the code to visualize the confusion matrix heatmap with annotated values
for the predicted and true labels.



IN[27]:
def will_it_rain(rainfall,humidity,raintoday,pressure):
  t = torch.as_tensor([rainfall,humidity,rain_today,pressure])\
      .float()\
      .to(device)
  output=net(t)
  return output.ge(0.5).item()    
  
  
  NOTES FOR [27]:
  The code performs the following steps:

The function receives the input parameters: rainfall, humidity, raintoday, and pressure.
The input values are converted into a PyTorch tensor using torch.tensor()
and then to the appropriate data type using .float().
The tensor t is moved to the same device as the neural network (device).
The tensor t is passed through the neural network (net) to obtain the output prediction.
The output prediction is converted to a binary label using output.ge(0.5).item(),
where values greater than or equal to 0.5 are considered as "Raining" and values 
less than 0.5 are considered as "No rain".
The predicted label is returned from the function using return.
You can now use the will_it_rain() function to make predictions by passing the 
relevant input values for rainfall, humidity, rain today, and pressure.




IN[28]:
prediction = will_it_rain(rainfall=10, humidity=10, raintoday=1, pressure=2)
print(prediction)



NOTES FOR IN[28]:
Make sure to use raintoday=1 for "Yes" or raintoday=0 for "No" to 
indicate if it is raining today. The function will return 1 if it 
predicts rain and 0 if it predicts no rain.

Please note that for the raintoday parameter, the value True should be 
changed to 1 to match the encoding used in the data preprocessing step.



IN[29]:
prediction = will_it_rain(rainfall=0, humidity=1, raintoday=0, pressure=100)
print(prediction)


NOTES FOR [29]:
In this example, the input values are rainfall=0 (no rainfall), humidity=1, 
raintoday=False (indicating no rain today), and pressure=100.
The function will return 0 if it predicts no rain and 1 if it predicts rain.

Please note that for the raintoday parameter, the value False should
be changed to 0 to match the encoding used in the data preprocessing step.











    
    
    







