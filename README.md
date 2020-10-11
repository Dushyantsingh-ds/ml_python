# ml_python
### What is NumPy?
NumPy is a scientific computing package (library) for python programming language.
Numpy is a powerful Python programming language library to solve numerical problems.

### What is the meaning of NumPy word?

Num stands for numerical and Py stands for Python programming language.
Python NumPy library is especially used for numeric and mathematical calculation like linear algebra, Fourier transform, and random number capabilities using Numpy array.
NumPy supports large data in the form of a multidimensional array (vector and matrix).

### Prerequisites to learn Python NumPy Library
NumPy Python library is too simple to learn. The basic python programming and its other libraries like Pandas and Matplotlib knowledge will help to solve real-world problems.
Some basic mathematics like vector and metrics are plus point.

### Representation of NumPy multidimensional arrays
![alt text](https://i2.wp.com/indianaiproduction.com/wp-content/uploads/2019/06/Python-NumPy-Tutorial.png?resize=768%2C432&ssl=1)

Python NumPy Tutorial
Fig 1.1 Multidimensional NumPy arrays
The above figure 1.1 shows one dimensional (1D), two dimensional (2D) and three dimensional (3D) NumPy array

$ One Dimensional NumPy array (1D): It means the collection of homogenous data in a single row (vector).

$ Two Dimensional NumPy arrays (2D): It means the collection of homogenous data in lists of a list (matrix).

$ Three Dimensional NumPy arrays (3D): It means the collection of homogenous data in lists of lists of a list (tensor).

### Why NumPy array instead of Python List ?
If you observe in Fig 1.1. To create a NumPy array used list. NumPy array and Python list are both the most similar. NumPy has written in C and Python. That’s a reason some special advantage over Python list is given below.

$ Faster
$ Uses less memory to store data.
$ Convenient.
![alt text](https://i0.wp.com/indianaiproduction.com/wp-content/uploads/2019/06/Python-NumPy-Tutorial-NumPy-vs-Python-List.png?resize=768%2C234&ssl=1)

### Why use NumPy for machine learning, Deep Learning, and Data Science?
![alt text](https://i0.wp.com/indianaiproduction.com/wp-content/uploads/2019/06/Python-NumPy-Tutorial-for-machine-learning-data-science-1.png?w=589&ssl=1)
Python NumPy Tutorial for machine learning data science
Fig 1.2 NumPy for Machine Learning
To solve computer vision and MRI, etc. So for that machine learning model want to use images, but the ML model can’t read image directly. So need to convert image into numeric form and then fit into NumPy array. which is the best way to give data to the ML model.

![alt text](https://i0.wp.com/indianaiproduction.com/wp-content/uploads/2019/06/Python-NumPy-Tutorial-for-machine-learning-data-science-2.png?w=527&ssl=1)
Fig 1.3 NumPy for Machine Learning

Machine Learning model also used to solve business problems. But we can’t provide ‘.tsv’, ‘.csv’ files data to the ML model, So for that also need to use NumPy array.

### In python NumPy tutorial at this movement, we have learned about Python NumPy Library theoretically but its time to do practicals.

### Practical Session of Python NumPy Tutorial
### How to install Python NumPy Library (package)?
To use the NumPy package first of all need to install it.

If you installed Anaconda Navigator and launched Jupyter Notebook or Spyder then no need to install NumPy. Anaconda Navigator installed NumPy itself. If you are using another IDE instead of Anaconda Navigator then follow below command in command prompt or terminal to install Python NumPy Library (Package).

`pip install numpy`

While entering the above command, your system has an internet connection because ‘pip’ package download ‘numpy’ package and then install it. After successful installation, You are ready to take the advantages of the NumPy package.

### How to import NumPy Library in IDE or How to use it?
To use NumPy first import it. For import NumPy, follows below syntax in the python program file.

`import numpy as np`
#### import: import keyword imports the NumPy package in the current file.

#### as:  as is a keyword used to create sort name of NumPy.

#### np: np is a short name given to NumPy, you can give any name (identifier) instead of it. If we use NumPy name in the program repeatedly so it will consume typing time and energy as well so for that we gave a short name for our convenience.

### Flow the below syntax to create NumPy ndarray (multidimensional array)
### How to create one dimensional NumPy array?
To create python NumPy array use array() function and give items of a list.

#### Syntax: numpy.array(object, dtype=None, copy=True, order=’K’, subok=False, ndmin=0)

```
import numpy as np # import numpy package
one_d_array = np.array([1,2,3,4]) # create 1D array
print(one_d_array) # printing 1d array 
```

```
Output >>> [1 2 3 4] 
```

### How to create two dimensional NumPy array?
To create 2D array, give items of lists in list to NumPy array() function.


```
import numpy as np # impoer numpy package
two_d_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # create 1D array
print(two_d_array) #printing 2D array
```

```
Output >>> [[1 2 3]
            [4 5 6]
            [7 8 9]]
            
```
            
In this way, you can create NumPy ndarray

Let’s going forward to learn more in python NumPy tutorial.

### How to check the type of ndarray?
The type() function give the type of data.

### Syntax: type(self, /, *args, **kwargs)

```
type(one_d_array) # give the type of data present in one_d_array variable
```
```
Output >>> numpy.ndarray
```
### How to check dimension of NumPy ndarray?
The ndim attribute help to find the dimension of any NumPy array.

#### Syntax: array_name.ndim

``` one_d_array.ndim # find the dimension of one_d_array
```
```
Output >>> 1

```
Value represent, one_d_array array has one dimension.

### How to check the size of the NumPy array?
The size attribute help to know, how many items present in a ndarray.

#### Syntax: array_name.size

```
one_d_array.size
```
```
Output >>> 4
```
value represent, total 4 item present in the one_d_array.

### How to check the shape of ndarray?
The shape attribute help to know the shape of NumPy ndarray. It gives output in the form of a tuple data type. Tuple represents the number of rows and columns. Ex: (rows, columns)

#### Syntax: array_name.shape

```
two_d_array.shape
```
```
Output >>> (3, 3)

```
The two_d_array has 3 rows and 3 columns.

### How to the data type of NumPy ndarray?
The dtype attribute help to know the data type of ndarray.

#### Syntax: array_name.dtype
```
one_d_array.dtype
```
```
dtype('int32')
```
As per the above output one-d_array contain integer type data. This data store in 32 bit format (4 byte).

Up to here you can create and know about NumPy ndarray in python NumPy tutorial. Let’s know more something interesting.

### Create metrics using python NumPy functions 
Ones metrics use NumPy ones() function.

#### Syntax: np.ones(shape, dtype=None, order=‘C’)

```
np.ones((3,3), dtype = int)
```
```
Output >>>
array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]])
```

### Zeros metrics use NumPy zeros() function.

#### Syntax: np.zeros(shape, dtype=None, order=‘C’)

```
np.zeros((3, 3), dtype = int)
```
```
Output >>>
array([[0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]])
```
### Empty metrics use NumPy empty() function.

#### Syntax: np.empty(shape, dtype=None, order=‘C’)

```
np.empty((2,4))
```
```
Output >>>
array([[0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
       [0.00000000e+000, 6.42285340e-321, 8.70018274e-313,6.95271921e-310]])
```
By default NumPy empty() function give float64 bit random value. According to your requirement change dtype.

### Create NumPy 1D array using arange() function

#### Syntax: np.arange([start,] stop[, step,], dtype=None)

```
arr = np.arange(1,13)
print(arr)
```
```
Output >>> [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
```
### Create NumPy 1D array using linspace() function

Return evenly spaced numbers over a specified interval.

#### Syntax: np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0,)

```
np.linspace(1,5,4)
```
```
Output >>> array([1.        , 2.33333333, 3.66666667, 5.        ])
```

### Convert 1D array to multidimensional array using reshape() function

#### Syntax: np.reshape(a, newshape, order=‘C’)

```
arr_reshape = np.reshape(arr, (3,4))
print(arr_reshape)
```
```
Output >>> 
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
```
### Convert multidimensional array in one dimensional

To convert multidimensional array into 1D use ravel() or flatten() function.

#### Syntax: np.ravel(array_name, order=‘C’)  or  array_name.ravel(order=‘C’)

```
array_name.flatten(order=‘C’)
arr_reshape.flatten()
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
```

### Transpose metrics

#### Syntax: np.transpose(array_name, axes=None)  or

``` 
array_name.T
```
```
Output >>>
array([[ 1,  5,  9],
       [ 2,  6, 10],
       [ 3,  7, 11],
       [ 4,  8, 12]])
```
###Conclusion
I hope, you learn each and every topic of python NumPy tutorial. This all topics important to do the project on machine learning and data science. Apart from the above-explained NumPy methods and operators, you can learn from numpy.org. This is an official website of python NumPy library. If you have fined any mistake in this tutorial of suggestions mention in the comment box. 
