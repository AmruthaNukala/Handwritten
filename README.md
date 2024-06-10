# Handwritten Digit Recognition using Machine Learning and Deep Learning

This project demonstrates the recognition of handwritten digits using various machine learning and deep learning algorithms.

## Published Paper

For more detailed information, you can refer to our published paper: [IJARCET-VOL-6-ISSUE-7-990-997](http://ijarcet.org/wp-content/uploads/IJARCET-VOL-6-ISSUE-7-990-997.pdf)

## Requirements

Ensure you have the following installed:
- Python 3.5+
- Scikit-Learn (latest version)
- Numpy (with mkl for Windows)
- Matplotlib

## Usage

### 1. Download the MNIST Dataset

Download the MNIST dataset files using the following commands:

```bash
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

Alternatively, you can download the [dataset from here](https://github.com/anujdutt9/Handwritten-Digit-Recognition-using-Deep-Learning/blob/master/dataset.zip), unzip the files, and place them in the appropriate folders.

### 2. Organize the Dataset Files

Unzip and place the files in the dataset folder inside the `MNIST_Dataset_Loader` folder for each ML algorithm folder. The structure should be as follows:

```
KNN
|_ MNIST_Dataset_Loader
   |_ dataset
      |_ train-images-idx3-ubyte
      |_ train-labels-idx1-ubyte
      |_ t10k-images-idx3-ubyte
      |_ t10k-labels-idx1-ubyte
```

Repeat this for the SVM and RFC folders.

### 3. Run the Code

Navigate to the desired algorithm directory and run the corresponding Python file. For example, to run the K Nearest Neighbors algorithm:

```bash
cd 1. K Nearest Neighbors/
python knn.py
```

or

```bash
python3 knn.py
```

This will execute the code and log output to the `summary.log` file. If you want to see the output on the command prompt, comment out lines 16, 17, 18, 106, and 107 in the `knn.py` file.

You can also use an IDE like PyCharm to run the Python files.

### 4. Run the CNN Code

The CNN code automatically downloads the MNIST dataset. Simply run the following command:

```bash
python CNN_MNIST.py
```

or

```bash
python3 CNN_MNIST.py
```

### 5. Save CNN Model Weights

To save the model weights after training, use the following command:

```bash
python CNN_MNIST.py --save_model 1 --save_weights cnn_weights.hdf5
```

or

```bash
python3 CNN_MNIST.py --save_model 1 --save_weights cnn_weights.hdf5
```

### 6. Load Saved Model Weights

To load previously saved model weights and skip training, use:

```bash
python CNN_MNIST.py --load_model 1 --save_weights cnn_weights.hdf5
```

or

```bash
python3 CNN_MNIST.py --load_model 1 --save_weights cnn_weights.hdf5
```

## Accuracy

### Machine Learning Algorithms:
- **K Nearest Neighbors**: 96.67%
- **SVM**: 97.91%
- **Random Forest Classifier**: 96.82%

### Deep Learning Algorithms:
- **Three Layer Convolutional Neural Network using TensorFlow**: 99.70%
- **Three Layer Convolutional Neural Network using Keras and Theano**: 98.75%

**Note**: All code is written in Python 3.5 and executed on an Intel Xeon Processor / AWS EC2 Server.

## Video Demonstration

Watch the video demonstration here: [YouTube Video](https://www.youtube.com/watch?v=7kpYpmw5FfE)

## Test Images Classification Output

![Output](Outputs/output.png "Classification Output")

---

Feel free to contribute to this project by forking the repository and submitting pull requests. For major changes, please open an issue first to discuss what you would like to change. If you find this project helpful, please give it a star!