# MNIST Dataset Recognition Code

Author: Saarthak Srivastava

[Python 3.8.5	| TensorFlow 2.3.0]

This is my first TensorFlow code. Hello TensorFlow World! :)

It takes training data and test input from the MNIST database and recognises the number which is present in the image that was used as test input. L1 distance (Manhattan distance or Snake distance) method is used to find the K-nearest neighbours (KNN) to make predictions. The results are upto 99% accurate.

The  **MNIST database**  (Modified National Institute of Standards and Technology database) is a large database  of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.

![MNIST sample images](https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/220px-MnistExamples.png)

Samples from MNIST dataset

The MNIST database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset. The original creators of the database keep a list of some of the methods tested on it. In their original paper, they use a  support-vector machine  to get an error rate of 0.8%. An extended dataset similar to MNIST called EMNIST has been published in 2017, which contains 240,000 training images, and 40,000 testing images of handwritten digits and characters.