# NaiveBayes-Classifier
The Naive Bayes classifier is a probabilistic algorithm used for classification tasks. It is based on Bayes' theorem, which states that the probability of a hypothesis (in this case, the class of an input) is proportional to the probability of the evidence (the input) given that hypothesis.

# How it works
The Naive Bayes classifier assumes that the features of the input are independent given the class. This means that the presence or absence of one feature does not affect the probability of the presence or absence of another feature.

# How to run code
The user need to have training image and testing image for the code to run. Both images should have bands similar to Worldview 2 satellite, and stacked with 3 bands of (R,G,B) of ground truth.
To run the code just type python Naivebayes.py in command prompt and user will be good to go.

# Types of Naive Bayes Classifiers
There are several types of Naive Bayes classifiers, including:
•	Gaussian Naive Bayes: assumes that the input features are normally distributed.
•	Multinomial Naive Bayes: used for discrete count data such as word counts.
•	Bernoulli Naive Bayes: used for binary data such as presence or absence of a feature.

# Advantages and disadvantages
Advantages of the Naive Bayes classifier include:
•	Fast and easy to implement.
•	Requires only a small amount of training data.
•	Can handle high-dimensional data.
•	Performs well on certain types of tasks such as text classification.

Disadvantages of the Naive Bayes classifier include:
•	Strong assumption of independence may not hold in real-world data.
•	May not perform well on tasks where feature interactions are important.
•	Can be sensitive to irrelevant features.

# Conclusion
The Naive Bayes classifier is a simple yet effective algorithm for classification tasks. It is especially useful in situations where data is limited or high-dimensional. However, its strong assumption of independence may not hold in all cases, and it may not perform well on tasks where feature interactions are important.

