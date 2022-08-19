# Example-AI-Projects

Some of the CSV files could not be commited to this repo because they are so big, if you need them please message me.

## Flow of a Fraud Detection System

### Pre processeing

Data is imported. Before training, a process of splitting the labelled is performed. Most of the labelled data is used to train the model, while a small portion is used for testing. 
The `app.py` script defined a test dataset size of `test_size = 0.2`. Meaning 80% of the data is used to train the model. This an important detail to note for the models training.

### Training

In the `app.py` script, 5 models are trained for fraud detection. Decision Tree, K-Nearest Neighbors, Logistic Regression, SVM, Random Forest Tree. These models are very common and the models used are a key detail to be noted.

### Evaluation

After the model is trained, and before the model is deployed to production, it is evaluated using the 20% of the labelled dataset set aside for testing.

#### Confusion Matrix

The confusion matrix is one of the most important assets to evaluating how a model performed againt the test set. The `app.py` script generates this confusion matrix.

<img width="444" alt="image" src="https://user-images.githubusercontent.com/108581791/185659685-6fafdee2-59d8-4877-bc0d-72ed63f246be.png">


#### Other Evaluation Metrics

Other evaluation metrics are calucalted from the confusion matrix: Accuracy, precision, recall, f1 and many more. Some data scientists even define their own methods of evaluation.

This system calculates accuracy and f1 for each of the models and displays these metrics on the console.
