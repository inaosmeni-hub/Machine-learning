# Titanic Survival Prediction — Neural Networks

## Abstract

This notebook implements and compares two neural network architectures for binary classification on the Titanic dataset — a collection of 891 passenger records with demographic and ticket information used to predict survival. A Basic Neural Network with a single hidden layer serves as a baseline, while a Deep Neural Network with three hidden layers, L2 regularisation, and Dropout is trained with early stopping. Both models are evaluated using accuracy, confusion matrices, ROC curves, and AUC. A hyperparameter search across four configurations examines the effect of learning rate, batch size, L2 penalty, and dropout rate on generalisation performance.

---

## Written Observations

**1. The Small Dataset Size Makes Overfitting the Central Challenge**
With only 891 rows, the Titanic dataset sits at the lower end of what neural networks typically require to generalise well. The Basic Neural Network is deliberately kept shallow — a single hidden layer of 16 units — to serve as a stable, low-variance baseline. The Deep Neural Network compensates for its greater capacity with L2 regularisation applied to the first two hidden layers and Dropout at a rate of 0.3, reducing the risk that the model memorises training patterns rather than learning transferable structure. This design choice reflects a core principle in applied deep learning: model complexity must be matched to dataset size, and regularisation is not optional on small tabular datasets.

**2. Early Stopping is Essential to Prevent Overfitting in the Deep Model**
The Deep Neural Network is trained with early stopping monitoring validation loss, with a patience of 10 epochs and `restore_best_weights=True`. Without this callback, continued training beyond the validation loss minimum would increase the gap between training and validation performance — a textbook overfitting pattern visible in the training history plots. By restoring the weights from the best checkpoint, the model that reaches the test set is the one that generalised best during training, not the one that simply trained the longest. The number of epochs actually run varies across hyperparameter configurations, confirming that early stopping is doing meaningful work rather than being a passive addition.

**3. Learning Rate of 0.001 with Adam Consistently Outperforms Alternatives**
The hyperparameter experiments test four learning rate values: 0.01, 0.001, 0.0001, and 0.005. A learning rate of 0.001 with Adam consistently achieves the best balance between convergence speed and generalisation. Higher rates (0.01 and 0.005) introduce instability in the validation loss curve — the loss decreases rapidly at first but oscillates or diverges before settling, preventing the model from finding a stable minimum. A very low rate (0.0001) converges too slowly within the epoch budget and frequently triggers early stopping before the model reaches a competitive solution. This confirms the widely observed behaviour that Adam with lr=0.001 is a robust default for small-to-medium classification tasks.

**4. AUC Is a More Informative Metric Than Accuracy for This Dataset**
The Titanic survival rate is approximately 38%, creating a moderate class imbalance. A naive classifier that always predicts "not survived" would achieve around 62% accuracy without learning anything meaningful. Accuracy alone therefore gives an inflated impression of model quality. The ROC curve and AUC metric measure the model's ability to correctly rank survivors above non-survivors across all possible classification thresholds, regardless of where the 0.5 decision boundary is set. Comparing AUC values between the Basic NN and Deep NN provides a threshold-independent view of discriminative power and reveals differences in model quality that accuracy alone would obscure.

**5. Model Depth Alone Does Not Guarantee Better Performance on Small Tabular Data**
The performance gap between the Basic NN and the Deep NN is narrower than one might expect given the difference in architecture. The Deep NN has substantially more parameters — 128 units in the first hidden layer alone compared to 16 in the basic model — yet the accuracy and AUC improvement is modest. This reflects a well-documented pattern: on small structured datasets, the benefit of additional layers is limited, and the regularisation overhead required to prevent overfitting can consume much of the capacity advantage. In practice, on datasets of this size, well-tuned shallow models often match or exceed deep ones, and the hyperparameter choices (learning rate, dropout, L2 strength) matter more than the number of hidden layers.

---

*Keywords: neural networks, binary classification, Titanic, dropout, L2 regularisation, early stopping, ROC-AUC, hyperparameter tuning, overfitting, deep learning*
