'''This script is a toy machine learning script that demonstrates the 
use of various machine learning algorithms.'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def linear_regression(display=True):
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    reg = LinearRegression().fit(X_train, y_train)
    predictions = reg.predict(X_test)

    if display:
        plt.scatter(y_test, predictions)
        x = np.linspace(np.min(y_test), np.max(y_test), 100)
        y = x
        plt.plot(x, y, '-r')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.title('Linear Regression')
        plt.show()


def logistic_regression(display=True):
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    clf = LogisticRegression(max_iter=5000).fit(X_train, y_train)
    predictions = clf.predict(X_test)

    if display:
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        conf_matrix = confusion_matrix(y_test, predictions)

        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

        # ROC Curve
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_test, predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        # Feature Importance
        coef = clf.coef_[0]
        plt.barh(range(len(coef)), coef)
        plt.yticks(range(len(coef)), data.feature_names)
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature Names')
        plt.title('Feature Importances')
        plt.show()

        # Classification Report
        from sklearn.metrics import classification_report

        print(classification_report(y_test, predictions))


def decision_tree(display=True):
    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    tree = DecisionTreeClassifier().fit(X_train, y_train)
    predictions = tree.predict(X_test)

    if display:
        pass


def random_forrest(display=True):
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_digits

    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    forest = RandomForestClassifier().fit(X_train, y_train)
    predictions = forest.predict(X_test)


def support_vector_machine(display=True):
    # Support Vector Machine
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits

    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    svm = SVC().fit(X_train, y_train)
    predictions = svm.predict(X_test)


def k_nearest_neighbors(display=True):
    # K-Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_iris

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    predictions = knn.predict(X_test)


def naive_bayes(display=True):
    # Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    from sklearn.datasets import load_iris

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    nb = GaussianNB().fit(X_train, y_train)
    predictions = nb.predict(X_test)


def gradient_boosting_machine(display=True):
    # Gradient Boosting Machine
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import load_digits

    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    gbm = GradientBoostingClassifier().fit(X_train, y_train)
    predictions = gbm.predict(X_test)

if __name__ == '__main__':
    linear_regression()
    raise SystemExit
    logistic_regression(display=False)
    decision_tree(display=False)
    random_forrest(display=False)
    support_vector_machine(display=False)
    k_nearest_neighbors(display=False)
    naive_bayes(display=False)
    gradient_boosting_machine(display=False)