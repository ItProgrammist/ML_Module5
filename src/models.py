from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score

def train_random_forest(X_train, X_test, y_train, y_test):

    clf_rf = RandomForestClassifier()
    parameters = {'n_estimators': [10, 20, 30], 'max_depth': [3, 5, 7, 10], 'min_samples_leaf': [10, 20, 30, 40, 50], 'min_samples_split': [3, 5, 7, 10]}
    grid_search_cv_rf_clf = GridSearchCV(clf_rf, parameters, cv=5)
    grid_search_cv_rf_clf.fit(X_train, y_train)
    best_rf_clf = grid_search_cv_rf_clf.best_estimator_
    y_pred2 = best_rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred2)

    return best_rf_clf, accuracy
