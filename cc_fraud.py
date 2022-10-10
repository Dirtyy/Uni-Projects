import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def convert_csv(csv):
    df = pd.read_csv(csv) #Read Dataset into memory
    df = df.dropna() #Get rid of rows with missing Values
    df = df.drop(['Amount', 'Time'], axis = 1)
    y = df['Class'] #Set the Classifier
    x = df.drop(['Class'], axis = 1) #Set the features  
    return x, y

x, y = convert_csv('University-Projects/creditcard.csv')
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42,test_size=.25) #set training/test data samples



#Deciscion Tree Algorithm
def decision_tree_alg(x_train,x_test,y_train,y_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    y_pred_dtc = dtc.predict(x_test)

    dtc_acc = accuracy_score(y_test, y_pred_dtc)
    dtc_conf = confusion_matrix(y_test, y_pred_dtc)
    dtc_auc  = roc_auc_score(y_test, y_pred_dtc)

    print(f"Accuracy Score of Decision Tree is : \n{dtc_acc}")
    print(f"\nConfusion Matrix : \n{dtc_conf}")
    print(f"\nArea Under Curve : \n{dtc_auc}")

#Random Forest Algorithm
def random_forest_alg(x_train,x_test,y_train,y_test):
    rfc = RandomForestClassifier(n_estimators = 100) #Set number of random decision trees
    rfc.fit(x, y)
    y_pred_rfc = rfc.predict(x_test)
    rfc_acc = accuracy_score(y_test, y_pred_rfc)
    rfc_conf = confusion_matrix(y_test, y_pred_rfc)
    rfc_auc  = roc_auc_score(y_test, y_pred_rfc)


    print(f"\nAccuracy Score of Random Forest is : \n{rfc_acc}")
    print(f"\nConfusion Matrix : \n{rfc_conf}")
    print(f"\nArea Under Curve : \n{rfc_auc}")

#K Nearest Neighbour Algorithm
def knn_alg(x_train,x_test,y_train,y_test):
    knn = KNeighborsClassifier(n_neighbors = 5) #Set Number of neighbours
    knn.fit(x_train,y_train)
    y_pred_knn = knn.predict(x_test)

    acc_knn = accuracy_score(y_test, y_pred_knn)   
    conf_knn = confusion_matrix(y_test, y_pred_knn)
    auc_knn  = roc_auc_score(y_test, y_pred_knn)

    print(f"\nAccuracy Score of K Nearest Neighbour is : \n{acc_knn}")
    print(f"\nConfusion Matrix : \n{conf_knn}")
    print(f"\nArea Under Curve : \n{auc_knn}")
 

decision_tree_alg(x_train,x_test,y_train,y_test)
random_forest_alg(x_train,x_test,y_train,y_test)
knn_alg(x_train,x_test,y_train,y_test)

