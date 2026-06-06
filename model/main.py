import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

def create_model(data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model (e.g., Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data():

    data = pd.read_csv('data/data.csv')

    # print(data.columns)

    # data = data.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')

    return data


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as g:
        pickle.dump(scaler, g)

    

    
if __name__ == "__main__":
    main()