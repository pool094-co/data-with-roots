import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def entrenar_knn_clasificador():
    
    df = pd.read_csv('logistic_dataset.csv')
    
    
    X = df.drop(columns=['desercion'])
    y = df['desercion']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    
    modelo = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    modelo.fit(X_train_scaled, y_train)
    
    return modelo, scaler