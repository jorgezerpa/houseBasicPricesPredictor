import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

houses_file_path = './src/datasets/melb_data.csv'
houses_data = pd.read_csv(houses_file_path)

#print(houses_data.describe()) #statistical info of the houses
#print(houses_data.columns) #show data columns

houses_data = houses_data.dropna(axis=0) #drop missing values


##prediction
houses_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

y = houses_data.Price
X = houses_data[houses_features]

# print(X.describe())


# ML
houses_model = DecisionTreeRegressor(random_state=1)

fitted_model = houses_model.fit(X,y)


#prediction 
# print("making predictions for the next houses:")
# print(X.head())
# print(houses_model.predict(X.head()))


#Validations of data using MAE
# predictedHomePrices = houses_model.predict(X)
# print(mean_absolute_error(y,predictedHomePrices))


#Validations of data using MAE with data splited
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

splited_model = DecisionTreeRegressor()
splited_model.fit(train_X, train_y)
splited_prediction = splited_model.predict(val_X)

print(mean_absolute_error(val_y, splited_prediction))


