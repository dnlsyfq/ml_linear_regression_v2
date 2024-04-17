# Univaritate Regression / Simple Linear Regression
* Predict dependent variable based on indepedent variable 


# Multivariate Regression
two or more independent variables to predict the value of the dependent variable.

## Data Preparation
1. Training set: the data used to fit the model
2. Test set: the data partitioned away at the very start of the experiment (to provide an unbiased evaluation of the model)
```
from sklearn.model_selection import train_test_split
 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
```

```
import codecademylib3_seaborn
import pandas as pd

# import train_test_split
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms','bathrooms','size_sqft','min_to_subway','floor','building_age_yrs','no_fee','has_roofdeck','has_washer_dryer',
'has_doorman','has_dishwasher','has_patio',
'has_gym']]

y = df['rent']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=6)

print('X Train:{}\nX Test:{}\nY Train: {}\nY Test: {}'.format(x_train.shape,x_test.shape,y_train.shape,y_test.shape))

```

```
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split


streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

# Add the code here:
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(x_train,y_train)
y_predict= mlr.predict(x_test)
print(y_test.shape)
print(y_predict.shape)
```

## Data Modeling

The .fit() method gives the model two variables that are useful to us:
```
.coef_, which contains the coefficients
.intercept_, which contains the intercept
```

Coefficients are most helpful in determining which independent variable carries more weight. For example, a coefficient of -1.345 will impact the rent more than a coefficient of 0.238, with the former impacting prices negatively and latter positively.

### Correlations

* used 14 features, so there are 14 coefficients
* In regression, the independent variables will either have a positive linear relationship to the dependent variable, a negative linear relationship, or no relationship. A negative linear relationship means that as X values increase, Y values will decrease. Similarly, a positive linear relationship means that as X values increase, Y values will also increase.

### Evaluate model accuracy

1. Residual Analysis @ R2 
The difference between the actual value y, and the predicted value ŷ is the residual e
```
e = y - ŷ
```

2. R2
* .score() method that returns the coefficient of determination R² of the prediction.
* R² is the percentage variation in y explained by all the x variables together.
```
1 - u/v

u , residual sum of squares:
((y - y_predict) ** 2).sum()

v, total sum of squares (TSS) // TSS tells you how much variation there is
((y - y.mean()) ** 2).sum()
```


For example, say we are trying to predict rent based on the size_sqft and the bedrooms in the apartment and the R² for our model is 0.72 — that means that all the x variables (square feet and number of bedrooms) together explain 72% variation in y (rent).
Now let’s say we add another x variable, building’s age, to our model. By adding this third relevant x variable, the R² is expected to go up. Let say the new R² is 0.95. This means that square feet, number of bedrooms and age of the building together explain 95% of the variation in the rent.
The best possible R² is 1.00 (and it can be negative because the model can be arbitrarily worse). Usually, a R² of 0.70 is considered good.
```
print(mlr.score(x_test,y_predict))
```
