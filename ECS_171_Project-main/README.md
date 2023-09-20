# ECS 171 Final Project

# Introduction

- Blood is a crucial resource for life-saving transfusions. Unfortunately, it is often in short supply and blood banks regularly face shortages due to the volunteer-based sourcing of an already scarce resource
- Using a dataset describing blood donors, we will train a model to predict whether a given patient will donate blood or not based on information like how many times they have previously donated blood and whether they have donated blood recently or not.
- Such a model could help blood banks and hospitals predict starvation periods and ration resources accordingly.

# Figures
Our Data Table:

![Data Table](project_images/Data%20Table.png)

Our Data Pairplot:

![Data Pairplot](project_images/Data%20Pairplot.png)

# Methods

## Data Preprocessing

1. Read in data and drop any NA values. We did not find any, so we did not have to drop any.

    ```
    data.isna().any()
    =====================================================
      Recency (months)                              False
      Frequency (times)                             False
      Monetary (c.c. blood)                         False
      Time (months)                                 False
      whether he/she donated blood in March 2007    False
      dtype: bool
    ```
2. Rename features to have simpler names. (For example, "whether he/she donated blood in March 2007" -> "donated")
    ```
    data.columns = ["recency", "frequency", "monetary", "time", "donated"]
    ```
3. Drop any unnecessary features. We found that the features "frequency" and monetary provide identical information to our model, so we only require one. We chose to drop monetary.
    ```
    data.drop(columns="monetary", inplace=True)
    ```
4. Partition into training and test data. We decided to use a train/test split of 80/20.
    ```
    train, test = train_test_split(data, test_size = .2, random_state = 33)
    X_train, y_train = train.drop(columns= "donated"), train["donated"]
    ```
5. Scale the data with MinMaxScaler().
    ```
    scaler = MinMaxScaler()
    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    ```

## Data Exploration

1. Generate pairplot using Seaborn
    ```
    sns.pairplot(data)
    ```
2. Generate correlation matrix using Seaborn
    ```
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, vmin=-1, vmax=1, center=0, annot=True, cmap = 'RdBu', fmt='.2g'
    ```
    ![image](https://user-images.githubusercontent.com/38890728/205831994-719d8529-c60b-4d3e-9bf6-5a4ab063cbac.png)

## Model 1

For our first model, we decided to use a Support Vector Classifier (SVC) with linear kernel.
```
model = LinearSVC(dual=False)
model.fit(X_train, y_train)
```

## Model 2

For our second model, we decided on using a Support Vector Machine (SVM) with RBF kernel.
```
svm = SVC(kernel = 'rbf')
svm.fit(X_train,y_train)
```

# Results

## Model 1

Classification Report for train data:

![Classification Report Model 1 - Train](project_images/Classification%20Report%20Model%201%20-%20Train.png)

Classifcation Report for test data:

![Classification Report Model 1 - Test](project_images/Classification%20Report%20Model%201%20-%20Test.png)

## Model 2

Classification Report for train data:

![Classification Report Model 2 - Train](project_images/Classification%20Report%20Model%202%20-%20Train.png)

Classification Report for test data:

![Classification Report Model 2 - Test](project_images/Classification%20Report%20Model%202%20-%20Test.png)

# Discussion

## Preprocessing

- Dataset feature names were renamed to more ideal names for simplicity.
- After performing an exploratory data analysis, we found that features frequency and monetary provided the same information, with a correlation score of 1.
- This was due to the fact that the same amount of blood is donated each visit. So the number of visits and the total blood donated are directly proportional to each other.
- We decided to drop monetary to avoid redundancy.
- Data was normalized using MinMaxScaler to preserve the underlying distribution of the data.

## Model 1

We decided to use a Linear SVM as our first model because our dataset represents a classification problem, and we have a relatively small sample size (<1000). Linear SVMs (and SVMs in general) tend to work better on smaller datasets. Judging from our pairplots and other figures, we see that our dataset is not exactly linearly separable, so we expect our first Linear SVM model to not perform great, however a Linear SVM is a great start to keep building more complex models upon, judging from our Classification Reports and Results. 

When comparing the Classfication Report of our training data vs, testing data, we see that all our measures of error (precision, recall, f1-score, support, and accuracy) are all within <2% of eachother. This tells us that our graph is definitely NOT overfitted, because an overfitted model would mean that our testing data would perform much more poorly than than our training data. However, we can see that our accuracy and precision are both below 80%. This means that our model incorrectly predicted the outcome ~1/5 of the time, and we also classified around the same amount as false positives. 

So, from analyzing all the metrics of our Classification Reports, we can safely predict that our model IS underfitted. Our accuracy and precision both scored poorly (below 80%), however we see that the Classification Reports between our training and testing data perform very similarily (each error metric being 1-2% off of each other). 

There are a multitude of reasons why the accuracy of our Linear SVM was so low. The first explanation is our dataset containing variables that have low correlation to our predicted results. Also, our dataset contains a low amount of features (only 3 were used to predict our classification for "donated"!). A low feature count is also another sign of underfitting. We could potentially do Feature Engineering to extract more information from our features using feature creation. Adding more variables would increase the complexity of our model, and potentially solve our problem of underfitting. 

Our next steps to improve our current underfitting model is to add more complexity to our model, either through Feature Engineering, changing the kernel of our SVM, or using a completely different model. In our first model, we only used one Linear SVC, in the future we plan to use neural networks for classification as well as an RBF SVC. 

## Model 2

When comparing the Classification Report for our training and testing data, we get different results than we did from Model 1. Looking at our measures of error, we can clearly see that our values have a greater difference for precision, recall, f1-score, and support, than we had for Model 1. What is apparent about this, though, is that our precision value did go up from our previous model, showing that there was some kind of improvement in our second model. Moreover, our accuracy value did go up by .01%, meaning that our second model did perform slightly better in terms of accuracy than Model 1 did. 

It appears that our model still is underfitted. Our accuracy and precision values both went up from the previous model, leading us to believe that our second model did, in fact, do a better job of correctly predicting whether a person donated blood or not. It's also important to note that our Classification Reports for our training and testing data in Model 2 were still relatively similar.

# Conclusion

- Blood donation can be reliably predicted based on donor information.
- Lack of data was our greatest limitation, as we only had data from one year. Training the model on data spanning multiple years would likely significantly increase performance.
- Further research would consist of fitting and evaluating different classification models, as well as obtaining a more robust data set.

# Collaboration

- Ibrahim Rizk 917576313: Worked on Model #1, worked on data preprocessing, data exploration, model fitting, worked on the write up
- Dylan Sun 919962938: Worked on Model #1, Classification Report for Model #1, Helped With Preprocessing Dataset, worked on discussion
- Michael Kim 918619680: Worked on Model #2, worked on the write up (specifically methods section), did some scheduling for the team
- Mark Hong 917491553: Worked on Model #2, worked on the write up (specifically figures, discussion, results, and methods)
- Abhijt Singh 918269124
- Harkamal Padda 918729538: Created discord server where we were able to collaborate and exhange information. Looked over the write-ups specifically for     model 2.
