**Name:    VAMSI RAM**

**E-mail : vamsiramg@gmail.com**

This readme file includes the summary of my learnings from the resources provided in the Data Science Internship by Wow Labz on Conduira online platform.


# 1.What does a Data Scientist do?
- Understand the problem
- Process the information
- How to plan the process
- How to present the output solution
- Communicate the results in an effective way


**Data Collection --> Data Cleaning --> Exploratory Data Analysis --> 
Data Analyzing --> Model Building and evaluation --> Visualizing the results.**

### Significance of Artificial Intelligence and Machine learning in the past

Going through the learning resources regarding the history of Machine Learning states that attempts to make use of the power of a machine to compute complex calculations is not any new age concept. In fact, several decades back Machine Learning has been developing from 1950’s and it is advancing day by day.

It is quite interesting to see how deep learning which is a subset of machine learning catered to the needs of one of the leading global IT industries like Google, in improving its Google Translate app.

The second case study mentioned the limitations of A.I by contrasting it to the general intelligence claiming that it would be difficult for the machine to perform the tasks that a 1-year-old would do. These speculations led to the development of artificial neurons that could perform basic logical functions.
The cat paper experiment proved the potential of the neural network’s ability to analyze unlabeled data. When over millions of still images were shown, the network isolated a pattern itself and produced an output resembling that of a cat.

This proved phenomenal advancement in the research of the Google Brain team. It is then said that, appropriately programmed computers, with the right inputs would in turn provide right outputs and have a mind in exactly the sense that human beings have minds.

# 2.DATA WRANGLING

- Collecting the required dataset is the first and most important step in the Data Science workflow.
- Data collected from various sources would not be of the same format. In fact there might be missing values.

If we take Python, **Pandas** is the widely used and most popular library for the case of data wrangling. It has the feature of defining the data as form of Data Frame that are structured tables through which we can query in the form of rows and columns.

Based on the suggested tutorials the basic commands that can be utilized while understanding more about the dataset and data wrangling are,

Consider that the data file is extracted and stored in the name of the dataframe called **data**,

**data.info()**  - *It gives the information about the dataset.*

**data.describe()**  - *It gives brief statistical description of the data.*

**data.head()**   - *It displays the first few entries in the dataframe.*

**data.tail()**   - *It displays the last few entries in the dataframe.*

**data.dtypes**   - *It describes the datatypes of the data.*


 - **Dropping null and duplicate values:**

Now in order to know the number of null values are in the datframe,
Consider the name of the dataframe as data, then the Python command is **print(data.isnull().sum())**

Now we can either remove these null values, or either replace these null values with mean or quartiles. It is upto the data scientist whether to replace or remove.

In order to remove the null values, **data=data.dropna()**

Similarly in order to remove the duplicate values, **data=data.drop_duplicates()**


- **Filtering and Grouping Data:**

We can filter the dataframe according to our needs with the help of column names.Filtering the data helps us to understand about different sectors in the data and we can draw conclusions based on our findings.
Now in order to understand whether there is any correlation between different aspects in the data, we can use pandas GroupBy function. This function returns us a GroupBy object which in turn has several methods for analyzing.

- **Time Series Data:**

It is one of the most important data forms when working with the financial data, weather patterns. We can inspect at certain periods of time. We can resample the data from seconds to minutes using Python Pandas datetime library.

- **Exporting Data:**

Once we are done with cleaning the data and analyzing the data, we need to draw some important insights correlating to our findings in order to proceed to the next step. Then we need to export the data to suitable file format which can be best understood and visualized in the best way possible.

# 3.MACHINE LEARNING
Machine Learning is a subset of Artificial Intelligence. When it comes to implementing ML algorithms in data science, there are broadly two types of categories:

## 3.1 Supervised Machine Learning
The data given to the machine contains labeled examples of different cases. The algorithm learns these different cases. Generally, if you want to predict a case of a success, you give the labeled data containing both the cases of success and failure and the machine learns and distinguish both the categories. 

### Types of Supervised Machine Learning:

**Classification problem:**
- To predict discrete values. 
- Classify Yes or No.
- Predicting categories.

**Regression problem:**
- To predict continuous value from range of numbers.
- Predicting numbers.

The main difference between classification and regression is the form of the output variable. If the output variable is discrete, it is classification, and if it is continuous, it is regression. You can think of discrete data as something that is counted, and continuous data as something that is measured. Discrete data refers to things like movie genres, categories of mail (spam/not spam) and so on. Continuous data refers to data like the height of a person, stock price, price of a house, and so on.

### 3.1.1 Linear Predictions:
These algorithms represent data as a multidimensional line. In the case of regression, the line will define the structure of the data, whereas in the case of classification, the line would be like the boundary between the two categories.

**Linear Regression:**
- In this algorithm we use the existing data and predict a required numerical value. We should ensure that the line should fit the data correctly before predicting.
- The best fit line is the line that is least off from the real data. We can define this with the help of the cost function.
- We can also make use of the Gradient Descent to find the best fit line for the data.
- The best line is that which minimizes the cost, that is the distance from each point of data to the line.

**Logistic Regression:**
- This algorithm predicts the category of the data. Similar to linear regression, we draw a line that best fits the existing data. This line is called as Decision Boundary.
- However, in case of logistic regression, the predicted variable is a categorical variable. In case of linear, we predicted a number but in case of logistic we predict the probability of the prediction to be in a certain category.
- The cost function depends on the how far off are the predictions from actual data.
- The sigmoid function is used to constrain the predicted output between 0 and 1.
- If there are multiple classes, we make use of one vs all classification for each.

We need to use Linear Models when the data is linearly separatable and also when the number of samples is low when compared to the number of features.

### 3.1.2 Non Linear Predictions:
In some cases the datasets would be much more complex where there might be more number of features and large number of samples to predict and there would not be any linear relation between the features.In such cases, non linear prediction algorithms can be used.

**Random Forests:**
- This uses a set of predictors and combines their results in order to obtain a final prediction.This method of combining several set of predictors provides more accurate and robust prediction models.
- This algorithm makes use of **Decision Tree** which considers one variable at a time and typically computes a binary output whether yes or no.
- Random forest basically combines the output of randomly created multiple decision trees to generate the final output.
- These also come with built-in cross validation which in turn reduces the overfitting case.

**Neural Networks:**

Some of the problems can not be solved with just a simple linear classifier.We need to use **multiple nodes** working together to solve such problems.
 - These contain an input node followed by hidden intermediate nodes which are the main parts of processing which at last given an output.
 - There can be several hidden layers and multiple number of nodes processing.
 - Once a sufficient number of input examples are given to the model,it gets trained by these examples, it can predict unseen inputs and give accurate outputs.
 - The more types of examples given to the model as input,the more accurate the output would be.


## 3.2 Unsupervised Machine Learning
In this case raw data collected is given directly to the machine without any labels to the data. The algorithm needs to discover the correlation or structure between the data.
Clustering is a method in which we aim to group subsets of the entities with one another based on some notion of similarity.

### 3.2.1 K- means Clustering:
- It splits the observations into n groups of equal variance.
- We need to choose the number of clusters and define it to the clustering algorithm and then the algorithm assigns the observations to the similar clusters with a centroid to each of the cluster.
- It iterates over the data and updates the observations and recalculates the cluster centre.
- We can optimize this clustering algorithm by monitoring several metrics such as graphing the variance and cross validation. 
- It scales large data points and is efficient until we choose the correct number of clusters required.

### 3.2.2 Hierarchical Clustering:
- This technique takes away the problem of having the user to pre-define the number of clusters.
- It assigns each of the data points seperate clusters and based on the similarity of these clusters,it combines the most similar clusters together.
- This process of combining similar clusters continues until only a single cluster is left.

There are two basic methods to generate Hierarchical Clustering which are:

- **Agglomerative:**
Initially it considers every data point as an individual cluster at every step and merge the nearest pairs of the cluster. At every iteration,the clusters merge with different clusters until one cluster is formed.

- **Divisive:**
It is the opposite of the agglomerative. In this we consider all of the data points as a single cluster.In every iteration we seperate the data points from the clusters which are not comparable and at the end we are left with n number of clusters.


### Conclusion:

In this summary firstly I have given a brief introduction of what a Data Scientist does and explained the workflow of a data science project.Next I have focussed on the first and most important step in the data science workflow which is Data Wrangling.I discussed some of the baisc commands in Python in order to understand the type of data which you are dealing with and how to clean and filter the data based on the problem statement.
Then I gave a brief description of Machine Learning and its significance in developing the prediction models.I have also discussed various types of prediction categories such as Classificaion and Regression and also mentioned the important and efficient algorithms that need to be implemented for different problem cases.Also different types of linear and non linear prediction models and algorithms are discussed.

I am greatly obliged and thank Wow Labz for providing me the valuable learning resources in order to kickstart my internship.I also take this oppurtunity to thank Conduira online for providing me a platform in order to learn and develop my skills.







