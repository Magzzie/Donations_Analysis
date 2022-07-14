# Neural_Network_Charity_Analysis
Using a deep-learning neural network to analyze and classify the success of charitable donations. 


## Background
*Alphabet Soup* is a non-profit philanthropic foundation dedicated to helping organizations that protect the environment, improve people's well-being, and unify the world. <br>

In the past 20 years, *Alphabet Soup* has raised and donated over 10 billion dollars. This money has been used to invest in life-saving technologies and organize reforestation groups worldwide. <br>

Unfortunately, not every donation the company makes is impactful. In some cases, an organization will take the money and disappear. As a result, *Alphabet Soup* president Andy Glad wishes to implement a new method to predict which organizations are worth donating to and which are too high risk. <br>
He is interested in a mathematical data-driven solution that can do this accurately. <br>

For that purpose, the company's senior data scientist, who is in charge of data collection and analysis for the entire organization, has requested an analysis of the impact of each donation and a new method to vet potential recipients. This analysis will help ensure that the foundation's money is used effectively. <br>

*Alphabet Soup*'s business team has provided a CSV data file that contains more than 34,000 organizations that have received funding from Alphabet Soup over the years. <br>
Within this dataset are several columns that capture metadata about each organization and can be used as features to train the predictive model. <br>

### Purpose
Analyzing previous donation records and vetting potential recipient organizations is too complex for regular statistical and machine learning models. <br>
Instead, we will design and train a deep learning neural network. This model will evaluate all types of input data and produce a clear decision-making result. <br>
Specifically, we will create a binary classifier capable of predicting whether applicants will be successful if funded by Alphabet Soup.

## Objectives
1. Preprocessing Data for a Neural Network Model.
2. Compile, Train, and Evaluate the Model
3. Optimize the Model.

## Resources 
- Data Sources: charity_data.csv, AlphabetSoupCharity.ipynb.
- Software & Framework: Python (3.7.13), Jupyter Notebook (6.4.11).
- Libraries & Packages: Pandas (1.3.5), matplotlib (3.5.1), Scikit-learn (1.0.2), tensorflow (2.3.0), keras-applications (1.0.8),  keras-preprocessing (1.1.2), 
- Online Tools: [Neural_Network_Charity_Analysis GitHub Repository](https://github.com/Magzzie/Neural_Network_Charity_Analysis)


## Methods & Code
1. Preprocessing Data for a Neural Network Model: <br>
Using Pandas library and Jupyter Notebook, we processed the dataset in order to compile, train, and evaluate the neural network model. 
    - We loaded the data file into a Pandas DataFrame and explored the included metadata.
    - The charity dataset contained (34,299) records organized in 12 columns about the previously funded organization by the Alphabet Soup foundation.
        |![Original Charity Applications DataFrame.](./Images/application_df.png)|
        |-|
    - We dropped the non-beneficial ID columns, 'EIN' and 'NAME.'
        |![Applications DataFrame After Removing Unnecessary ID Columns.](./Images/application_noid_df.png)|
        |-|
    - Then, we identified the categorical variables with more than ten unique values using the nunique() method and bucketed them according to their corresponding density plots. 
        - The first categorical variable bucketed was **application types**, where the density plot showed that the most common unique values had more than 500 instances within the dataset. Based on that graph, we created an 'Other' bin to contain all application types with less than 500 instances.       
            |![Application Type Density Plot.](./Images/application_type_density_plot.png)|
            |-|
        - The second categorical variable was **classification**. Using the density plot of the classification column unique values count, we decided to bin all classification values with less than 1,800 instances to an 'Other' class.         
            |![Classification Density Plot](./Images/classification_density_plot.png)|
            |-|        
    - Next, we used Scikit-learn's OneHotEncoder module to encode the categorical variables in the dataset and created a separate DataFrame of the encoded columns.          
        |![Encoded Columns DataFrame.](./Images/encode_df.png)|
        |-|
    - After encoding, we merged the encoded columns' DataFrame with the original application's DataFrame and dropped the unencoded categorical columns.    
    - Next, we needed to standardize our numerical variables using Scikit-Learn's StandardScaler class. However, we must split our data into the training and testing sets before standardization to not incorporate the testing values into the scale. Testing values are only for the evaluation of the model.
        - We defined our target column as 'IS_SUCCESSFUL' since it represents the outcome of funding a particular organization, which would be the prediction's goal. 
        - The input features influencing the neural network to predict the outcome were all encoded columns except for the predefined target column.
        - We used the train_test_split model from the Scikit-learn library to split our dataset into training and testing according to the default setting of 75%/25%, respectively. 
        - Lastly, we instantiated the StandardScaler model, trained it, and transformed the training and testing features separately.  



## Results 

- The charity_data.csv file initially contained (34,299) records of data in 12 columns. 
- The metadata of each organization included in the charity dataset were:
    - **EIN** and **NAME** —Identification columns
    - **APPLICATION_TYPE** —Alphabet Soup application type
    - **AFFILIATION** —Affiliated sector of industry
    - **CLASSIFICATION** —Government organization classification
    - **USE_CASE** —Use case for funding
    - **ORGANIZATION** —Organization type
    - **STATUS** —Active status
    - **INCOME_AMT** —Income classification
    - **SPECIAL_CONSIDERATIONS** —Special consideration for application
    - **ASK_AMT** —Funding amount requested
    - **IS_SUCCESSFUL** —Was the money used effectively
- We started the preprocessing phase of the analysis by dropping variables that would not add relative information to the prediction model. These were ID variables (EIN, NAME).
- After removing uninformative ID columns, bucketing values of categorical variables, and encoding all non-numerical columns in the dataset, we ended up with 44 columns for the neural network model. 
        |![Final Encoded Applications DataFrame.](./Images/application_new_df.png)|
        |-|
- We identified the input features that will influence the prediction of successful donations: application types, industry affiliation, government organization classification, the use case for funding, status, income classifications, asking amount, and any special considerations listed for an organization. 
- The prediction target is the funded organization's success in using the money effectively. 
- The resulting training and testing sections of the charity dataset were as follows: 
    - X_train shape: (25724, 43)
    - X_test shape: (8575, 43)
    - y_train shape: (25724,)
    - y_test shape: (8575,)
- The features of both training and testing sections of the encoded dataset were scaled using Scikit-learn's StandardScler so that all numerical values have a mean of 0 and standard deviation of 1. 


## Conclusions


---
