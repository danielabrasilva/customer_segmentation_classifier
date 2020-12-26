# Customer Segmentation and Classifier


### Instructions
1. Run the `customer_segmentation.ipynb`, or
2. Run `process_data.py` followed by `clustering.py` and `classifier.py`.
3. Access the blog post https://danielabrsil.medium.com/a-customer-segmentation-report-for-arvato-financial-services-83d459356b2a to read the documentation of this work.

### Project Motivation

The purpose of this project is to identify possible new customers for a German mail-order company, Bertelsmann Arvato. The identification of these customers is done using a supervised machine learning model, which takes into account the general demographic characteristics of the German population, as well as the results of the segmentation of these individuals through a clustering model.

### Project Overview
This project is divided into two stages:

I. Customer segmentation: At this stage, we have the azdias.csv dataset for the general population, and also the customers.csv dataset for the company’s customers. Both sets contain columns referring to the demographic characteristics of individuals. The idea here is to bring these sets together and use an unsupervised learning model to segment these individuals into different groups and thereby identify the groups that have the most customer adherence. For this, I tested the K-means, Aglomerative Clustering, DBSCAN and OPTICS algorithms, all from the scikit-learn library. As the execution time of K-means was much shorter than the others, I opted for this solution, because that way it would be easier to find the ideal cluster number.

The population was segmented into 6 groups, from which it was noticed that individuals who were already Arvato customers were mostly in groups 1 and 5. The result of clustering is shown in the table below, where the green bar indicates an increase in customer concentration and the red bar indicates a decrease in concentration for each cluster.

![alt text](https://miro.medium.com/max/408/1*_WOIntk74BD0JvELf_deyw.png)

II. Classification of the population into client or non-client: In this step, we use a data set to train a supervised learning model that classifies whether an individual will be a possible client or not. Before training, the set is treated and segmented into groups according to the clustering model from the previous step. Gradient boosted trees, Linear Support Vector Machine, and Logistic Regression algorithms were tested and the one that showed the best performance according to the Area Under Curve ROC (AUC ROC) metric was Gradient Boosted trees as we can see below.

- No Skill: ROC AUC=0.500
- XGB: ROC AUC=0.721
- SVM: ROC AUC=0.604
- Logreg: ROC AUC=0.660

![alt text](https://miro.medium.com/max/432/1*w2gm5vb1UDWOSnfwqEESWQ.png)


### Libraries
NumPy, pandas, scikit-learn, xgboost, seaborn and matplotlib

### File Descriptions

There are three project components:

 1. A Python script `process_data.py`, which clean the data.
 
 2. A Python script `clustering.py` with a customer segmentation.
       
 3. A Python script `classifier.py` which classify individuals in customers or non-customers. 
 
### Licensing, Authors, Acknowledgements

Must give credit to [Udacity](https://www.udacity.com/) for the data.
