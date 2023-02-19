# Starbucks Customer Rewards

## Background

This project is part of the AWS Machine Learning Engineer Nanodegree with Udacity. Udacity partnered with Starbucks to provide simulated data on customer behavior.

Starbucks, the American coffeeshop chain, periodically sends promotional offers to users via channels such as a mobile app or email. As part of an incentives and marketing strategy, we want to better understand how customers are responding to these offers, and how we can determine the best offer to provide.

Providing better offers to customers can help to build brand loyalty, maximize engagement, and increase net revenue.

## Problem Statement

How can we leverage experimental data to enchance the rewards program at Starbucks?
We want to better understand what offers to provide individual users based on their purchasing habits.

## Datasets and Inputs

As provided by Udacity:

- The program used to create the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers.
- Each person in the simulation has some hidden traits that influence their purchasing patterns and are associated with their observable traits. People produce various events, including receiving offers, opening offers, and making purchases.
- As a simplification, there are no explicit products to track. Only the amounts of each transaction or offer are recorded.
- There are three types of offers that can be sent: buy-one-get-one (BOGO), discount, and informational. In a BOGO offer, a user needs to spend a certain amount to get a reward equal to that threshold amount. In a discount, a user gains a reward equal to a fraction of the amount spent. In an informational offer, there is no reward, but neither is there a requisite amount that the user is expected to spend. Offers can be delivered via multiple channels.
- The basic task is to use the data to identify which groups of people are most responsive to each type of offer, and how best to present each type of offer.

The data is divides into 3 sections, collected over a 30-day period:

- user profiles for rewards program customers
- reward offer types
- purchase records

## Approach/Solution Statement

I will apply supervised machine learning to predict the propensity that a customer will complete a transaction given an offer.

Multiple classification models are available, including but not limited to:

- Logistic Regression
- SVM
- Random Forest
- Naive Bayes
- Neural Network

### Benchmark

We can evaluate the models against the base performance of the rewards program. For instance, if the predictive performance of the model is equivalent to the base probabilities of users engaging with offers, then it can be disregarded.

### Evaluation Metrics

For a classification model, we can consider the core model performance metrics:  

- Accuracy
- F1 score
- Precision
- Recall

False positives would be to provide offers to users who would not engage with them.
False negatives are overlooking users who would engage with offers.  
In this context, false negatives are considered to be worse, so we can consider recall to be more important than precision

If we consider the marginal profit associated with each transaction, then we can estimate the monetary gain associated with deploying this type of model to customers as well.

### Design

The steps in this project include the following:  

- Data exploration: The provided data will be processed and explored in order to understand the core properties, and uncover possible insights that could be leveraged to build a model or even basic heuristics.
- Data cleaning and preprocessing: Data should be cleaned and processed so that it can be used to train a machine learning model. This includes handling invalid data, feature engineering, and preparing train/test sets.
- Model training: This includes establishing a benchmark model, tuning hyperparameters, and evaluating performance between different model choices on a validation set.
- Final report: Report on insights in the data as well as model performance and deployment plans
