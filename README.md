# Customer-churn.-Discovering-why-customers-are-cancelling-their-subscriptions-using-machine-learning.

Customer churn is the term used when a customer simply ceases to be a customer. They stop doing business with a company or discontinue their subscription. Knowing why customers might stop purchasing your goods and services is integral to any business.

The dataset within this depository contains information about customers who cancelled their subscriptions with Telco, a telecommunications company. Telco sells a variety of services and offers subscriptions in various kinds of contracts. In this repository I examine the features of the data set in order to determine the main factors causing customers to cease doing business with Telco.

Data Source: https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/

This investigation is composed of 2 parts split between 2 Jupyter notebooks. Part 1 is an exploratory analysis of the data. Part 2 is the modelling section that seeks to predict customer churn and outline the factors driving churn.

Summary: The main drivers of churn came from customers that were using Telco’s Fiber optic service and from customers that were paying more than what was typical. Customers that were least likely to churn had one/two year contracts with Telco. A success story for Telco was brought via its tech support as customers who used it were less likely to churn. Ultimately Telco needs to improve its fiber optic offering, maintain the quality of its tech support and lock in customers with contracts by enticing them with lower prices.

Logistic Regression, Random Forests and Support Vector Machines were used to model the data. Various techniques were also employed to increase predictive performance and aid in the examination of the features that drive churn. Altering class weights, synthetic sampling, recursive feature elimination and feature interactions were all explored.

Ultimately Logistic Regression yielded the best results (based on f1 score). The coefficients derived from the model were very revealing as to what was driving churn. The random forest was just slightly inferior. A decision tree based on the forest’s parameters was also very informative.

