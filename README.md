# CS_540_final_project_danylam_shahids1
CS 540: Artificial Intelligence Final Project  
Spam Email Classification  
Mian Danyal, Shazman Shahid  

## Code description
main.py runs a classification learning model using 3 models, Logistic Regression, Stochastic Gradient Descent (SGD Classifier) and Multinomial Naive Bayes (Multinommial NB). We test on two datasets of spam/ham emails (ham means not spam), one with ~5,000 emails, the other with ~190,000 emails. These are input into the code as "spam_emails_dataset_5" or "spam_emails_dataset_190".  
  
We compare the 3 models, and evaluate the best model with accuracy, classification report, and cross validation. We visualize the data through word clouds and bar graphs for spam emails vs ham emails. We also create ROC curves for the 3 models, the train learning curves, the test learning curves, and then a train/test learning curve for the best model.  
  
Our code can easily be extended for use with other datasets and also to include more models.