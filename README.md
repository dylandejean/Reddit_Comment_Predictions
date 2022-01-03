# Reddit Comment Predictions

This project is designed to use a dataset of 20k Reddit comments and predict which subreddit each one came from. The model was created using a training set of 200 comments independent of the 20k being predicted.

This was a final project for STAT 380 (Data Science using R) at The Pennsylvania State University

## Methods

Each comment is embedded and then, using t-SNE, the data is reduced to just three dimentions. 

XGBoost is then used to build and hypertune a model that predicts probabilities of each comment being a part of the 10 possible subreddits.

## Results

The final predictions are found in /project/volume/data/processed/submit.csv. Each row corresponds to the comments in the test set, with probabilities for that comment belonging to each of the 10 possible subreddits.
