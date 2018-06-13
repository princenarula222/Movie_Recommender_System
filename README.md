# Movie_Recommender_System
This repository provides an implementation of a movie recommender system. 
Working principle is based on Item-Item collaborative filtering, considering users and movies as two different sets of items working 
together to learn the model parameters for each user and movie.

Refer the following links for a better understanding of the concepts utilized in this implementation.
https://www.youtube.com/watch?v=9AP-DgFBNP4
https://www.youtube.com/watch?v=YW2b8La2ICo

Movies dataset provided by movielens.org has been utilized for training the model. The dataset is included in this repository.


# Training the model
Run 'Recommender.py' to train the model.


# Result
Following files are generated in the root folder upon completion of training.

x.csv - stores learned parameters for users
theta.csv - stores learned parameters for movies
training.csv - stores the training data in a tabular format
estimation.csv - stores the predicted ratings in a tabular format
difference.csv - stores the arithmetic difference between estimation.csv and training.csv
recommend0.csv - stores the ids of movies recommended to user 1 (predicted rating > 2.5)

I have placed my results in the 'Result'(Result/) folder for reference.

#Tabular format
Each row represents a movie.
Each column represents a user.
