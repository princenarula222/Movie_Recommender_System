from jval import *
from thetaxcalc import thetaxcalc
import csv

np.set_printoptions(threshold=np.inf)


movieratings = np.genfromtxt("ratings.csv", skip_header=1, dtype="U8", delimiter=",")  # reading the data from csv file
r = movieratings[:, 0:3]
r = r.astype(float)                                                                    # changing data type to float

movies = r[:, 1]
users = r[:, 0]
movies = set(movies)
users = set(users)
nom = len(movies)
nou = len(users)
users = np.array(list(users))
movies = np.array(list(movies))
movies = np.sort(movies, axis=None)
users = np.sort(users, axis=None)

matrix = np.zeros((nom, nou), float)
rated = np.zeros((nom, nou))                                             # 1: user has rated the movie, 0: otherwise


def getkey(item):
    return item[1]


r = np.array(sorted(r, key=getkey))                                      # sorting the data with movie id as key

for i in r:
    t = np.where(movies == i[1])
    j = t[0][0]
    k = int(i[0])-1
    matrix[j, k] = i[2]                                              # creating table of movie id, user id and ratings
    rated[j, k] = 1


nom = 800
nou = 100
matrix = matrix[0:nom, 0:nou]                                        # restricting data set to 800 movies and 100 users
rated = rated[0:nom, 0:nou]

b = np.sum(matrix, axis=1)
a = np.count_nonzero(rated, axis=1)
for i in range(a.shape[0]):
    if a[i] == 0:
        a[i] = 1

target = np.matrix((np.divide(b, a))).T                                 # calculating average movie rating
print(target.shape)
matrix = matrix - target                                                # mean normalization

n = 18                                                                  # considering 18 movie genres
x = np.random.rand(n, nom)
theta = np.zeros((n, nou))                                              # initialize with zero or random values

matrat = np.multiply(matrix, rated)
prev = jval(x, theta, matrat, rated)                                    # calculating cost function value
curr = np.inf
epsi = np.inf


while epsi > 0.0001:
    x, theta = thetaxcalc(matrat, rated, nom, nou, n, x, theta)         # updating x and theta using gradient descent
    curr = jval(x, theta, matrat, rated)
    epsi = prev - curr
    prev = curr
    print(epsi)


predicted = np.matmul(theta.T, x).T + target
matt = np.multiply(matrix + target, rated) + np.multiply(predicted, np.ones((nom, nou))-rated)
matrix = matt + np.multiply(matt*(-1), np.ones((nom, nou))-rated)

recommend0 = []
for i in range(nom):
    if rated[i, 0] == 0:
        if predicted[i, 0] > 2.5:
            recommend0.append(movies[i])                                # movies recommended to user 1


difference = np.subtract(predicted, matt)

movies = np.mat(movies)
users = np.mat(users)
movies = movies[:, 0:nom]
users = users[0, 0:nou]
matstr = matrix.astype('|S40')
predictstr = predicted.astype('|S40')
diffstr = difference.astype('|S40')
movies = movies.astype('|S40')
users = users.astype('|S40')
ap = np.array([["MID/UID"]])
users = np.hstack((ap, users))
matstr = np.hstack((movies.T, matstr))
matstr = np.vstack((users, matstr))
predictstr = np.hstack((movies.T, predictstr))
predictstr = np.vstack((users, predictstr))
diffstr = np.hstack((movies.T, diffstr))
diffstr = np.vstack((users, diffstr))


recommend0 = np.array([recommend0])
recommend0 = recommend0.T
head = np.array([["MID/User1"]])
recommend0 = np.vstack((head, recommend0))


b = open('training.csv', 'w')                                            # storing data into files
a = csv.writer(b, delimiter=',')
a.writerows(matstr.tolist())
b.close()

b = open('estimation.csv', 'w')
a = csv.writer(b, delimiter=',')
a.writerows(predictstr.tolist())
b.close()

b = open('difference.csv', 'w')
a = csv.writer(b, delimiter=',')
a.writerows(diffstr.tolist())
b.close()

b = open('recommend0.csv', 'w')
a = csv.writer(b, delimiter=',')
a.writerows(recommend0.tolist())
b.close()

b = open('x.csv', 'w')
a = csv.writer(b, delimiter=',')
a.writerows(x)
b.close()

b = open('theta.csv', 'w')
a = csv.writer(b, delimiter=',')
a.writerows(theta)
b.close()

print(curr)
