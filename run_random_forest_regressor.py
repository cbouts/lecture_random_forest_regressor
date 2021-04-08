import pandas
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt 

# sklearn implements this to export the random forest machine in a tree diagram data format. it saves this as a dot file. then it uses external program called "graphviz" to graph the tree.
# you have to install the graphviz program correctly in order to do this. but it only works well in linux. so we use the server to install graphviz. 

dataset = pandas.read_csv("temperature_data.csv")

# can ignore the index, or, since it reflects the time of the year, can leave it in
# "actual" is the target, all else (except index) is data

# # looking at the data:
# print(dataset)
# print(dataset.shape)
# print(dataset.describe())


# days of the week are category variables, we need to make dummies.
dataset = pandas.get_dummies(dataset)
print(dataset.columns)
# this gets the dummies for the entire dataset
# the result gives us a column of ...s. but really we want to know what all of the coluns are.
# so to see the coluns, we do:

# # we should also get dummies for month. HOW DO WE DO THIS?
# dataset = pandas.get_dummies(dataset["month"].values)
# print(dataset.columns)

target = dataset['actual'].values
data = dataset.drop('actual', axis = 1).values
# this allows us to get everything except for the actual
print(data)

kfold_object = KFold(n_splits = 4)
kfold_object.get_n_splits(data)


i = 0
for training_index, test_index in kfold_object.split(data):
	# we use the method split inside the kfold obj to split the data into 4 parts. then we get the training and testing indices to guide the model to run the correct test and training indices.
	print("case ", i)
	i = i+1
	# print("training: ", training_index)
	# print("test: ", test_index)
	# the for loop above runs 4 rounds, one that uses each of the splits we've already gotten as the test. so it uses each of the 4 possible scenarios we've created..
	data_training = data[training_index]
	# # gets the data array using the training index, meaning it gets the data array for only the relevant rows according to the training_index.
	data_test = data[test_index]
	# # gets the data array using the test index, meaning it gets data array from the only relevant rows according to test_index.
	target_training = target[training_index]
	# gets the data array for the target group's training data. 
	target_test = target[test_index]
	# gets data array for the target group's test data. 
	machine = RandomForestRegressor(n_estimators = 201, max_depth = 30)
	machine.fit(data_training, target_training)
	# fits in the TRAINING data and target, so as to act like we have no idea what's in our manufactured test group.
	new_target = machine.predict(data_test)
	# pretending that our data_test is the new data
	# but really we already know that the answer here is target_test, so we can see how well predictions align with target_test.
	# to measure this, we could use r2 score (need to import it first)
	print("Mean absolute error: ", metrics.mean_absolute_error(target_test,new_target))
	# mean absolute error is absolute value because we don't want the pos and neg errors to cancel each other out.

# we can do several things to examine the result of the regressor.
# we can look at feature importance. IN NOTES IT'S EXPLAINED. 
# here we do feature importance on the entire dataset because there are relatively few observations
machine = RandomForestRegressor(n_estimators = 201, max_depth = 30)
machine.fit(data, target)
feature_list = dataset.drop('actual', axis=1).columns
# creating a list of the feature names
feature_importances_raw = list(machine.feature_importances_)
# creating a list of the feature importances
print(feature_importances_raw)

# # we want to pair the two lists together. we use an in-line for loop to do this:
# feature_importances = [(feature, importance) for feature, importance in zip(feature_list, feature_importances_raw)]
# # for each feature and importance pair within the zipped two lists, it loops through the rows to give us the feature and impotance pairs.
# # writes out the feature and importance for the first one, for the second one, for the third one, etc.
# # this allows us to even do operations with it.
# print(feature_importances)


# doing feature importances with a rounded importance to 3 decimal points;
feature_importances = [(feature, round(importance,3)) for feature, importance in zip(feature_list, feature_importances_raw)]
# for each feature and importance pair within the zipped two lists, it loops through the rows to give us the feature and impotance pairs.
# writes out the feature and importance for the first one, for the second one, for the third one, etc.
# this allows us to even do operations with it.
# print(feature_importances)

feature_importances = sorted(feature_importances, key = lambda x:x[1], reverse = True)
# key is the criterion by which you want to sort the list.
# we want to sort by the importance of features in descending orders. this isn't sorting by names. so we use the lambda.
# lambda x:x[1]
	# we want to take item at index position 1 (which is the feature importance in this case) and do sorting according to this item.
	# reverse=true bc want to list from highest to lowest feature importance.
# print(feature_importances)

# another inline for loop to fix the printing
[print('{} : {}'.format(*i)) for i in feature_importances]
# python fits the first thing (the variable name) inside the first braces, and the second thing (feature importance) inside the second thing
# if you wanted to make the output look even better, you can make it the case that items in the first column take up a given number of character spaces (13 here)
	# [print('{:13} : {}'.format(*i)) for i in feature_importances]

# at this point, we know that the most important features are yesterday's temp and the historical average, while everything else is less than 0.03, so basically less than nothing.
# the importance of lag2 seems like it should be important. but given that we know yesterday's temperature, temperature from 2 days ago is actually not important.
# if computational power is of concern, we can consider dropping the least important factors in our regressions.

feature_importances_raw = sorted(feature_importances_raw, reverse = True)
# this line sets up feature_importances_raw up to be presented in the bar graph in order of importance (descending)

# plotting the importances:
x_values = list(range(len(feature_importances_raw)))
# x values are the values on the x axis.
# in bar chart, x_values means how many bars you have. you can get how many bars you have by going to the feature importances (which are the things you want 
# to plot) and getting the length of the list of these importances. the length of this is 10. then you put the length into the range. the range forms an array of [0,1,2,3,4,5,6,7,8,9]. then you put it in a list so that matplotlib can understand it.
plt.bar(x_values, feature_importances_raw, orientation='vertical')
# when we plot the bar, we want the raw feature importances. we can set orientation to vertical.
plt.xticks(x_values, feature_list, rotation='vertical')
# adds feature names to the x axis, but they still hang off the edge..
plt.ylabel('importance')
# labels the y with "importance"
plt.xlabel('feature')
plt.title('Feature Importances!!!')
plt.tight_layout()
# this makes it the case that everything fits on the plot.
plt.savefig("feature_importances.png")
plt.close()


# these pictures are very helpful when you have a lot more features which make it difficult to read the list of the features/therir importances.

# you could sort the feature importances before placing them in the graph. 

# when there are only 2 important features, you can simplify your model, and have the accuracy rate stay the same.
# simplify the model specifications, n_trees etc.

# ***************** DRAWING A TREE
from sklearn.tree import export_graphviz
# sklearn implements this to export the random forest machine in a tree diagram data format. it saves this as a dot file. then it uses external program called "graphviz" to graph the tree.
# you have to install the graphviz program correctly in order to do this. but it only works well in linux. so we use the server to install graphviz. 
import pydot 
# on the server, do:
# python3 -m pip install pydot
# sudo apt install graphviz

tree = machine.estimators_[4]
# our machine is an rf machine. it contains a lot of trees. each of the trees can have a different structure. if you want to plot a tree, you need to know which one you want to take out.
# to take out tree number 4, it takes out the fifth tree and stores it in the tree variable
export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')











