# Decision Tree example for warehouse priority classification
# Copyright 2018 Denis Rothman MIT License. See LICENSE.
import pydotplus
import collections
from sklearn import tree

# DECISION TREE LEARNING :DECISION TREE CLASSIFIER
#https://en.wikipedia.org/wiki/Decision_tree_learning
 
# 1. Data Collection created from the value of each O1 location in
#    the warehouse sample based on 3 features:
#  a) priority/location weight which bears a heavy weight to make a decison becauseof the cost of transporting distances
#  b) a volume priority weight which is set to 1 because in the weights were alrady measured to create reward matrix
#  c) high or low probablities determined by an optimization factor. For this example, it reamains distance

# 2.Providing the features of the dataset
features = [ 'Priority/location', 'Volume', 'Flow_optimizer' ]

Y = ['Low', 'Low', 'High', 'High', 'Low', 'Low']    

# 3. The data itself extracted from the result matrix
X = [ [256, 1,0],     
      [320, 1,0],
      [500, 1,1],
      [400, 1,1],
      [320, 1,0],
      [256, 1,0]]
 
# 4. Running the standard inbuilt tree classifier
classify = tree.DecisionTreeClassifier()
classify = classify.fit(X,Y)

# 5.Producing visualization if necessary
info = tree.export_graphviz(classify,feature_names=features,out_file=None,filled=False,rounded=False)
graph = pydotplus.graph_from_dot_data(info)
 
edges = collections.defaultdict(list) 
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
 
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0] 

graph.write_png('warehouse_example_decision_tree.png')
print("Open the image to verify that the priority level prediction  of the results fits the reality of the reward matrix inputs")
      
