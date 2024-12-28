# Who Would Win Model (WWWModel)
# ML model for determining which of two fictional characters would win in a fight
# TODO: come up with cool name for model

from math import log
import numpy as np

# So far, this will be a decision tree
# TODO: convert to random decision forest to smoothen out noise

# Two characters will be chosen from a list of characters
# TODO: data scraping for characters not in list???!!!

# Each character will have an associated vector of features that corresponds to the following
# Each vector x is in R^10:
#   [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], where:
#       x1 = Mobility - overall, ie includes flying, teleporting, running, etc...
#       X2 = Physical Damage - only punches, kicking, weapons etc...
#       x3 = Elemental/Magic-esque Damage - typically casted abilities or not directly physical
#       X4 = Defense
#       X5 = Dexterity - variety of abilities a character utilizes like spells, weapons, etc...
#       x6 = Intelligence
#       X7 = Experience
#       x8 = X-Factor 1
#   All in [1-10] scale NATURAL NUMBERS
# TODO: improve? normalize? reduce dimensionality? hmm...

NUM_OF_FEATURES = 8
MAX_THRESHOLD = 10
MIN_THRESHOLD = -(MAX_THRESHOLD)
MIN_NODE_DP_COUNT = 3

# The input to the model would be the vector of the difference between the two chosen character vectors
# Then, this is inputted into the tree which classifies {1, 2}, where 1 means character 1 wins and 2 2 wins.

# Example character vectors:
# TODO: make a file for the data... and maybe get data via data scraping... where will i find this data lol
GOJO = np.array([10, 7, 10, 10, 7, 7, 9, 9])
GOKU = np.array([9, 10, 10, 8, 7, 6, 10, 10])
LUFFY = np.array([8, 9, 0, 7, 7, 6, 8, 8])
SAITAMA = np.array([10, 10, 0, 10, 4, 6, 8, 10])
GUY = np.array([10, 9, 4, 5, 5, 7, 9, 9])
NANAMI = np.array([5, 8, 0, 8, 5, 8, 9, 3])
GAARA = np.array([6, 4, 8, 8, 6, 7, 8, 6])
ITACHI = np.array([6, 5, 9, 6, 10, 9, 7, 7])
REINER = np.array([6, 8, 0, 9, 4, 6, 7, 6])
ACE = np.array([6, 5, 8, 4, 5, 6, 6, 6])
SHIKAMARU = np.array([4, 4, 7, 4, 9, 10, 8, 3])
TOJI = np.array([9, 10, 0, 8, 9, 8, 8, 7])
JOGO = np.array([6, 7, 9, 6, 6, 7, 7, 7])

# Takes as input two character vectors and returns the input vector that can be used on the decision tree
def make_input(characterOne, characterTwo):
    return characterOne - characterTwo

# Temporary training datapoints:
TRAINING_DATAPOINTS = [
    (make_input(GOJO, GOKU), 2),
    (make_input(GOKU, SAITAMA), 2),
    (make_input(GOJO, SAITAMA), 2),
    (make_input(GOKU, JOGO), 1),
    (make_input(GOJO, TOJI), 1),
    (make_input(GUY, NANAMI), 1),
    (make_input(JOGO, NANAMI), 1),
    (make_input(JOGO, ACE), 1),
    (make_input(JOGO, GAARA), 1),
    (make_input(GAARA, REINER), 2),
    (make_input(TOJI, LUFFY), 2),
    (make_input(TOJI, NANAMI), 1),
    (make_input(ITACHI, REINER), 1),
    (make_input(GAARA, ITACHI), 2),
    (make_input(NANAMI, REINER), 1),
    (make_input(GUY, GAARA), 1),
    (make_input(ACE, SHIKAMARU), 1),
    (make_input(NANAMI, SHIKAMARU), 2),
    (make_input(REINER, SHIKAMARU), 1),
    (make_input(LUFFY, GOKU), 2)
]
NUM_OF_TRAINING_DP = len(TRAINING_DATAPOINTS)
OUTPUTS = [1, 2] # class 1 - character 1 wins, class 2 - character 2 wins

class Node:
   def __init__(self, dp_list):
      self.dp_list = dp_list # list of all datapoints in node j
      self.dp_count = len(dp_list)
      self.split_feature = None
      self.split_threshold = None
      self.left = None
      self.right = None

def entropy(dp_list):
    if len(dp_list) == 0:
        return 0

    # each row is all the datapoints from node with class/output i
    dp_classes_matrix = []
    for i in OUTPUTS:
        dp_classes_matrix.append([])
    
    for dp in dp_list:
        dp_output = dp[1]
        dp_classes_matrix[dp_output-1].append(dp)
    
    sum = 0
    for i in OUTPUTS:
        pi = len(dp_classes_matrix[i-1]) / len(dp_list) # probability of class i in node
        sum += pi * log(pi, 2)

    return -(sum)

# Using split_feature x > split_threshold z
def information_gain(node, split_feature, split_threshold):
    if node is None or node.dp_count == 0:
        return 0

    # datapoints that go to the left subtree or right subtrees, respectively, after split
    left_subtree_dps = []
    right_subtree_dps = []
    for dp in node.dp_list:
        if (dp[0][split_feature] > split_threshold):
            right_subtree_dps.append(dp)
        else:
            left_subtree_dps.append(dp)
    
    left_entropy = entropy(left_subtree_dps)
    right_entropy = entropy(right_subtree_dps)
    result = entropy(node.dp_list)
    result += -(len(left_subtree_dps)/node.dp_count)*left_entropy
    result += -(len(right_subtree_dps)/node.dp_count)*right_entropy
    return result

# for root assumes: node = Node(TRAINING_DATAPOINTS)
def build_decision_tree(node):
    # base case
    if (entropy(node.dp_list) == 0) or (len(node.dp_list) < MIN_NODE_DP_COUNT):
        node.left = None
        node.right = None
        return node

    # getting best split function based on highest ig
    split_fcn = [-1, -1, -1]
    for feature in range(NUM_OF_FEATURES):
        for threshold in (MIN_THRESHOLD, MAX_THRESHOLD):
            ig = information_gain(node, feature, threshold)
            if ig > split_fcn[0]:
                split_fcn = [ig, feature, threshold]
    
    if split_fcn[0] == -1:
        node.left = None
        node.right = None
        return node
    
    # current node split is best possible split
    node.split_feature = split_fcn[1]
    node.split_threshold = split_fcn[2]

    # left and right child nodes datapoint list
    right_dp_list = []
    left_dp_list = []
    for dp in node.dp_list:
        if dp[0][node.split_feature] > node.split_threshold:
            right_dp_list.append(dp)
        else:
            left_dp_list.append(dp)
    
    right_child_node = Node(right_dp_list)
    left_child_node = Node(left_dp_list)
    node.right = build_decision_tree(right_child_node)
    node.left = build_decision_tree(left_child_node)
    return node