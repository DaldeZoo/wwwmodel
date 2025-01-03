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

FEATURE_COUNT = 8
MAX_THRESHOLD = 10
MIN_THRESHOLD = -(MAX_THRESHOLD)
MIN_NODE_DP_COUNT = 3

# The input to the model would be the vector of the difference between the two chosen character vectors
# Then, this is inputted into the tree which classifies {1, 2}, where 1 means character 1 wins and 2 2 wins.

# Example character tuples:
# TODO: make a file for the data... and maybe get data via data scraping... where will i find this data lol
# characters are represented by a tuple: idx 0 is their name and 1 is their stats/features vector
NAME = 0
STATS = 1
GOJO = ("Gojo", np.array([10, 7, 10, 10, 7, 7, 9, 9]))
GOKU = ("Goku", np.array([9, 10, 10, 8, 7, 6, 10, 10]))
LUFFY = ("Luffy", np.array([8, 9, 0, 7, 7, 6, 8, 8]))
SAITAMA = ("Saitama", np.array([10, 10, 0, 10, 4, 6, 8, 10]))
GUY = ("Guy", np.array([10, 9, 4, 5, 5, 7, 9, 9]))
NANAMI = ("Nanami", np.array([5, 8, 0, 8, 5, 8, 9, 3]))
GAARA = ("Gaara", np.array([6, 4, 8, 8, 6, 7, 8, 6]))
ITACHI = ("Itachi", np.array([6, 5, 9, 6, 10, 9, 7, 7]))
REINER = ("Reiner", np.array([6, 8, 0, 9, 4, 6, 7, 6]))
ACE = ("Ace", np.array([6, 5, 8, 4, 5, 6, 6, 6]))
SHIKAMARU = ("Shikamaru", np.array([4, 4, 7, 4, 9, 10, 8, 3]))
TOJI = ("Toji", np.array([9, 10, 0, 8, 9, 8, 8, 7]))
JOGO = ("Jogo", np.array([6, 7, 9, 6, 6, 7, 7, 7]))

# Takes as input two character tuples and returns the input vector that can be used on the decision tree
def get_input(character_one, character_two):
    return character_one[STATS] - character_two[STATS]


# Temporary training datapoints:
# these are labeled training datapoints ie (input, output)
TRAINING_DATAPOINTS = [
    (get_input(GOJO, GOKU), 1),
    (get_input(GOKU, GOJO), 0),

    (get_input(GOKU, SAITAMA), 1),
    (get_input(SAITAMA, GOKU), 0),

    (get_input(GOJO, SAITAMA), 1),
    (get_input(SAITAMA, GOJO), 0),

    (get_input(GOKU, JOGO), 0),
    (get_input(JOGO, GOKU), 1),

    (get_input(GOJO, TOJI), 0),
    (get_input(TOJI, GOJO), 1),

    (get_input(GUY, NANAMI), 0),
    (get_input(NANAMI, GUY), 1),

    (get_input(JOGO, NANAMI), 0),
    (get_input(NANAMI, JOGO), 1),

    (get_input(JOGO, ACE), 0),
    (get_input(ACE, JOGO), 1),

    (get_input(JOGO, GAARA), 0),
    (get_input(GAARA, JOGO), 1),

    (get_input(GAARA, REINER), 1),
    (get_input(REINER, GAARA), 0),

    (get_input(TOJI, LUFFY), 1),
    (get_input(LUFFY, TOJI), 0),

    (get_input(TOJI, NANAMI), 0),
    (get_input(NANAMI, TOJI), 1),

    (get_input(ITACHI, REINER), 0),
    (get_input(REINER, ITACHI), 1),

    (get_input(GAARA, ITACHI), 1),
    (get_input(ITACHI, GAARA), 0),

    (get_input(NANAMI, REINER), 0),
    (get_input(REINER, NANAMI), 1),

    (get_input(GUY, GAARA), 0),
    (get_input(GAARA, GUY), 1),

    (get_input(ACE, SHIKAMARU), 0),
    (get_input(SHIKAMARU, ACE), 1),

    (get_input(REINER, SHIKAMARU), 0),
    (get_input(SHIKAMARU, REINER), 1),
    
    (get_input(LUFFY, GOKU), 1),
    (get_input(GOKU, LUFFY), 0)
]
# ensure both orderings of characters are provided in training datapoints

OUTPUTS = (0, 1) # class 0 - character 1 wins, class 1 - character 2 wins
class Node:
   def __init__(self, dp_list):
      self.dp_list = dp_list # list of all datapoints in node j
      self.dp_count = len(dp_list)
      self.split_feature = -1
      self.split_threshold = -1
      self.class_label = -1
      self.left = None
      self.right = None

def entropy(dp_list):
    if len(dp_list) == 0:
        return 0

    # each element i is the number of datapoints in dp_list with class label i
    dp_classes_count = []
    for i in OUTPUTS:
        dp_classes_count.append(0)
    for dp in dp_list:
        dp_output = dp[1]
        dp_classes_count[dp_output] += 1

    sum = 0
    for i in OUTPUTS:
        if dp_classes_count[i] > 0:
            class_probability = dp_classes_count[i] / len(dp_list) # probability of class i in node
            sum += class_probability * log(class_probability, 2)

    return -(sum)

# Using split_feature x > split_threshold z
def information_gain(node, split_feature, split_threshold):
    if node is None or node.dp_count == 0:
        return -1

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

# removes datapoint list and count from each node, as is no longer needed after tree is built
def clean_tree(root):
    if root is None:
        return
    clean_tree(root.left)
    clean_tree(root.right)
    root.dp_count = None
    root.dp_list = None

def build_decision_tree(node):
    # base case: no uncertainty or num of dps small enough, then becomes leaf node
    if (entropy(node.dp_list) == 0) or (len(node.dp_list) < MIN_NODE_DP_COUNT):
        node.left = None
        node.right = None
        node.class_label = classify_node(node)
        return node

    # getting best split function based on highest ig
    split_fcn = [-1, -1, -1]
    for feature in range(FEATURE_COUNT):
        for threshold in range(MIN_THRESHOLD, MAX_THRESHOLD):
            ig = information_gain(node, feature, threshold)
            if ig > split_fcn[0]:
                split_fcn = [ig, feature, threshold]
    
    if split_fcn[0] == -1:
        node.left = None
        node.right = None
        node.class_label = classify_node(node)
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

    clean_tree(node)
    return node
# TODO: after tree is built, dp_list and counts in each node are pointless...

def classify_node(node):
    # each element i is the number of datapoints in node.dp_list with class label i
    classes_count = []
    for i in OUTPUTS:
        classes_count.append(0)

    for dp in node.dp_list:
        dp_output = dp[1]
        classes_count[dp_output] += 1

    return classes_count.index(max(classes_count))

# assumes fully built decision tree
# returns winner character tuple
def get_winner(node, characters):
    split_feature = node.split_feature
    split_threshold = node.split_threshold
    input = get_input(characters[0], characters[1])
    while (node.left != None): # perfect binary tree, checking only left suffices
        if input[split_feature] > split_threshold:
            node = node.right
        else:
            node = node.left
    winner = node.class_label
    return characters[winner]

# running and testing model
node = Node(TRAINING_DATAPOINTS)
root = build_decision_tree(node)
characters = (GOJO, GOKU)
winner = get_winner(root, characters)
print(f"{winner[0]} wins!")
characters = (GOKU, GOJO)
winner = get_winner(root, characters)
print(f"{winner[0]} wins!")

# TODO: need more data for better results