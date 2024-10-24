"""
criterion
"""

import math


def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    def entropy(labels):
        total = sum(labels.values())
        return -sum((count / total) * math.log2(count / total) for count in labels.values())

    # Calculate entropy of the parent node
    parent_entropy = entropy(all_labels)

    # Calculate weighted entropy of child nodes
    total_samples = len(y)
    left_weight = len(l_y) / total_samples
    right_weight = len(r_y) / total_samples

    left_entropy = entropy(left_labels)
    right_entropy = entropy(right_labels)

    weighted_child_entropy = left_weight * left_entropy + right_weight * right_entropy

    # Calculate information gain
    info_gain = parent_entropy - weighted_child_entropy
   




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Calculate parent entropy
    def entropy(labels):
        total = sum(labels.values())
        return -sum((count / total) * math.log2(count / total) for count in labels.values())
    
    parent_entropy = entropy(__label_stat(y, l_y, r_y)[0])

    # Avoid division by zero
    if parent_entropy == 0:
        return 0
    
    # Calculate info gain ratio
    info_gain_ratio = info_gain / parent_entropy

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Calculate Gini index
    # Gini(D) = 1 - sum(p_i^2)
    # where p_i is the proportion of samples belonging to class i

    # Calculate parent Gini index
    def gini(labels):
        total = sum(labels.values())
        return 1 - sum((count / total)**2 for count in labels.values())
    
    parent_gini = gini(all_labels)
    
    # Calculate Gini index for left and right children
    left_gini = gini(left_labels)
    right_gini = gini(right_labels)

    # Calculate weighted Gini index after split
    # Gini_split(D, A) = (|D1|/|D|) * Gini(D1) + (|D2|/|D|) * Gini(D2)
    # where D1 and D2 are the two subsets created by splitting on attribute A
    left_weight = sum(left_labels.values()) / sum(all_labels.values())
    right_weight = sum(right_labels.values()) / sum(all_labels.values())
    weighted_gini = left_weight * left_gini + right_weight * right_gini

    # Calculate Gini gain
    # Gini_gain(D, A) = Gini(D) - Gini_split(D, A)
    before = parent_gini
    after = weighted_gini

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Calculate error rate
    # Error rate = 1 - (count of majority class / total count)
    def error_rate(labels):
        total = sum(labels.values())
        if total == 0:
            return 0  # Return 0 if there are no labels
        max_label = max(labels.values()) if labels else 0
        return 1 - (max_label / total)


    # Calculate parent error rate
    parent_error = error_rate(all_labels)

    # Calculate error rates for left and right children
    left_error = error_rate(left_labels)
    right_error = error_rate(right_labels)

    # Calculate weighted error rate after split
    # Weighted_Error = (|D1|/|D|) * Error(D1) + (|D2|/|D|) * Error(D2)
    # where D1 and D2 are the two subsets created by splitting
    total_samples = sum(all_labels.values())
    if total_samples == 0:
        return 0  # Return 0 if there are no samples

    left_weight = sum(left_labels.values()) / sum(all_labels.values())
    right_weight = sum(right_labels.values()) / sum(all_labels.values())
    weighted_error = left_weight * left_error + right_weight * right_error

    # Calculate Error rate reduction
    # Error_reduction = Error(D) - Weighted_Error
    before = parent_error
    after = weighted_error

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
