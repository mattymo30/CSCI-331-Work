import math
import pickle
import sys

"""
file: lab3.py
CSCI-331
author: Matthew Morrison msm8275

Either train or predict a given set of examples compared to a
decision tree or adaboost algorithm
"""

learning_types = ["dt", "ada"]
features_lst = []
data_points = []
data_size = 0
features_size = 0
max_dt_depth = 22
max_adaboost_depth = 3

class TrainingDataPoint:
    def __init__(self):
        self.words = ""
        self.language = ""
        self.weight = 0

class PredictDataPoint:
    def __init__(self):
        self.words = ""

class TreeNode:
    def __init__(self):
        self.feature = ""
        self.attr_val = ""
        self.importance = 0
        self.children = []
        self.label = None # for leaf node base cases

class Hypothesis:
    def __init__(self):
        self.dt = None
        self.w = 0

def interpret_training_examples(examples_file):
    """
    given a file of training data, extrapolate the lines into usable
    data for the training algorithm
    :param examples_file: labeled examples file consisting of the following:
        -  label designating en for English or nl for Dutch
        - a | separating the label and words
        - a sequence of 15 words from a Wikipedia article
    """
    global data_size
    with open(examples_file, encoding="utf-8") as f:
        exs = f.readlines()
        for ex in exs:
            ex = ex.split("|")
            lang = ex[0]
            words = ex[1]
            data_point = TrainingDataPoint()
            data_point.language = lang
            data_point.words = words.lower()
            data_points.append(data_point)
            data_size += 1

def interpret_predict_examples(examples_file):
    """
    given a file of predict data, extrapolate the lines into usable
    data for the training algorithm
    :param examples_file: labeled examples file consisting of the following:
        - a sequence of 15 words from a Wikipedia article
    """
    global data_size
    with open(examples_file, encoding="utf-8") as f:
        exs = f.readlines()
        for ex in exs:
            data_point = PredictDataPoint()
            data_point.words = ex.lower()
            data_points.append(data_point)
            data_size += 1

def interpret_features(features_file):
    """
    given a file of features data, extract the lines into usable features to
    either train or predict
    :param features_file: the file containing the features to be used
    """

    global features_size
    with open(features_file) as f:
        features = f.readlines()
        for feature in features:
            features_lst.append(feature.strip())
            features_size += 1

def dtl(examples, attributes, parent_examples, depth, max_depth):
    """
    perform decision tree learning on a given set of examples
    :param examples: the list of example data points
    :param attributes: the list of possible attributes
    :param parent_examples: list of examples from the previous iteration
    :param depth: current depth of the dtl algorithm
    :param max_depth: maximum depth of the dtl algorithm
    :return: a tree, consisting of tree nodes,
    """
    if depth == max_depth:
        if parent_examples[0].weight == 0:
            return majority_answer(examples)
        return majority_answer_weighted(examples)
    if len(attributes) == 0:
        if parent_examples[0].weight == 0:
            return majority_answer(examples)
        return majority_answer_weighted(examples)
    if len(examples) == 0:
        if parent_examples[0].weight == 0:
            return majority_answer(parent_examples)
        return majority_answer_weighted(examples)

    # count each example in this iteration that are English and Dutch
    en_count = 0
    nl_count = 0
    ex_count = 0

    for data in examples:
        if data.language == "en":
            en_count += 1
        else:
            nl_count += 1
        ex_count += 1
    if en_count == ex_count or nl_count == ex_count:
        leaf = TreeNode()
        leaf.label = examples[0].language
        return leaf

    # find the best attribute by calculating importance function
    attr_imp_lst = []
    for attr in attributes:
        attr_imp = importance(attr, examples)
        attr_imp_lst.append((attr_imp, attr))

    # Sort by importance descending
    attr_imp_lst.sort(reverse=True)

    best_importance, best_attribute = attr_imp_lst[0]

    # create new node for the tree
    tree = TreeNode()
    tree.feature = best_attribute
    tree.importance = best_importance

    # split up based on if it has the attribute (word) or not
    ex_true = [d for d in examples if best_attribute in d.words]
    ex_false = [d for d in examples if best_attribute not in d.words]
    attr_lst = [a for a in attributes if a != best_attribute]

    true_child = dtl(ex_true, attr_lst, examples, depth + 1, max_depth)
    true_child.attr_val = True
    tree.children.append(true_child)

    false_child = dtl(ex_false, attr_lst, examples, depth + 1, max_depth)
    false_child.attr_val = False
    tree.children.append(false_child)

    return tree

def majority_answer(data_list):
    """
    get the majority answer and create a leaf node out of it
    :param data_list: the list of examples to go through
    :return: a new leaf node
    """
    en_count = 0
    nl_count = 0

    for data in data_list:
        if data.language == "en":
            en_count += 1
        else:
            nl_count += 1

    leaf = TreeNode()
    if en_count >= nl_count:
        leaf.label = "en"
    else:
        leaf.label = "nl"
    return leaf

def majority_answer_weighted(data_list):
    """
    get the majority answer and create a leaf node out of it with weighted
    examples
    :param data_list: the list of examples to go through
    :return: a new leaf node
    """
    en_weight = 0
    nl_weight = 0

    for data in data_list:
        if data.language == "en":
            en_weight += data.weight
        else:
            nl_weight += data.weight

    leaf = TreeNode()
    if en_weight >= nl_weight:
        leaf.label = "en"
    else:
        leaf.label = "nl"
    return leaf

def importance(attribute, examples):
    """
    calculate the importance of an attribute based on
    the examples in the given set
    :param attribute: the attribute to test against
    :param examples: the list of examples to test with
    :return: the numerical importance of the attribute
    """
    # check if we need to account for weighted examples
    if examples[0].weight == 0:
        return gain(attribute, examples)

    # incorporate weight into gain
    return gain_weighted(attribute, examples)

def gain(attribute, examples):
    """
    calculate the gain of an attribute based on the examples in the given set
    :param attribute: the attribute to test against
    :param examples: the list of examples to test with
    :return: the numerical importance of the attribute
    """
    en_count = 0
    for data in examples:
        if data.language == "en":
            en_count += 1

    en_prob = en_count / len(examples)

    en_entropy = get_binary(en_prob)
    en_rem = remainder(attribute, examples)

    return en_entropy - en_rem

def remainder(attribute, examples):
    """
    calculate the remainder of information for the attribute, which
    is used to calculate its importance
    :param attribute: the attribute to test against
    :param examples: the list of examples to test with
    :return: the numerical representation of the attribute's remaining info
    """
    # count if attr is seen in example or not
    attr_in_exs = 0
    attr_not_in_exs = 0
    # counters if example is el given attr is seen or not
    attr_in_en_counter = 0
    attr_not_in_en_counter = 0

    for data in examples:
        # get value for attribute
        if attribute in data.words:
            attr_in_exs += 1
            if data.language == "en":
                attr_in_en_counter += 1
        else:
            attr_not_in_exs += 1
            if data.language == "en":
                attr_not_in_en_counter += 1

    attr_in_exs_val = 0
    attr_not_in_exs_val = 0

    if attr_in_exs > 0:
        attr_in_exs_val = (attr_in_exs / len(examples)) * get_binary(attr_in_en_counter / attr_in_exs)
    if attr_not_in_exs > 0:
        attr_not_in_exs_val = (attr_not_in_exs / len(examples)) * get_binary(attr_not_in_en_counter / attr_not_in_exs)
    return attr_in_exs_val + attr_not_in_exs_val

def get_binary(prob):
    """
    get the entropy of a boolean variable
    :param prob: the probability of this boolean variable occurring
    :return: 0 if probability is 0 or 1,
    else run through equation and return final val
    """
    # edge cases
    if prob == 0 or prob == 1:
        return 0
    # B(X) = prob(X) * log2(1/prob(X)) + (1-prob(X)) * log2(1/(1-prob(X))
    else:
        prob_info = prob * math.log2(1/prob)
        other_prob = 1 - prob
        other_prob_info = other_prob * math.log2(1/other_prob)
        return prob_info + other_prob_info

def gain_weighted(attribute, examples):
    """
    calculate the gain of an attribute based on the weighted examples in the
    given set
    :param attribute: the attribute to test against
    :param examples: the list of weighted examples to test with
    :return: the numerical importance of the attribute
    """
    en_total_weight = 0
    total_weight = 0
    for data in examples:
        if data.language == "en":
            en_total_weight += data.weight
        total_weight += data.weight

    en_weighted_prob = en_total_weight / total_weight

    en_entropy = get_binary(en_weighted_prob)
    en_rem = remainder_weighted(attribute, examples)

    return en_entropy - en_rem

def remainder_weighted(attribute, examples):
    """
    calculate the remainder of information for the attribute, which
    is used to calculate its importance
    :param attribute: the attribute to test against
    :param examples: the list of weighted examples to test with
    :return: the numerical representation of the attribute's remaining info
    """
    # hold total weight for if attr is seen in example or not
    attr_in_exs_weight = 0
    attr_not_in_exs_weight = 0
    # hold total weight if example is en or not
    attr_in_en_weight = 0
    attr_not_in_en_weight = 0

    for data in examples:
        # check if attr in example
        if attribute in data.words:
            attr_in_exs_weight += data.weight
            if data.language == "en":
                attr_in_en_weight += data.weight
        else:
            attr_not_in_exs_weight += data.weight
            if data.language == "en":
                attr_not_in_en_weight += data.weight

    attr_in_exs_val = 0
    attr_not_in_exs_val = 0

    if attr_in_exs_weight > 0:
        attr_in_exs_val = (attr_in_en_weight / 1) * get_binary(attr_in_en_weight / attr_in_exs_weight)
    if attr_not_in_exs_weight > 0:
        attr_not_in_exs_val = (attr_not_in_en_weight / 1) * get_binary(attr_not_in_en_weight / attr_not_in_exs_weight)
    return attr_in_exs_val + attr_not_in_exs_val

def adaboost(examples, k):
    """
    perform adaboost on a given list of examples
    :param examples: the list of examples to train with
    :param k: the depth, or number of iterations to go through
    :return: a list of hypotheses (stumps) with their associated weight
    """
    h = []
    init_weight = 1 / len(examples)
    w = []
    for i, ex in enumerate(examples):
        ex.weight = init_weight
        w.append(init_weight)
    for i in range(k):
        # use dtl to create a stump
        hk = dtl(examples, features_lst, examples, 0, 1)
        hypo = Hypothesis()
        hypo.dt = hk
        err = 0
        num_correct = 0
        num_incorrect = 0
        incorrect_weight = w[0]
        correct_weight = 0
        for ex in examples:
            # check if predicted lang based on dt matches
            actual_lang = ex.language
            predicted_lang = interpret_ex(ex, hk)
            # compute error when incorrect
            if actual_lang != predicted_lang:
                ex_weight = ex.weight
                err += ex_weight
                num_incorrect += 1
            else:
                num_correct += 1
        # exit if error ever reaches above 0.5
        if err > 1/2: break
        delta_w = err / (1-err)
        for j, ex in enumerate(examples):
            actual_lang = ex.language
            predicted_lang = interpret_ex(ex, hk)
            # update weight for correctly classified examples
            if actual_lang == predicted_lang:
                updated_weight = ex.weight * delta_w
                w[j] = updated_weight
                # set correct weight to help normalize
                if correct_weight == 0:
                    correct_weight = updated_weight
        w = normalize_weights(w, num_correct, num_incorrect, correct_weight, incorrect_weight)
        # update weights for all examples
        for j, ex in enumerate(examples):
            ex.weight = w[j]
        hypo.w = .5 * math.log((1-err)/err)
        h.append(hypo)
    return h

def interpret_ex(example, dt):
    """
    interpret a dt to predict what the final value of an example
    should be
    :param example: the example to predict
    :param dt: the decision tree to base decision on
    :return: what the final value of an example should be, either
    en for English or nl for Dutch
    """
    while dt.label is None:
        node_feat = dt.feature
        dp_attr_value = node_feat in example.words
        # get the children
        for child in dt.children:
            if child.attr_val == dp_attr_value:
                dt = child
                break
    # reached leaf node, check what its value is
    final_guess = dt.label
    return final_guess

def normalize_weights(weights, num_correct, num_incorrect, correct_weight, incorrect_weight):
    """
    normalize the weights in a given adaboost cycle
    :param weights: the list of all weights in the examples
    :param num_correct: the number of correct answers
    :param num_incorrect: the number of incorrect answers
    :param correct_weight: the weight of a correct answer
    :param incorrect_weight: the weight of an incorrect answer
    :return: updated list of normalized weights
    """
    norm = (num_correct * correct_weight) + (num_incorrect * incorrect_weight)
    norm_correct_weight = correct_weight / norm
    norm_incorrect_weight = incorrect_weight / norm
    for i, w in enumerate(weights):
        if w == correct_weight:
            weights[i] = norm_correct_weight
        else:
            weights[i] = norm_incorrect_weight
    return weights

def train(examples, features, hypothesis_out, learning_type):
    """
    Reading in labeled examples, train either a decision tree or
    :param examples: labeled examples file consisting of the following:
        -  label designating en for English or nl for Dutch
        - a | separating the label and words
        - a sequence of 15 words from a Wikipedia article
    :param features: file containing a list of at least five distinct
    features of either the Dutch or English language
    :param hypothesis_out: the file name to write the model to
    :param learning_type: the type of learning algorithm to run,
    either "dt" for decision tree or "ada" for adaboost
    """
    interpret_training_examples(examples)
    interpret_features(features)

    if learning_type == "dt":
        tree = dtl(data_points, features_lst, data_points, 0, max_dt_depth)
        with open(hypothesis_out, 'wb') as f:
            pickle.dump(tree, f)

    elif learning_type == "ada":
        hypotheses = adaboost(data_points, max_adaboost_depth)
        with open(hypothesis_out, 'wb') as f:
            pickle.dump(hypotheses, f)

def predict(examples, features, hypothesis):
    """
    Given the specified hypothesis model, predict each example
    as English or Dutch text
    :param examples: contains lines of 15-word sentence fragments in
    either English or Dutch language
    :param features: the list of distinct features to test on
    :param hypothesis: the trained decision tree or ensemble created
    by the train function

    predict will print out for each input example the predicted label,
    either en for English or nl for Dutch
    """
    interpret_predict_examples(examples)
    interpret_features(features)

    f = open(hypothesis, 'rb')
    hypothesis = pickle.load(f)

    # single tree indicates a dt
    if type(hypothesis) is TreeNode:
        predict_dt(data_points, hypothesis)
    # list of hypotheses were found, indicates adaboost
    else:
        predict_ada(data_points, hypothesis)




def predict_dt(examples, tree):
    """
    predict the examples' language based on a single decision tree
    :param examples: the list of examples to predict
    :param tree: the decision tree
    predictions will be printed out to console
    """
    # predictions = []
    for example in examples:
        guess = interpret_ex(example, tree)
        print(guess)
        # predictions.append(guess)
    # compare(predictions, "answers.dat")


def predict_ada(examples, hypotheses):
    """
    predict the examples' language based on adaboost hypotheses, consisting
    of several stumps and their weights
    :param examples: the list of examples to predict
    :param hypotheses: a list of hypotheses (stumps) with their associated weight
    predictions will be printed out to console
    """
    # predictions = []
    for example in examples:
        en_guess = 0
        dl_guess = 0
        # must run through each tree hypothesis
        for hypothesis in hypotheses:
            guess = interpret_ex(example, hypothesis.dt)
            if guess == "en":
                en_guess += hypothesis.w
            else:
                dl_guess += hypothesis.w
        if en_guess >= dl_guess:
            print("en")
            # predictions.append("en")
        else:
            print("nl")
            # predictions.append("nl")
    # compare(predictions, "answers2.dat")

def compare(predictions, answers_file):
    """
    compare a tree's predictions to the actual answers
    :param predictions: a list of predictions made
    :param answers_file: a .dat file that contains the answers to
    the examples predicted on

    results will be printed out to console
    """
    answers = []
    with open(answers_file, 'rb') as f:
        ans = f.readlines()
        for a in ans:
            a = a.decode("utf-8")
            answers.append(a.strip())

    correct = 0
    incorrect = 0

    for i in range(len(predictions)):
        prediction = predictions[i]
        answer = answers[i]

        if prediction == answer:
            correct += 1
        else:
            incorrect += 1
    print("Correct: ", correct)
    print("Incorrect: ", incorrect)

def main():
    if len(sys.argv) == 1:
        print("No args provided")
        return
    function = sys.argv[1]

    # check valid function called
    if function == "train":
        args = sys.argv[2:]
        if len(args) != 4:
            print("Usage: python lab3.py train <examples> <features> <hypothesisOut> <learning-type>")
            return
        examples = args[0]
        features = args[1]
        hypothesis_out = args[2]
        learning_type = args[3]
        if learning_type not in learning_types:
            print("Learning type must be \"dt\" or \"ada\"")
            return
        train(examples, features, hypothesis_out, learning_type)
    elif function == "predict":
        args = sys.argv[2:]
        if len(args) != 3:
            print("Usage: python lab3.py predict <examples> <features> <hypothesis>")
            return
        examples = args[0]
        features = args[1]
        hypothesis = args[2]
        predict(examples, features, hypothesis)
    else:
        print("Invalid function provided, only train and predict are allowed")
        return

    """
    global data_points
    train("train.dat", "features.txt", "best_ada.model", "ada")
    data_points = []
    predict("test2.dat", "features.txt", "best_ada.model")
    """

if __name__ == "__main__":
    main()