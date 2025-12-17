import sys
from enum import Enum
import copy
"""
file: lab2.py
CSCI-331
author: Matthew Morrison msm8275

Given a Knowledge Base in Conjunctive Normal Form, determine if the
KB is satisfiable based on if it finds the empty clause or not
"""

predicates = dict()
variables = dict()
constants = dict()
functions = dict()
clauses = [] # keep in list to preserve ordering that was causing
             # uncertainty in some answers

NEW_VAR_LETTER = "Q"
NEW_VAR_NUMBER = 200

class TermType(Enum):
    CONSTANT = 1,
    VARIABLE = 2,
    FUNCTION = 3

class Clause:
    def __init__(self):
        self.predicates = []

    def __eq__(self, other):
        return isinstance(other, Clause) and self.predicates == other.predicates

    def __hash__(self):
        return hash(tuple(self.predicates))


class Predicate:
    def __init__(self):
        self.is_negated = False
        self.pred_name = ""
        self.terms = []

    def __eq__(self, other):
        return (
                isinstance(other, Predicate) and
                self.pred_name == other.pred_name and
                self.is_negated == other.is_negated and
                self.terms == other.terms
        )

    def __hash__(self):
        return hash((self.pred_name, self.is_negated, tuple(self.terms)))

class Term:
    def __init__(self):
        self.term_type = None
        self.term = None

    def __eq__(self, other):
        return isinstance(other, Term) and self.term_type == other.term_type and self.term == other.term

    def __hash__(self):
        return hash((self.term_type, self.term))


class Constant:
    def __init__(self):
        self.string_rep = ""

    def __eq__(self, other):
        return isinstance(other, Constant) and self.string_rep == other.string_rep

    def __hash__(self):
        return hash(self.string_rep)


class Variable:
    def __init__(self):
        self.string_rep = ""

    def __eq__(self, other):
        return isinstance(other, Variable) and self.string_rep == other.string_rep

    def __hash__(self):
        return hash(self.string_rep)


class Function:
    def __init__(self):
        self.function_name = ""
        self.terms = []

    def __eq__(self, other):
        return isinstance(other, Function) and self.function_name == other.function_name and self.terms == other.terms

    def __hash__(self):
        return hash((self.function_name, tuple(self.terms)))


def convert_kb(knowledge_base_file):
    """
    Given a knowledge base file, convert the text input into
    valid objects to perform resolution on
    :param knowledge_base_file: the knowledge base
    """
    with open(knowledge_base_file) as file:
        lines = file.readlines()

        # get predicates
        predicate_line = lines[0].split()[1:]
        if len(predicates) >= 0:
            for predicate in predicate_line:
                predicate_obj = Predicate()
                predicate_obj.string_pred_name = str(predicate)
                predicates[predicate_obj.string_pred_name] = predicate_obj

        # get variables
        variable_line = lines[1].split()[1:]
        if len(variable_line) >= 1:
            for variable in variable_line:
                variable_obj = Variable()
                variable_obj.string_rep = str(variable)
                variables[variable_obj.string_rep] = variable_obj

        # get constants
        constant_line = lines[2].split()[1:]
        if len(constant_line) >= 1:
            for constant in constant_line:
                constant_obj = Constant()
                constant_obj.string_rep = str(constant)
                constants[constant_obj.string_rep] = constant_obj

        # get functions
        function_line = lines[3].split()[1:]
        if len(function_line) >= 1:
            for function in function_line:
                function_obj = Function()
                function_obj.function_name = str(function)
                functions[function_obj.function_name] = function_obj

        # get all clauses and interpret them
        clauses_list = lines[5:]
        for clause in clauses_list:
            clause_predicates = clause.split()
            curr_clause = Clause()
            for predicate in clause_predicates:
                p = convert_predicate(predicate)
                curr_clause.predicates.append(p)
            clauses.append(curr_clause)



def convert_predicate(predicate):
    """
    given a predicate string, convert it to a predicate object
    for usage in resolution
    :param predicate: the predicate string representation
    :return: predicate object
    """
    is_negated = False

    i = 0

    # check if the predicate is negated
    if predicate[0] == "!":
        is_negated = True
        # increment so we can ignore negation symbol
        i = 1

    # scrap string for terms
    pred_str = ""
    pred = None
    while i < len(predicate):
        curr_char = predicate[i]
        # found end of pred name and it has terms, interpret terms
        if curr_char == "(":
            # get original
            pred = copy.deepcopy(predicates[pred_str])
            pred.is_negated = is_negated
            pred.pred_name = pred_str
            # get the terms
            terms = predicate[i+1:-1]
            # convert and exit loop
            pred.terms = convert_terms(terms)
            pred_str = ""
            break
        pred_str += predicate[i]
        i += 1
    # have predicate with no terms, just add
    if pred_str != "":
        pred = copy.deepcopy(predicates[pred_str])
        pred.is_negated = is_negated
        pred.pred_name = pred_str
    return pred

def convert_function(function):
    """
    given a function string, convert it to a function object
    for usage in resolution

    :param function: the function string representation
    :return: function object
    """
    # scrap string for terms
    func_str = ""
    func = None
    i = 0
    while i < len(function):
        curr_char = function[i]
        # found end of func name, interpret terms
        if curr_char == "(":
            # get original
            func = copy.deepcopy(functions[func_str])
            func.function_name = function
            # get the terms
            terms = function[i + 1:-1]
            # convert and exit loop
            func.terms = convert_terms(terms)
            break
        func_str += curr_char
        i += 1
    return func

def convert_terms(terms):
    """
    convert terms in a function or predicate to usable objects
    :param terms: the string representation of terms
    :return: a list of term objects
    """
    terms_lst = []

    curr_str = ""
    i = 0
    while i < len(terms):
        curr_char = terms[i]
        # hit a function, need to convert first
        if curr_char == "(":
            curr_str += curr_char
            # get complete terms for function
            i += 1
            open_paren = 1
            while open_paren > 0:
                curr_char = terms[i]
                curr_str += curr_char
                if curr_char == "(":
                    open_paren += 1
                elif curr_char == ")":
                    open_paren -= 1
                i += 1
            func = convert_function(curr_str)
            terms_lst.append(func)
            curr_str = ""
            continue
        # hit a constant or variable
        elif curr_char == ",":
            if curr_str != "":
                if curr_str in variables:
                    var = copy.deepcopy(variables[curr_str])
                    terms_lst.append(var)
                elif curr_str in constants:
                    constant = copy.deepcopy(constants[curr_str])
                    terms_lst.append(constant)
                else:
                    print("term does not exist in KB")
                    sys.exit(1)
                curr_str = ""
                i += 1
                continue
            else:
                curr_str = ""
                i += 1
        else:
            curr_str += curr_char
            i += 1
    if curr_str != "":
        if curr_str in variables:
            var = copy.deepcopy(variables[curr_str])
            terms_lst.append(var)
        elif curr_str in constants:
            constant = copy.deepcopy(constants[curr_str])
            terms_lst.append(constant)
        else:
            print("term does not exist in KB")
            sys.exit(1)

    # for every term, now convert into a term object
    term_objects = []
    for term in terms_lst:
        new_term = Term()
        if type(term) == Variable:
            new_term.term_type = TermType.VARIABLE
        elif type(term) == Constant:
            new_term.term_type = TermType.CONSTANT
        elif type(term) == Function:
            new_term.term_type = TermType.FUNCTION
        else:
            print("term does not exist in KB")
            sys.exit(1)
        new_term.term = term
        term_objects.append(new_term)

    return term_objects

def resolution(knowledge_base_file):
    """
    Given a knowledge base file, perform resolution to determine if
    the KB is satisfiable or not.
    A KB is satisfiable if, through resolution, it cannot find any
    new clause
    A KB is not satisfiable if, through resolution, it discovers the empty
    clause

    The function will print out "yes" if satisfiable, otherwise "no"
    :param knowledge_base_file:
    """
    global clauses
    # convert KB to usable objects
    convert_kb(knowledge_base_file)

    while True:
        # holds any new created clauses found
        clause_list = list(clauses)
        new_clauses = set()
        for i in range(len(clause_list)):
            c1 = clause_list[i]
            for j in range(i + 1, len(clause_list)):
                c2 = clause_list[j]

                resolvents = resolve(c1, c2)

                # clauses could not be resolved, continue
                if len(resolvents) == 0:
                    continue

                # found empty clause, print no and exit
                for clause in resolvents:
                    if len(clause.predicates) == 0:
                        print("no")
                        return
                # union and remove original clauses
                for clause in resolvents:
                    new_clauses.add(clause)
        # no new clauses found, print yes and exit
        if is_subset(new_clauses, clauses):
            print("yes")
            return
        # else union new clauses to original clauses
        for new_clause in new_clauses:
            if new_clause not in clauses:
                clauses.append(new_clause)


def resolve(c1, c2):
    """
    attempt to resolve two clauses
    :param c1: clause 1
    :param c2: clause 2
    :return: a set of resolvents, or new clauses, or None if
    resolving fails
    """
    c1_preds = c1.predicates
    c2_preds = c2.predicates

    resolvents = []
    for c1_pred in c1_preds:
        for c2_pred in c2_preds:
            # are the predicates complements?
            if is_complement(c1_pred, c2_pred):
                # check and see if they have terms
                if len(c1_pred.terms) == 0 and len(c2_pred.terms) == 0:
                    # no terms? just resolve without unification
                    new_clause = Clause()
                    # union and remove the preds being compared
                    new_preds = []
                    for pred in c1.predicates:
                        if pred != c1_pred:
                            new_preds.append(pred)
                    for pred in c2.predicates:
                        if pred != c2_pred:
                            new_preds.append(pred)
                    new_clause.predicates = new_preds
                    # check for two complementary preds
                    if check_complement_preds(new_preds):
                        resolvents.append(new_clause)
                # mismatch of number of terms, cannot resolve
                elif len(c1_pred.terms) != len(c2_pred.terms):
                    continue
                # terms are of the same length, need to attempt to unify
                else:
                    theta = dict()
                    # single terms? just take out of list and call unify
                    # helps to ignore case when checking if they are lists
                    if len(c1_pred.terms) == 1 and len(c2_pred.terms) == 1:
                        theta = unify(c1_pred.terms[0], c2_pred.terms[0], theta)
                    # have to senf full lists as args
                    else:
                        theta = unify(c1_pred.terms, c2_pred.terms, theta)
                    # terms cannot be unified, continue
                    if theta is None:
                        continue

                    new_clause = Clause()
                    # union and remove the preds being compared
                    new_preds = []
                    for pred in c1.predicates:
                        if pred != c1_pred:
                            new_preds.append(pred)
                    for pred in c2.predicates:
                        if pred != c2_pred:
                            new_preds.append(pred)
                    if len(theta) != 0:
                        substitute(new_preds, theta)
                    new_clause.predicates = new_preds
                    # check for two complementary preds
                    if check_complement_preds(new_preds):
                        resolvents.append(new_clause)

    return resolvents

def unify(x, y, theta):
    """
    given two expressions, x and y, attempt to make them identical
    :param x: the first expression
    :param y: the second expression
    :param theta: a dict of pairs that can be used for substitutions
    for the expressions to make them the same
    :return: theta, if the two expressions can become identical,
    or None if they cannot be unified
    """
    # already found None? then expressions cannot be unified
    if theta is None:
        return None
    # are expressions already the same?
    elif x == y:
        return theta
    # unify vars
    elif type(x) == Term and type(x.term) == Variable:
        return unify_var(x.term, y, theta)
    elif type(y) == Term and type(y.term) == Variable:
        return unify_var(y.term, x, theta)
    # compound expressions need to be broken up
    elif (type(x) == Term and type(x.term) == Function) and (type(y) == Term and type(y.term)) == Function:
        return unify(x.terms, y.terms, unify(x.function_name, y.function_name, theta))
    # lists need to be broken down one by one
    elif isinstance(x, list) and isinstance(y, list):
        return unify(x[1:], y[1:], unify(x[0], y[0], theta))
    else:
        return None

def substitute(new_preds, theta):
    """
    given that a new clause has been created, substitute terms that have been
    found through the unify function
    :param new_preds: set of new predicates for the clause
    :param theta: a dict of pairs that can be used for substitutions
    for the expressions to make them the same
    :return:
    """
    # loop through each predicate
    for new_pred in new_preds:
        # loop through each term
        for i, term_obj in enumerate(new_pred.terms):
            term = term_obj.term
            if term in theta:
                new_term = copy.deepcopy(theta[term])
                new_obj = Term()
                if type(new_term) == Variable:
                    new_obj.term_type = TermType.VARIABLE
                elif type(new_term) == Constant:
                    new_obj.term_type = TermType.CONSTANT
                elif type(new_term) == Function:
                    new_obj.term_type = TermType.FUNCTION
                else:
                    print("term does not exist in KB")
                    sys.exit(1)
                new_obj.term = new_term
                new_pred.terms[i] = new_obj
            # term is function, need to check internal terms
            elif type(term) == Function:
                term.terms = substitute_func(term, theta)

def substitute_func(function, theta):
    """
    helper function to assist substituting terms for internal terms
    (a function within a predicate)
    :param function: the internal function
    :param theta: a dict of pairs that can be used for substitutions
    for the expressions to make them the same
    :return: the list of substituted terms for the function
    """
    # loop through each term
    func_terms = function.terms
    for i, term in enumerate(func_terms):
        if term in theta:
            new_term = copy.deepcopy(theta[term])
            new_obj = Term()
            if type(new_term) == Variable:
                new_obj.term_type = TermType.VARIABLE
            elif type(new_term) == Constant:
                new_obj.term_type = TermType.CONSTANT
            elif type(new_term) == Function:
                new_obj.term_type = TermType.FUNCTION
            else:
                print("term does not exist in KB")
                sys.exit(1)
            new_obj.term = term
            func_terms[i] = new_obj
        # term is function, need to check internal terms
        elif type(term) == Function:
            term.terms = substitute_func(term, theta)
    return func_terms

def check_complement_preds(new_preds):
    """
    given a new clause has been created, check that two
    complementary predicates do not exist in the new clause
    :param new_preds: the set of new predicates in the clause
    :return: True is no two complementary predicates exist,
    False otherwise
    """
    new_preds = list(new_preds)
    for p1 in new_preds:
        for p2 in new_preds:
            if p1 != p2 and is_complement(p1, p2):
                return False
    return True

def is_complement(pred1, pred2):
    """
    check if two predicates are complements to each other

    two predicates are complements of each other if they are
    opposite negations and have the same name
    :param pred1: the first predicate
    :param pred2: the second predicate
    :return: True if they are complements, false otherwise
    """
    if (pred1.is_negated != pred2.is_negated and
        pred1.pred_name == pred2.pred_name):
        return True
    return False

def is_subset(new_clauses, clauses_set):
    """
    given a new set of clauses created by resolution,
    check if the new set of clauses is subset to the original
    set of clauses
    :param new_clauses: the set of new clauses
    :param clauses_set: the set of original clauses
    :return: True if new set of clauses is subset,
    False otherwise
    """
    for new_clause in new_clauses:
        if new_clause not in clauses_set:
            # new clause doesn't exist, so can't be subset
            return False
    # all new clauses already exist in clause set
    return True

def unify_var(var, x, theta):
    """
    attempt to unify a variable given another term, x
    :param var: the variable to unify
    :param x: the other term to unify
    :param theta: a dict of pairs that can be used for substitutions
    for the expressions to make them the same
    :return: None if there is an issue with unifying, updating theta,
    or calling back Unify if var or x already exist in theta
    """
    # var already exists to be substituted
    if var in theta:
        value = theta[var]
        return unify(value, x, theta)
    # x already exists to be substituted
    elif x in theta:
        value = theta[x]
        return unify(var, value, theta)
    # failure if var occurs inside term x
    elif occur_check(var, x):
        return None
    else:
        if type(x) == Term and type(x.term) == Variable:
            new_var = create_new_free_var()
            theta[var] = new_var
        else:
            theta[var] = x.term
        return theta

def occur_check(var, x):
    """
    check if the variable already occurs in x
    x must be a complex term for this to happen
    :param var: the variable to be tested on
    :param x: the term to test on
    :return: True if the term is present in x, False otherwise
    """
    # only complex term can be a function
    if type(x) == Function:
        # get x's terms
        x_terms = x.terms
        for x_term in x_terms:
            if var == x_term:
                return True
    return False

def create_new_free_var():
    """
    for two new variables in unification, create a new
    free variable so naming conflicts do not occur
    :return: a new variable object
    """
    global NEW_VAR_NUMBER
    new_var = Variable()
    new_var.string_rep = NEW_VAR_LETTER + str(NEW_VAR_NUMBER)
    NEW_VAR_NUMBER += 1
    variables[new_var.string_rep] = new_var
    return new_var

def main():

    if len(sys.argv) < 2:
        print("Usage: python3 lab2.py KB.cnf")
        return
    knowledge_base = sys.argv[1]

    # knowledge_base = "universals+constants/uc01
    #
    # .cnf"
    resolution(knowledge_base)

if __name__ == "__main__":
    main()