You will be given a single file encoding a knowledge base (KB) in conjunctive normal form (CNF), for example. The file will first list the predicates, variables, constants, and functions used in the knowledge base. It will then list the clauses on each line with whitespace separating each literal. Negation will be indicated by "!". For instance, the clauses may read:

!dog(x0) animal(x0)  
dog(Kim1)

Which represents two clauses:

[¬dog(x0), animal(x0)]  
[dog(Kim1)]

Which can be expressed in FOL as:

(∀ x ¬dog(x) ∨ animal(x)) ∧ dog(Kim1)

Your program should then determine if the KB is satisfiable and print either, "yes" (it is satisfiable; it cannot find any new clauses) or "no" (it is not satisfiable; it finds the empty clause).

Input:  
python lab2.py [knowledge_base]

Example Input:  
python lab2.py testcases/functions/f1.cnf

