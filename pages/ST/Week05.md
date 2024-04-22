---
title: "Logic Coverage Criteria in Software Testing"
---

These notes provide a detailed exploration of logic coverage criteria, their application in software testing, and their significance in ensuring thorough testing of complex systems. We delve into the theoretical foundations of logic, explore various coverage criteria, and examine their practical implementation through illustrative examples.

# Introduction to Logic: Building Blocks for Reasoning

## Propositional Logic: Combining Truth Values

* **Atomic Propositions:** Declarative sentences that are either true or false. Examples include "The sky is blue" or "2 + 2 = 4".
* **Logical Connectives:** Operators that combine atomic propositions to form more complex statements. Common connectives include:
    * **∨ (Disjunction/OR):** True if at least one of the propositions is true.
    * **∧ (Conjunction/AND):** True only if all propositions are true.
    * **¬ (Negation/NOT):** Reverses the truth value of a proposition.
    * **⊃ (Implication):** True unless the first proposition is true and the second is false.
    * **≡ (Equivalence/IFF):** True only if both propositions have the same truth value.
* **Formulas:** Combinations of propositions and connectives that represent logical statements.
* **Truth Tables:** Tools for systematically evaluating the truth value of a formula under all possible combinations of truth values for its atomic propositions.
* **Satisfiability:** A formula is satisfiable if there exists at least one assignment of truth values to its propositions that makes the formula true.
* **Validity:** A formula is valid (a tautology) if it is true under all possible assignments of truth values to its propositions.

## Predicate Logic: Reasoning with Variables and Functions

* **Predicates:** Expressions involving variables and functions that evaluate to true or false. Examples include "x > y" or "isEven(n)".
* **Clauses:** Predicates that do not contain any logical operators.
* **Satisfiability in Predicate Logic:** Checking whether a predicate can be made true by assigning appropriate values to its variables and functions. This problem is generally undecidable, but SAT/SMT solvers can handle many practical cases.

# Logic Coverage Criteria: Evaluating Test Thoroughness

## Predicate Coverage (PC)

* **Definition:** Requires that each predicate in the program be evaluated to both true and false during testing.
* **Relation to Graph Coverage:** Equivalent to edge coverage when predicates are associated with branches in the control flow graph.
* **Example:** For the predicate `(x > y) ∨ C ∨ f(z)`, PC would require tests where the predicate is true (e.g., x=5, y=3, C=true, f(z)=false) and false (e.g., x=1, y=4, C=false, f(z)=false).

## Clause Coverage (CC)

* **Definition:** Requires that each individual clause within every predicate be evaluated to both true and false during testing.
* **Relation to PC:** Does not subsume PC. For instance, the predicate `a ∨ b` can satisfy CC with tests {a=T, b=F} and {a=F, b=T}, but PC is not satisfied as the predicate remains true in both cases.

## Combinatorial Coverage (CoC) / Multiple Condition Coverage

* **Definition:** Requires testing all possible combinations of truth values for the clauses within each predicate.
* **Feasibility:** Often impractical due to the exponential number of test cases required (2^n for n clauses).

## Active Clause Coverage (ACC): Targeting Influential Clauses

* **Motivation:** Focuses on situations where a specific clause ("major clause") determines the outcome of the predicate, regardless of the values of other clauses ("minor clauses").
* **Determination:** A major clause `ci` determines a predicate `p` if changing the truth value of `ci` changes the truth value of `p`, while keeping the values of minor clauses fixed.
* **General Forms of ACC:**
    * **General Active Clause Coverage (GACC):** Requires testing each clause as a major clause, ensuring it determines the predicate's outcome under some combination of minor clause values.
    * **Correlated Active Clause Coverage (CACC):** Like GACC, but requires that the chosen minor clause values result in the predicate being true for one value of the major clause and false for the other.
    * **Restricted Active Clause Coverage (RACC):** Like CACC, but further restricts the minor clause values to be the same for both true and false evaluations of the major clause.
* **Relationship between ACC forms:** CACC subsumes RACC, and GACC subsumes both CACC and RACC.
* **Example (CACC):** For the predicate `a ∧ (b ∨ c)`, CACC would require tests covering combinations such as:
    * `a=T, b=T, c=T` (a is the major clause, b ∨ c is true)
    * `a=F, b=T, c=T` (a is the major clause, b ∨ c is true, but the predicate is false)

## Inactive Clause Coverage (ICC): Ensuring Clause Independence

* **Motivation:** Complements ACC, focusing on situations where a clause does not influence the predicate's outcome.
* **General Forms of ICC:**
    * **General Inactive Clause Coverage (GICC):** Requires testing each clause as a major clause, ensuring it does not determine the predicate's outcome under any combination of minor clause values.
    * **Restricted Inactive Clause Coverage (RICC):** Like GICC, but restricts the minor clause values to be the same across tests where the major clause is true and where it is false, while the predicate remains true and false respectively.

# Applying Logic Coverage to Source Code: Challenges and Considerations

* **Reachability:** Ensuring that the code under test can be reached with specific input values. Symbolic execution and other techniques can aid in achieving reachability.
* **Controllability:** Ensuring that input values can be used to indirectly assign desired values to variables within predicates, especially internal variables not directly controlled by inputs.
* **Example (Thermostat):** Consider the predicate `p = (a ∨ (b ∧ c)) ∧ d`, derived from a thermostat program:
    * Clause `a` involves the internal variable `dTemp`. We need to find ways to control `dTemp` through input settings (e.g., setting period and day of the week).
    * Test cases for CACC would involve manipulating input values to achieve specific combinations of truth values for the clauses, ensuring that each clause acts as a major clause that determines the outcome of the predicate.

# Subsumption Relations and Infeasibility

* **Subsumption Hierarchy:** CoC subsumes all other criteria; CACC subsumes RACC and PC; GACC subsumes CACC and RACC.
* **Infeasibility:** Certain criteria, particularly RACC, can be infeasible in practical scenarios due to constraints on the system or the nature of the logical expressions.
* **Dealing with Infeasibility:** Ignoring infeasible requirements, considering counterparts in subsumed criteria, or employing best-effort strategies.

# Conclusion

Logic coverage criteria offer a valuable framework for designing test cases that systematically evaluate the behavior of complex predicates and ensure thorough testing of software systems. Understanding the theoretical underpinnings of logic, the various coverage criteria, and their practical application allows testers to effectively assess and improve the quality and reliability of software.
