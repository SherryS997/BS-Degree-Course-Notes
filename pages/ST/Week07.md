---
title: "Symbolic Testing and DART"
---

These notes delve into the concepts of symbolic testing and DART (Directed Automated Random Testing), providing a comprehensive understanding of these techniques and their applications in software testing.

# Symbolic Testing

**Introduction:**

Symbolic testing is a powerful approach to software testing that analyzes programs by considering input values as symbolic variables rather than concrete data. This allows exploration of various execution paths and identification of potential issues without exhaustively testing every possible input combination.

**Key Concepts:**

* **Symbolic Execution:**
    * Analyzes a program by considering input values as symbolic variables (e.g., α1, α2) instead of concrete data (e.g., 1, 2).
    * Tracks how symbolic values propagate through the program, representing variables as expressions of these symbols.
    * Enables exploring different execution paths and analyzing program behavior under various input scenarios.
* **Path Condition (PC):**
    * A logical formula representing constraints on symbolic input values for a specific execution path.
    * It captures the conditions under which a particular path is taken, expressed as a conjunction of predicates.
* **Symbolic State (σ):**
    * A mapping between program variables and their corresponding symbolic expressions at a given point in the execution.
    * It reflects the current state of the program in terms of symbolic values.

**Steps in Symbolic Execution:**

1. **Initialization:**
    * Symbolic state (σ) is set to an empty map.
    * Path condition (PC) is set to true.
2. **Read Statements:**
    * When encountering a read statement, a new symbolic variable is introduced and mapped to the variable being read in σ.
3. **Assignment Statements:**
    * For assignments like `v = e`, the symbolic expression of `e` is evaluated in the current symbolic state and the result is mapped to `v` in σ.
4. **Conditional Statements:**
    * For `if (e) then S1 else S2`:
    * **"Then" Branch:** PC is updated to `PC ∧ σ(e)` and symbolic execution continues with this updated PC and σ.
    * **"Else" Branch:** A new path condition PC' is created as `PC ∧ ¬σ(e)` and symbolic execution branches out with PC' and a copy of σ.
5. **Termination:**
    * Symbolic execution ends when:
    * Reaching an exit statement.
    * Encountering an error.
    * PC becomes unsatisfiable (no concrete values can satisfy the constraints).

**Generating Test Cases:**

* After exploring a path, the final PC is solved using a constraint solver to find concrete input values satisfying the constraints.
* These concrete values are used as test inputs to exercise the specific execution path explored during symbolic execution.

**Advantages:**

* **Efficiency:** A single symbolic execution can represent numerous concrete test cases, improving test coverage efficiently.
* **Thoroughness:** Explores different execution paths, including corner cases, which might be missed with traditional testing methods.
* **Automation:** Can be automated to generate test cases and analyze program behavior.

**Disadvantages:**

* **Path Explosion:** Complex programs may have a vast number of paths, making exhaustive exploration infeasible.
* **Constraint Solving Challenges:** Certain path conditions may be complex or unsolvable by constraint solvers.
* **Limited Scope:** May struggle with programs involving external libraries, system calls, or complex data structures.

**Modern Advancements:**

* **Advanced Solvers:** Modern SMT solvers can handle complex constraints, improving the effectiveness of symbolic testing.
* **Hybrid Techniques:** Concolic testing combines concrete and symbolic execution to overcome limitations of pure symbolic execution.
* **Tool Support:** Tools like KLEE, CUTE, and PEX facilitate symbolic execution for various programming languages.

# DART (Directed Automated Random Testing)

**Overview:**

DART is a concolic testing tool that automates unit testing by combining random testing with symbolic execution. It aims to achieve high code coverage by directing test case generation towards unexplored execution paths.

**Key Features:**

* **Automatic Test Driver Generation:** DART eliminates the need for manual test driver development by automatically creating a driver that interacts with the program's interface and simulates its external environment.
* **Concolic Execution:** DART performs both concrete and symbolic execution simultaneously. It starts with random inputs and uses symbolic execution to gather path constraints. These constraints are then used to generate new input values that force the program along different execution paths.
* **Directed Search:** DART uses a "stack" to track the history of branch decisions made during execution. By analyzing this history and negating relevant path constraints, DART strategically generates new input values to explore previously unexplored paths.
* **Error Detection:** DART can detect various errors, including program crashes, assertion violations, and non-termination.

**DART Algorithm:**

1. **Interface Extraction:** DART parses the source code to identify the program's external interface, including external variables, functions, and input parameters.
2. **Random Test Driver Generation:**
    * A test driver is automatically created, which randomly initializes input values and calls the program's functions.
    * External functions are simulated to provide random return values.
3. **Concolic Execution:**
    * The program is executed on the random inputs, while simultaneously gathering path constraints symbolically.
    * Symbolic expressions are evaluated to track how values propagate through the program.
4. **Directed Search:**
    * The "stack" maintains the history of branch decisions.
    * When a path is explored, DART analyzes the stack and negates relevant predicates in the path constraint to guide the program towards a new path.
    * A constraint solver is used to find input values that satisfy the modified path constraints, leading to new test cases.

**Example:**
Consider the function `h` below:

```c
int f(int x) { return 2*x; }
int h(int x, int y) {
    if (x != y)
        if (f(x) == x+10)
            abort(); /* error */
    return 0;
}
```

1. **Random Input:** DART generates random values for `x` and `y`, say `x = 26` and `y = 34`.
2. **Path Constraints:** The path constraints gathered are `x != y` and `2*x != x+10`.
3. **Negation and Solving:** DART negates the second constraint and solves `x != y && 2*x == x+10`, resulting in `x = 10` and `y = any value other than 10`.
4. **New Input:** The new input `(x = 10, y = 45)` forces the program to take the previously unexplored branch and reach the error state.

**Benefits of DART:**

* **Automation:** Reduces manual effort in test case generation and execution.
* **High Coverage:** Effectively explores different paths, improving code coverage.
* **Error Detection:** Can identify various types of errors during testing.

**Limitations of DART:**

* **Complexity:** May struggle with programs involving complex data structures, concurrency, or external dependencies.
* **Constraint Solving:** Similar to symbolic testing, DART relies on constraint solvers, which can encounter limitations with complex constraints.

**Overall, symbolic testing and DART offer valuable techniques for software testing, enabling efficient exploration of execution paths and enhancing test coverage. While they have limitations, ongoing research and advancements continue to improve their capabilities and applicability in various testing scenarios.**