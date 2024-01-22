# Software Development Life Cycle (SDLC)

## Introduction to Software Development Life Cycle (SDLC)

In the dynamic realm of the software industry, the Software Development Life Cycle (SDLC) emerges as a pivotal and systematic process encompassing various stages: designing, developing, testing, and releasing software. Its ultimate objective is to deliver software of the highest quality, aligning with customer expectations. Guiding this intricate process is the ISO/IEC standard 10207, which meticulously defines software lifecycle processes.

## Phases of SDLC

### 1. Planning and Requirements Definition

The initial phase involves a meticulous identification of development goals, stakeholders, and feasibility studies. Rigorous analysis, validation, and documentation of requirements take precedence. A comprehensive project plan is then crafted, incorporating timelines and resource allocation.

### 2. Design and Architecture

This phase delves into the intricate details of software modules and internals. Designing identifies these modules, while architecture defines module connections, operating systems, databases, and user interface aspects. Feasibility studies and system-level test cases are conducted, culminating in the creation of design and architecture documents.

### 3. Coding (Development) Phase

Implementation of low-level design in adherence to coding guidelines takes center stage in this phase. Developers, in turn, conduct unit testing, while project management tools meticulously track progress. The output comprises executable code, comprehensive documentation, and meticulously crafted unit test cases.

### 4. Testing Phase

The testing phase is a critical juncture where software undergoes thorough examination for defects. This includes integration testing, system testing, and acceptance testing. The iterative process of defect identification, rectification, and retesting continues until all functionalities meet the defined criteria. The output comprises detailed test cases and comprehensive test documentation.

### 5. Maintenance Phase

Post-deployment, the maintenance phase kicks in, addressing errors post-release and accommodating customer feature requests. Regression testing ensures continued software integrity, with both reusing and creating new test cases as necessary.

## SDLC Models

### 1. V Model

The V Model stands out for its emphasis on testing, incorporating both verification and validation. It follows a traditional waterfall model, mapping testing phases directly to corresponding development phases. This model places a premium on thorough testing practices.

### 2. Agile Software Development

An amalgamation of methodologies, Agile Software Development prioritizes adaptability and rapid development. This involves developing in small, manageable subsets with incremental releases, fostering quick delivery, customer interactions, and rapid response. Agile models often include iterations or sprints.

## Other SDLC Models

Beyond the V Model and Agile, the software industry features a myriad of other SDLC models, each with its unique approach. Models like Big Bang, Rapid Application Development, Incremental Model, and the Waterfall Model cater to diverse project requirements and circumstances.

## Umbrella Activities

### 1. Project Management

Integral to SDLC are umbrella activities, including project management. This involves team management, task delegation, resource planning, duration estimation, intermediate releases, and overall project planning.

### 2. Documentation

Documentation forms the backbone of SDLC, with essential artifacts encompassing code, test cases, and various documents. The Requirements Traceability Matrix (RTM) emerges as a crucial tool, linking artifacts across different phases and ensuring a seamless flow of information.

### 3. Quality Assurance

Ensuring the readiness of software for the market involves dedicated efforts from software quality auditors, inspection teams, and certification and accreditation teams. Quality Assurance activities play a vital role in maintaining the overall integrity and reliability of the software product.

# Introduction to Terminology and Types

## Introduction

### Definition of Software Testing
Software testing is a comprehensive process involving the scrutiny of various artifacts, including code, design, architecture documents, and requirements documents. The core objective is to validate and verify these artifacts, ensuring the software's reliability and functionality.

### Goals of Software Testing
The overarching goals encompass providing an unbiased, independent assessment of the software, verifying its compliance with business capabilities, and evaluating associated risks that may impact its performance.

## Standard Glossary

- **Verification:** This process determines whether the products meet specified requirements at various stages of the software development life cycle.
  
- **Validation:** Evaluation of the software at the end of the development phase, ensuring it aligns with standards and intended usage.

- **Fault:** A static defect within the software, often originating from a mistake made during development.

- **Failure:** The visible, external manifestation of incorrect behavior resulting from a fault.

- **Error:** The incorrect state of the program when a failure occurs, indicating a deviation from the intended behavior.

## Historical Perspective

Drawing from the historical lens, luminaries like Edison and Lovelace utilized terms such as "bug" and "error" to emphasize the iterative process of identifying and rectifying faults and difficulties in inventions.

## Testing Terminology

- **Test Case:** A comprehensive entity comprising test inputs and expected outputs, evaluated by executing the test case on the code.

- **Test Case ID:** An identifier crucial for retrieval and management of test cases.

- **Traceability:** The establishment of links connecting test cases to specific requirements, ensuring thorough validation.

## Types of Testing

1. **Unit Testing:** A meticulous examination carried out by developers during the coding phase to test individual methods.

2. **Integration Testing:** An evaluation of the interaction between diverse software components.

3. **System Testing:** A holistic examination of the entire system to ensure alignment with design requirements.

4. **Acceptance Testing:** Conducted by end customers to validate that the delivered software meets all committed requirements.

## Quality Parameters Testing

- **Functional Testing:** Ensures the software functions precisely as intended.

- **Stress Testing:** Evaluates software performance under extreme conditions to assess its robustness.

- **Performance Testing:** Verifies if the software responds within specified time limits under varying conditions.

- **Usability Testing:** Ensures the software offers a user-friendly interface, enhancing the overall user experience.

- **Regression Testing:** Validates that existing functionalities continue to work seamlessly after software changes.

## Methods of Testing

- **Black Box Testing:** A method that evaluates the software without delving into its internal structure, relying solely on inputs and requirements.

- **White Box Testing:** Testing carried out with a comprehensive understanding of the software's internal structure, design, and code.

- **Gray Box Testing:** An intermediate approach that combines elements of both black box and white box testing.

## Testing Activities

1. **Test Case Design:**
   - Critical for efficiently identifying defects.
   - Requires a blend of computer science expertise, domain knowledge, and mathematical proficiency.
   - Emphasis on the development of effective test case design algorithms.
\newpage

2. **Test Automation:**
   - Involves the conversion of test cases into executable scripts.
   - Addresses preparatory steps and incorporates concepts of observability and controllability.
   - Utilizes both open-source and proprietary test automation tools.

3. **Execution:**
   - Automated process involving the execution of test cases.
   - Utilizes a selection of open-source or proprietary tools chosen by the organization.

4. **Evaluation:**
   - The critical analysis of test results to determine correctness.
   - Manual intervention may be required for fault isolation.
   - Crucial for drawing inferences about the software's quality.

# Testing Goals and Process Levels

## Introduction
In the realm of software testing, the pursuit of testing goals is intricately tied to the specificities of the software product in question and the maturity of an organization's quality processes. This diversity in objectives and approaches underscores the importance of comprehending the nuanced landscape of testing process levels, which range from the rudimentary Level 0 to the pinnacle of maturity at Level 4.

## Testing Process Maturity Levels

1. **Level 0: Low Maturity**

   At this embryonic stage, there is an absence of a clear demarcation between testing and debugging activities. The predominant focus revolves around expedient product releases, potentially at the expense of a rigorous testing regimen.

2. **Level 1: Testing for Correctness**

   The next tier witnesses a paradigm shift as testing endeavors to validate software correctness. However, a common misunderstanding prevails — an attempt to prove complete correctness through testing, an inherently unattainable feat.

3. **Level 2: Finding Errors**

   As organizations ascend to Level 2, there is a conscious recognition of testing as a mechanism to unearth errors by actively showcasing failures. However, a resistance lingers when it comes to acknowledging and addressing errors identified in the code.

4. **Level 3: Sophisticated Testing**

   Level 3 marks a watershed moment where testing is not merely a reactive measure but is embraced as a robust technique for both identifying and eliminating errors. A collaborative ethos emerges, with a collective effort to mitigate risks in software development.

5. **Level 4: Mature Process-Oriented Testing**

   At the pinnacle of maturity, testing transcends mere procedural activities; it metamorphoses into a mental discipline. Integrated seamlessly into mainstream development, the focus is on continuous quality improvement. Here, test engineers and developers synergize their efforts to deliver software of the highest quality.

## Significance for the Course

Understanding the nuances of testing process levels assumes paramount importance as it serves as the bedrock for tailoring testing approaches. The focus of this course is strategically directed towards the technical intricacies relevant to Levels 3 and 4, where testing is not just a process but an integral aspect of the software development mindset.

## Controllability and Observability

- **Controllability:** This pertains to the ability to provide inputs and execute the software module. It underscores the necessity of having a structured approach to govern the input parameters and execution environment.
  
- **Observability:** The study and recording of outputs form the crux of observability. This involves a meticulous examination of the software's responses, contributing significantly to the overall understanding of its behavior.

## Illustration

The challenges of controllability and observability find illustration in real-world scenarios. Designing effective test cases becomes paramount to ensure both the reachability of various modules and the meticulous observation of their outputs. This practical application reinforces the theoretical concepts discussed in the course.

## Test Automation Tool: JUnit

The course introduces JUnit as the designated test automation tool. JUnit's utility is elucidated through a discussion of its prefix and postfix annotations, providing a structured approach to manage controllability and observability. Subsequent classes delve into both the theoretical underpinnings and the hands-on application of JUnit, ensuring a comprehensive understanding of its role in the testing process.

# Conclusion

In this comprehensive exploration of Software Development Life Cycle (SDLC) and Software Testing, we've delved into the intricacies of the development process, testing methodologies, and the critical role of quality assurance. The SDLC, guided by the ISO/IEC standard 10207, unfolds through phases like Planning, Design, Coding, Testing, and Maintenance. Various SDLC models, such as the V Model and Agile, cater to diverse project requirements, emphasizing thorough testing practices and adaptability.

The umbrella activities of Project Management, Documentation, and Quality Assurance form the backbone of SDLC, ensuring seamless project execution, comprehensive documentation, and software reliability. The introduction to terminology and types in software testing, along with a historical perspective, offers a holistic view of the discipline.

The lecture further explores testing terminology, types of testing, quality parameters testing, and methods of testing. It sheds light on crucial aspects like test case design, test automation, execution, and evaluation. The discussion on Testing Goals and Process Levels provides insights into the diverse maturity levels, ranging from Level 0 to Level 4, where testing transforms from a reactive measure to a mature, process-oriented discipline.

# Points to Remember

1. **SDLC Phases:**
   - Planning and Requirements Definition
   - Design and Architecture
   - Coding (Development) Phase
   - Testing Phase
   - Maintenance Phase

2. **SDLC Models:**
   - V Model: Emphasis on testing, verification, and validation.
   - Agile Software Development: Prioritizes adaptability and rapid development.

3. **Umbrella Activities:**
   - Project Management: Team management, task delegation, resource planning.
   - Documentation: Essential artifacts, Requirements Traceability Matrix (RTM).
   - Quality Assurance: Ensures software readiness and overall integrity.

4. **Testing Goals:**
   - Validation and verification of artifacts.
   - Unbiased, independent assessment of software.
   - Evaluation of compliance with business capabilities.

5. **Testing Process Levels:**
   - Level 0 to Level 4, representing increasing maturity.
   - Level 4 signifies a mature, process-oriented testing mindset.

6. **Controllability and Observability:**
   - Controllability: Structured approach to input parameters and execution.
   - Observability: Meticulous examination of software responses.

7. **Test Automation Tool: JUnit:**
   - Designated tool for test automation.
   - Prefix and postfix annotations for structured controllability and observability.