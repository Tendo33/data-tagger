def input_difficulty_rating(input: str) -> str:
    user_message = f"""
# Task: User Query Analysis

You are an expert query analyzer. Your task is to carefully analyze the given user query. The ultimate goal of this analysis is to route the query to the most appropriate specialist agent or knowledge base. You need to identify the user's intent, determine the required knowledge, and assess the query's difficulty.

## User Query
```

{input}

```
## Analysis Instructions:
1.  **Identify Intent**: Clearly describe what the user is trying to accomplish or what question they're asking. Look for both explicit requests and implied needs.
2.  **Determine Knowledge**: Pinpoint the specific domains, concepts, or information required to fully address the query. Be specific in your keywords.
3.  **Rate Difficulty**: Rate the query's difficulty on a scale of 0 to 5, based on the complexity of reasoning and the specificity of knowledge required. **You may use decimal values (e.g., 2.5, 3.7, etc.) to indicate intermediate difficulty levels.**
    * 0: Extremely simple, requires no specialized knowledge (e.g., "hello").
    * 1: Basic, requires minimal general knowledge (e.g., "what is the capital of France?").
    * 2: Straightforward, requires some common knowledge (e.g., "explain photosynthesis in simple terms").
    * 3: Moderate, requires solid domain knowledge (e.g., "how to implement a singleton pattern in Python?").
    * 4: Complex, requires advanced knowledge and reasoning (e.g., "compare the economic impacts of Keynesian vs. Austrian school theories").
    * 5: Expert-level, requires deep, specialized knowledge and complex reasoning (e.g., "devise a novel algorithm for protein folding prediction").
4.  **Handle Ambiguity**: If the query is too vague, ambiguous, or nonsensical to be analyzed, set the intent to "ambiguous query" and the difficulty to 3.0.

## Output Format
Given the user query, in your output, you first need to identify the user intent and the knowledge needed to solve the task in the user query.
Then, rate the difficulty level of the user query as a float number between 0 and 5 (decimals allowed).

Now, please output the user intent and difficulty level below in a JSON format by filling in the placeholders in []:

```
{{   
    "intent": "The user wants to [....]",
    "knowledge": "To solve this problem, the models need to know [....]",
    "difficulty": "[0-5, float, decimals allowed]"
}}
```

## Examples

### Example 1
**User Query**: "Can you tell me how to build a simple to-do list app using React and TypeScript? I need to know the basic components and state management."
**Output**:

{{
    "intent": "The user wants a step-by-step guide or tutorial on creating a to-do list application using React with TypeScript, specifically asking for component structure and state management techniques.",
    "knowledge": "Requires knowledge of web development, specifically: React.js library, TypeScript language, front-end component architecture, and state management principles (e.g., useState, useReducer).",
    "difficulty": "3.0"
}}


### Example 2

**User Query**: "what's the weather like in tokyo tomorrow"
**Output**:

{{
    "intent": "The user is asking for the weather forecast for Tokyo for the next day.",
    "knowledge": "Requires access to real-time weather forecast data services and knowledge of the geographical location of Tokyo.",
    "difficulty": "0.5"
}}

## Output Format & Constraints

Your response **MUST** be a single, valid JSON object and nothing else. Do not include any explanatory text before or after the JSON.

"""
    return user_message


def input_classification(input: str) -> str:
    user_message = f"""
# IDENTITY AND GOAL
You are an expert AI assistant specializing in query analysis and task classification. Your goal is to accurately categorize a user's query based on their primary intent. This classification will be used to route the query to the most appropriate specialized agent.

# TASK DESCRIPTION
Analyze the user query provided in the `<query_to_classify>` block. Based on your analysis, you will assign a `primary_tag` and, if applicable, a list of `other_tags`.

- The `primary_tag` MUST represent the user's **main intent** or the **dominant action** required to fulfill the request.
- The `other_tags` list should include any secondary tasks or aspects present in the query.
- You MUST select tags exclusively from the `<available_tags>` list.
- Your final output MUST be a single, valid JSON object and nothing else.

# AVAILABLE TAGS
<available_tags>
[
    "Information seeking",      # Users ask for specific information or facts about various topics.
    "Reasoning",                # Queries require logical thinking, problem-solving, or processing of complex ideas.
    "Planning",                 # Users need assistance in creating plans or strategies for activities and projects.
    "Editing",                  # Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
    "Coding & Debugging",       # Users seek help with writing, reviewing, or fixing code in programming.
    "Math",                     # Queries related to mathematical concepts, problems, and calculations.
    "Role playing",             # Users engage in scenarios requiring the AI to adopt a character or persona.
    "Data analysis",            # Requests involve interpreting data, statistics, or performing analytical tasks.
    "Creative writing",         # Users seek assistance with crafting stories, poems, or other creative texts.
    "Advice seeking",           # Users ask for recommendations or guidance on various personal or professional issues.
    "Translation",              # Users ask for translation of text from one language to another.
    "Brainstorming",            # Involves generating ideas, creative thinking, or exploring possibilities.
    "Others"                    # Any queries that do not fit into the above categories or are of a miscellaneous nature.
]
</available_tags>

# EXAMPLES
<examples>
1.  **User Query**: "Help me plan a 4-day trip to Tokyo and find some good, cheap ramen spots."
    **Output**:
    ```json
    {{
        "primary_tag": "Planning",
        "other_tags": ["Information seeking", "Advice seeking"]
    }}
    ```

2.  **User Query**: "Can you write a python script to parse a CSV file and then explain how it works?"
    **Output**:
    ```json
    {{
        "primary_tag": "Coding & Debugging",
        "other_tags": ["Information seeking"]
    }}
    ```

3.  **User Query**: "Write a short, sad poem about autumn, then rephrase it to sound more hopeful."
    **Output**:
    ```json
    {{
        "primary_tag": "Creative writing",
        "other_tags": ["Editing"]
    }}
    ```
</examples>

# QUERY TO CLASSIFY
```
{input}
```

# OUTPUT
```
{{

    "primary_tag": "<primary tag>",

    "other_tags": ["<tag 1>", "<tag 2>", ... ]

}}
```
Please provide your response in the specified JSON format.
"""
    return user_message


def input_quality_rating(input: str) -> str:
    user_message = f"""
# Role & Goal

    You are a meticulous Query Quality Analyst. Your goal is to score a user's query based on a rigorous, quantitative framework and provide a concise justification for your scoring.

# Scoring Criteria

    You must evaluate the user query against the following three criteria, each on a scale of 1 to 5 (where 1 is the worst and 5 is the best, and decimals are allowed).

    1.  **Clarity (1-5)**: How clear and grammatically correct is the query? Is the user's intent easily understandable without ambiguity?
        * 1-2: Very confusing, full of errors, intent is impossible to grasp.
        * 2.1-3: Moderately clear, but has some ambiguities or awkward phrasing.
        * 3.1-4: Mostly clear and well-phrased.
        * 4.1-5: Perfectly clear, concise, and grammatically flawless.

    2.  **Specificity (1-5)**: Does the query provide enough specific details, context, and constraints for an AI to generate a high-quality, relevant response?
        * 1-2: Extremely vague, lacks all necessary context or detail.
        * 2.1-3: Contains a general topic but misses key details, constraints, or format requirements.
        * 3.1-4: Reasonably specific, providing most of the necessary information.
        * 4.1-5: Highly specific, providing all necessary context, examples, constraints, and desired output format.

    3.  **Coherence (1-5)**: Are the different parts of the query logically connected? Does it represent a single, well-defined goal?
        * 1-2: Incoherent, contains contradictory requests or multiple unrelated questions.
        * 2.1-3: Mostly coherent, but parts of the query may not align perfectly.
        * 3.1-4: Coherent and focused on a single goal.
        * 4.1-5: Perfectly coherent, with all elements working together to define a precise task.

# User Query
    ```
    {input}
    ```

# Task & Output Format

    1.  **Analyze** the user query based on the criteria above.
    2.  **Provide a brief assessment** in the `input_quality_explanation` field, justifying your scores for each criterion.
    3.  **Calculate the final score** by taking the average of the three criteria scores, rounded to one decimal place. The final_score should be a float number between 1 and 5.
    4.  **Output the results** in the following JSON format. **DO NOT** output anything other than the JSON object.

    ```
    {{
        "input_quality_explanation": "[Your brief analysis justifying the scores...]",
        "scores": {{
        "clarity": "[1-5]",
        "specificity": "[1-5]",
        "coherence": "[1-5]"
        }},
        "input_quality": "[1-5]"
    }}
    ```
"""
    return user_message


def response_quality_rating(query: str, response: str) -> str:
    user_message = f"""
# Role & Goal

You are an exacting AI Response Quality Auditor. Your purpose is to meticulously evaluate an AI's response based on its original query, using a quantitative scoring system. Your feedback must be precise, objective, and actionable.

# Context: The Original User Query
{query}
# Context: The AI's Response to Evaluate
{response}
# Scoring Criteria

You must evaluate the AI's response against the following four criteria, each on a scale of 1 to 5 (where 1 is worst, 5 is best, and decimals are allowed). The evaluation **MUST** be based on the context of the original user query.

1.  **Accuracy (1-5)**: Is the information provided in the response factually correct and free of errors?
    * 1-2: Contains significant factual errors or fabrications.
    * 2.1-3: Mostly accurate but contains minor inaccuracies or unverified claims.
    * 3.1-4: Accurate and reliable.
    * 4.1-5: Flawlessly accurate, precise, and well-sourced if applicable.

2.  **Completeness (1-5)**: Does the response fully address all explicit and implicit parts of the user's query?
    * 1-2: Fails to address the core question.
    * 2.1-3: Addresses the main question but misses significant sub-points or nuances.
    * 3.1-4: Addresses all explicit parts of the query thoroughly.
    * 4.1-5: Comprehensively answers the query, anticipating follow-up questions and covering all angles.

3.  **Clarity (1-5)**: Is the response well-structured, easy to understand, and free of jargon?
    * 1-2: Incoherent, confusing, or poorly written.
    * 2.1-3: Understandable, but the structure or language could be improved.
    * 3.1-4: Clear, well-organized, and easy to follow.
    * 4.1-5: Exceptionally clear, elegantly structured, and perfectly articulated for the target audience.

4.  **Helpfulness (1-5)**: How effectively does the response help the user achieve their goal? Is it practical and actionable?
    * 1-2: Useless or even counterproductive.
    * 2.1-3: Provides some value but is not directly helpful or actionable.
    * 3.1-4: Helpful and effectively solves the user's problem or answers their question.
    * 4.1-5: Extremely helpful, providing exceptional value, actionable steps, and empowering the user.

# Task & Output Format

1.  **Analyze** the `response` in the context of the `query`.
2.  **Provide a brief assessment** in the `response_quality_explanation` field, justifying your scores for each of the four criteria.
3.  **Calculate the final score** by averaging the four criteria scores, rounded to one decimal place. The final_score should be a float number between 1 and 5.
4.  **Output the results** in the following JSON format. You must only output the JSON object.

```
{{
    "response_quality_explanation": "[Your brief analysis justifying the scores for each criterion...]",
    "scores": {{
        "accuracy": "[1-5]",
        "completeness": "[1-5]",
        "clarity": "[1-5]",
        "helpfulness": "[1-5]"
    }},
    "response_quality": "[1-5]"
}}
```

"""
    return user_message


def combined_quality_rating(query: str, response: str) -> str:
    user_message = f"""
# Role & Goal

You are a highly analytical Conversation Quality Judge. Your mission is to conduct a comprehensive, quantitative audit of a full user-AI interaction. You must evaluate both the user's initial query and the AI's corresponding response with objectivity and precision, based on the established criteria.

# Interaction to Evaluate

## 1. User Input
{query}

## 2. AI Response
{response}

# Evaluation Framework

You will conduct two separate evaluations.

---

### **Part 1: Input Quality Evaluation**
Score the **User Input** on a 1-5 scale for each criterion:

* **Clarity (1-5)**: Is the language clear and unambiguous?
* **Specificity (1-5)**: Does it provide enough specific detail and context?
* **Coherence (1-5)**: Is the goal well-defined and internally consistent?

---

### **Part 2: Response Quality Evaluation**
Score the **AI Response** (in the context of the User Input) on a 1-5 scale for each criterion:

* **Accuracy (1-5)**: Is the information factually correct?
* **Completeness (1-5)**: Does it fully address all parts of the input?
* **Clarity (1-5)**: Is the response well-structured and easy to understand?
* **Helpfulness (1-5)**: How effectively does it help the user achieve their goal?

---

# Task & Output Format

1.  Perform the two evaluations based on the framework above.
2.  For each part, write a brief explanation justifying your scores.
3.  For each part, calculate a `final_score` by averaging its criteria scores (rounded to one decimal place).
4.  For each part, final_score should be a float number between 1 and 5.
5.  Combine both evaluations into a single JSON object as specified below. **You must only output this JSON object.**

```
{{   
    "input_quality": "[1-5]",
    "response_quality": "[1-5]",
    "input_quality_explanation": "[Detailed explanation of input quality assessment...]",
    "response_quality_explanation": "[Detailed explanation of response quality assessment...]"
}}
```
"""
    return user_message
