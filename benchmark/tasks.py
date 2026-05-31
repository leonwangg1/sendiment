"""
tasks.py — the benchmark task suite.

DESIGN NOTE: there is no off-the-shelf benchmark for agent memory effect. So we
built one with intentional recurrence: tasks come in FAMILIES where solving one
correctly should make the next easier IF AND ONLY IF the agent retained a useful
lesson. This is the only way to honestly measure the memory effect.

Each task has:
  - id, family, prompt, the model's job (what to produce)
  - a grading rubric: criteria + weights
  - optional `lesson_hint`: the rule we hope the Sediment-enabled agent extracts

Families are designed so the LATER tasks in each family reward knowing the
EARLIER tasks' lesson. A memory-less agent will rediscover; a memory-having
agent should ace them.
"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Task:
    id: str
    family: str
    prompt: str            # what the user asks the test-agent to do
    rubric: list[dict]     # [{criterion, weight, check}]  weights sum to 1.0
    lesson_hint: Optional[str] = None  # the lesson we hope Sediment captures

# --------------------------------------------------------------------------- #
# FAMILY 1: Model-targeted prompt engineering                                  #
# Lesson to learn: Anthropic models prefer XML delimiters; OpenAI models prefer markdown.
# --------------------------------------------------------------------------- #
F1 = [
    Task(
        id="f1-1", family="model_delimiters",
        prompt="Write a short system prompt for Claude that asks it to summarize meeting notes. The summary should have three sections: decisions, action items, and open questions. Make the structure unambiguous to the model.",
        rubric=[
            {"criterion": "Uses XML-style delimiters (<decisions>, <action_items>, etc.) for section structure", "weight": 0.5},
            {"criterion": "Defines a clear role/persona for Claude", "weight": 0.25},
            {"criterion": "Specifies the three required sections", "weight": 0.25},
        ],
        lesson_hint="WHEN target model is Anthropic/Claude → PREFER XML-style delimiters for structure",
    ),
    Task(
        id="f1-2", family="model_delimiters",
        prompt="Write a short system prompt for Claude that extracts entities from a news article into categories: people, organizations, locations. Make the structure unambiguous.",
        rubric=[
            {"criterion": "Uses XML-style delimiters for the three entity categories", "weight": 0.5},
            {"criterion": "Defines a clear extraction task", "weight": 0.25},
            {"criterion": "Specifies the three required categories", "weight": 0.25},
        ],
        lesson_hint="WHEN target model is Anthropic/Claude → PREFER XML-style delimiters for structure",
    ),
    Task(
        id="f1-3", family="model_delimiters",
        prompt="Write a short system prompt for Claude that reviews a code diff and outputs issues by severity: blocking, major, minor. Make the structure unambiguous.",
        rubric=[
            {"criterion": "Uses XML-style delimiters for the three severity sections", "weight": 0.5},
            {"criterion": "Defines a code reviewer role", "weight": 0.25},
            {"criterion": "Specifies the three severity levels", "weight": 0.25},
        ],
        lesson_hint="WHEN target model is Anthropic/Claude → PREFER XML-style delimiters for structure",
    ),
]

# --------------------------------------------------------------------------- #
# FAMILY 2: Structured output discipline                                       #
# Lesson to learn: always append a self-verification checklist for JSON outputs.
# --------------------------------------------------------------------------- #
F2 = [
    Task(
        id="f2-1", family="json_verification",
        prompt="Write a prompt that asks the model to extract product information from a webpage and return it as JSON with fields: name, price, currency, in_stock. The output MUST be valid JSON every time, no exceptions.",
        rubric=[
            {"criterion": "Specifies the exact JSON schema with all four fields", "weight": 0.3},
            {"criterion": "Includes a self-verification step (e.g. 'before responding, verify the JSON is valid and contains all required fields')", "weight": 0.5},
            {"criterion": "Mentions handling missing data (null vs omitted)", "weight": 0.2},
        ],
        lesson_hint="WHEN output must be strict JSON → PREFER appending a self-verification checklist step",
    ),
    Task(
        id="f2-2", family="json_verification",
        prompt="Write a prompt that asks the model to classify a customer support ticket and return JSON with fields: category, urgency, sentiment, suggested_action. JSON must always be valid.",
        rubric=[
            {"criterion": "Specifies the exact JSON schema with all four fields", "weight": 0.3},
            {"criterion": "Includes a self-verification step before output", "weight": 0.5},
            {"criterion": "Constrains the field values (e.g. enum for urgency)", "weight": 0.2},
        ],
        lesson_hint="WHEN output must be strict JSON → PREFER appending a self-verification checklist step",
    ),
    Task(
        id="f2-3", family="json_verification",
        prompt="Write a prompt that asks the model to parse a meeting transcript and return JSON with attendees (array), key_decisions (array), and follow_up_items (array of objects with owner and due_date). JSON must always be valid.",
        rubric=[
            {"criterion": "Specifies the nested JSON schema correctly", "weight": 0.3},
            {"criterion": "Includes a self-verification step before output", "weight": 0.5},
            {"criterion": "Handles the nested follow_up_items structure", "weight": 0.2},
        ],
        lesson_hint="WHEN output must be strict JSON → PREFER appending a self-verification checklist step",
    ),
]

# --------------------------------------------------------------------------- #
# FAMILY 3: Few-shot example discipline                                        #
# Lesson to learn: include CONCRETE examples in few-shot prompts, never placeholders.
# --------------------------------------------------------------------------- #
F3 = [
    Task(
        id="f3-1", family="few_shot_concrete",
        prompt="Write a few-shot prompt that teaches the model to classify movie reviews as positive, negative, or neutral.",
        rubric=[
            {"criterion": "Includes at least 2 fully-realized concrete examples (with actual review text and label)", "weight": 0.7},
            {"criterion": "Examples cover at least 2 of the 3 classes", "weight": 0.3},
        ],
        lesson_hint="WHEN writing few-shot prompts → PREFER concrete realized examples, never placeholders like [EXAMPLE]",
    ),
    Task(
        id="f3-2", family="few_shot_concrete",
        prompt="Write a few-shot prompt that teaches the model to convert natural language questions into SQL queries against a `users(id, name, signup_date, plan)` table.",
        rubric=[
            {"criterion": "Includes at least 2 fully-realized concrete examples (real question and real SQL)", "weight": 0.7},
            {"criterion": "Examples demonstrate filtering and at least one other SQL operation", "weight": 0.3},
        ],
        lesson_hint="WHEN writing few-shot prompts → PREFER concrete realized examples, never placeholders like [EXAMPLE]",
    ),
    Task(
        id="f3-3", family="few_shot_concrete",
        prompt="Write a few-shot prompt that teaches the model to detect the language of a short text and respond with the ISO 639-1 code.",
        rubric=[
            {"criterion": "Includes at least 2 fully-realized concrete examples (actual text + actual code)", "weight": 0.7},
            {"criterion": "Examples cover at least 2 different languages", "weight": 0.3},
        ],
        lesson_hint="WHEN writing few-shot prompts → PREFER concrete realized examples, never placeholders like [EXAMPLE]",
    ),
]

ALL_TASKS: list[Task] = F1 + F2 + F3
