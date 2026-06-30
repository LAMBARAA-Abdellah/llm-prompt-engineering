# Workflow Documentation

## End-to-End Workflow

### 1. Setup Phase

```
Install dependencies → Configure .env → Verify API keys → Run tests
```

### 2. Prompt Engineering Demo

```
Select technique → Build prompt → Send to LLM → Display response + metrics
```

### 3. Sentiment Analysis Workflow

```mermaid
flowchart LR
    A[Load IMDB Dataset] --> B[Clean & EDA]
    B --> C[Apply Prompt Techniques]
    C --> D[Generate Predictions]
    D --> E[Compare with Gold Labels]
    E --> F[Compute Metrics]
    F --> G[Generate Charts]
    G --> H[Save Results]
```

**Steps:**
1. Load IMDB reviews (or built-in sample)
2. Clean text data, compute statistics
3. Generate EDA visualizations
4. Run 6 prompting techniques on each sample
5. Compute Accuracy, Precision, Recall, F1
6. Generate confusion matrices per technique
7. Create comparison bar charts (accuracy, speed, cost)
8. Export results to CSV

### 4. Spam Detection Workflow

Same pipeline as sentiment analysis but with SMS Spam dataset and ham/spam labels.

### 5. Image Generation Workflow

```
Craft prompt → OpenAI DALL-E API → Download image → Save to images/generated/
```

### 6. Image Transformation Workflow

```
Source PNG → Optional mask → Edit prompt → DALL-E Edit API → Save to images/transformed/
```

## Evaluation Pipeline

| Step | Input | Output |
|------|-------|--------|
| Load Data | CSV/Archive | pandas DataFrame |
| Predict | Text + Technique | Label prediction |
| Score | Predictions + Gold labels | Accuracy, F1, etc. |
| Compare | All technique results | Comparison DataFrame |
| Visualize | Metrics | PNG charts in outputs/ |

## Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Main as main.py
    participant UC as UseCase
    participant Prompt as PromptModule
    participant LLM as LLMClient
    participant Eval as Evaluator

    User->>Main: Select demo
    Main->>UC: Initialize use case
    UC->>UC: Load & clean dataset
    loop For each technique
        UC->>Prompt: Build prompt
        Prompt->>LLM: chat(messages)
        LLM->>LLM: Retry on failure
        LLM-->>Prompt: LLMResponse
        Prompt-->>UC: Prediction
    end
    UC->>Eval: Evaluate all predictions
    Eval->>Eval: Compute metrics
    Eval-->>User: Comparison table + charts
```

## Class Diagram

```mermaid
classDiagram
    class BaseLLMClient {
        <<abstract>>
        +model: str
        +max_retries: int
        +chat(messages) LLMResponse
        +simple_prompt(text) LLMResponse
        #_chat_completion(messages) LLMResponse
    }

    class OpenAIClient {
        +generate_image(prompt) List~str~
        +edit_image(path, prompt) List~str~
    }

    class OllamaClient {
        +list_models() List~str~
        +pull_model(name) void
    }

    class GroqClient {
        +_estimate_cost(response) float
    }

    class PromptEvaluator {
        +evaluate_technique(name, df, fn) TechniqueResult
        +compare_all() DataFrame
        +plot_comparison() void
    }

    class LLMResponse {
        +content: str
        +execution_time: float
        +total_tokens: int
        +metadata: dict
    }

    BaseLLMClient <|-- OpenAIClient
    BaseLLMClient <|-- OllamaClient
    BaseLLMClient <|-- GroqClient
    BaseLLMClient --> LLMResponse
    PromptEvaluator --> BaseLLMClient
```
