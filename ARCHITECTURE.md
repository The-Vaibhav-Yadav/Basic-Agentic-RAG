# Agentic RAG Architecture

## 🔄 Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    User asks a question                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  ATTEMPT 1 (or retry with context if previous failed)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  1. RESEARCHER AGENT    │
         │  • Analyzes question    │
         │  • Chooses tool:        │
         │    - PDF Search Tool    │
         │    - Web Search Tool    │
         │  • Gathers research     │
         └───────────┬─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │  2. FACT CHECKER AGENT  │
         │  • Reviews research     │
         │  • Checks accuracy      │
         │  • Returns:             │
         │    VERDICT: TRUE/FALSE  │
         │    REASON: ...          │
         └───────────┬─────────────┘
                     │
            ┌────────┴────────┐
            │                 │
      VERDICT: TRUE     VERDICT: FALSE
            │                 │
            ▼                 ▼
   ┌────────────────┐  ┌──────────────────┐
   │  3. WRITER     │  │  RETRY LOGIC     │
   │  • Writes      │  │  • retries < 2?  │
   │    answer      │  │  • Yes: RETRY    │
   │  • User sees   │  │  • No: Return    │
   │    this!       │  │    best answer   │
   └────────┬───────┘  └────────┬─────────┘
            │                   │
            ▼                   │
   ┌────────────────┐           │
   │  FINAL ANSWER  │◄──────────┘
   │  (Clean, no    │
   │   verdict)     │
   └────────────────┘
```

## 📊 Task Execution Order

1. **Research Task** → Gathers information from PDF or web
2. **Fact Checking Task** → Validates research accuracy (internal only)
3. **Writing Task** → Creates final answer for user

## 🔁 Retry Mechanism

- **Max Retries**: 2 (3 total attempts)
- **Trigger**: VERDICT: FALSE from fact checker
- **Context Passing**: Previous incorrect research + fact checker feedback
- **Final Fallback**: Returns best available answer with disclaimer

## 👥 Agent Roles

### 1. Researcher Agent
- **Goal**: Find accurate information
- **Tools**: PDF Search Tool, Web Search Tool
- **Aware**: Can see previous failed attempts and adjusts strategy

### 2. Fact Checker Agent
- **Goal**: Verify research accuracy
- **Input**: Research findings
- **Output**: VERDICT + REASON (not shown to user)

### 3. Writer Agent
- **Goal**: Create clear answers
- **Input**: Verified research
- **Output**: Final answer (shown to user)

## 🎯 Key Features

- ✅ **Research validation** before writing
- ✅ **Automatic retries** with improved context
- ✅ **Clean output** (no internal verdicts shown)
- ✅ **Progress tracking** in console
- ✅ **Smart tool selection** (PDF vs web)

## 🛡️ Quality Assurance

The fact checker ensures:
- Research is factually accurate
- Sources are appropriate
- Information addresses the question
- No contradictions exist

## 💡 Example Flow

**Question**: "What is self-attention?"

1. **Researcher**: Searches PDF → finds 3 relevant chunks
2. **Fact Checker**: ✅ VERDICT: TRUE (research is accurate)
3. **Writer**: Creates answer → "Self-attention is a mechanism..."
4. **User sees**: Only the writer's clean answer

**If Fact Check Fails**:
1. **Fact Checker**: ❌ VERDICT: FALSE (incomplete research)
2. **System**: Retries with context about what was wrong
3. **Researcher**: Tries different keywords/approach
4. **Process repeats** until success or max retries

