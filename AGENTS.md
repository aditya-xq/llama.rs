# Self-Improving Agent Instructions

## When to Learn
- You make a mistake
- User corrects you
- You discover better approaches through research

## How to Learn

### 1. Capture the Specific
When an error occurs, record:
- What failed (exact error/behavior)
- Why it failed (root cause)
- How to fix it (specific solution)

### 2. Extract General Rules
From specific cases, derive reusable patterns:
```
If [condition], then [action] instead of [previous action]
```

### 3. Update AGENTS.md
Add learnings under appropriate sections:
- **Code style**: conventions to follow
- **Patterns**: successful approaches
- **Pitfalls**: what to avoid
- **Workflow**: process improvements

## Format for Learnings

```markdown
## [Category]: [Brief Title]
- **Situation**: [When this applies]
- **Lesson**: [What to do instead]
- **Example**: [If concrete]
```

## Priority
1. Fix the immediate task
2. Document the learning in AGENTS.md
3. Apply the learning to current task if relevant
