# Agent Instructions

## Scope
These instructions apply to all work in this repository.

## Priority
1. Fix the immediate task.
2. Record reusable learnings in this file when warranted.
3. Apply the learning to the current task if it helps.

## When to Learn
Update this file when:
- You make a mistake.
- The user corrects you.
- You discover a clearly better approach through testing or research.

## How to Learn
For each learning, capture:
- What failed.
- Why it failed.
- What to do instead.

Convert the specific case into a reusable rule:

```text
If [condition], then [action] instead of [previous action].
```

Use this format:

```markdown
## [Category]: [Title]
- **Situation**: When this applies
- **Lesson**: What to do
- **Example**: Concrete example, if useful
```

Recommended categories:
- `Code Style`
- `Patterns`
- `Pitfalls`
- `Workflow`

## Engineering Standards

### Design
- Keep types and functions focused on one responsibility.
- Prefer small modules by domain when adding a new feature area.
- Depend on typed results and traits instead of direct printing or tightly coupled logic.
- Prefer composition of small checks/workflows over large monolithic commands.

### Async and I/O
- Add timeouts to external I/O and long-running async operations.
- Run independent async work concurrently when it improves latency.
- Keep side effects at the edges of the system.

### Errors and State
- Return `Result` with useful context.
- Model success, partial success, timeout, and failure states explicitly.
- Do not conflate different failure modes into one generic path.

### DRY and CLI UX
- Reuse shared output/styling helpers instead of duplicating formatting.
- Reuse shared state/result types where possible.
- Keep command output consistent and predictable.

## Rust Workflow

### Before Editing
- Inspect the existing module structure.
- Find nearby patterns and follow them unless there is a strong reason not to.
- Identify related modules, tests, and user-facing behavior that may be affected.

### During Refactoring
- Keep functions small and focused.
- Prefer typed diagnostic/check results over inline printing.
- Fix closely related bugs you encounter when they are in scope and low-risk.

### After Changes
- Run `cargo check`.
- Run `cargo test`.
- Add or update unit, integration, or E2E tests when behavior changes.
- Verify edge cases relevant to the change.

Useful test commands:

```bash
cargo check
cargo test
cargo test --lib
cargo test --test unit
cargo test --test integration
cargo test --test e2e
```

## Recorded Learnings

## Patterns: Domain-Driven Modules
- **Situation**: Adding a new feature area such as diagnostics, monitoring, or health checks
- **Lesson**: Create a dedicated module tree instead of growing unrelated files
- **Example**: Add `src/<feature>/mod.rs` with focused submodules

## Patterns: Typed Diagnostic Results
- **Situation**: Implementing checks, probes, or detection logic
- **Lesson**: Return typed state structs instead of printing directly inside the check
- **Example**:
```rust
#[derive(Debug, Clone)]
pub struct DiagnosticResult {
    pub success: bool,
    pub data: Option<Info>,
    pub error: Option<String>,
}
```

## Patterns: Parallel Async Operations
- **Situation**: Running multiple independent I/O-bound checks
- **Lesson**: Use `tokio::join!` to execute them concurrently
- **Example**:
```rust
let (a, b) = tokio::join!(check_a(), check_b());
```

## Patterns: Timeout External Operations
- **Situation**: Network calls, subprocesses, or any async work that may hang
- **Lesson**: Wrap the operation in a timeout with a reasonable bound
- **Example**:
```rust
timeout(Duration::from_secs(5), async_operation).await
```

## Patterns: Explicit State Handling
- **Situation**: Operations with more than one meaningful outcome
- **Lesson**: Represent states explicitly with enums or clear structs
- **Example**: Distinguish success, partial success, timeout, and failure

## Workflow: Fix Related Nearby Bugs
- **Situation**: You find a clearly related bug while working in the same area
- **Lesson**: Fix it when the scope is small and the behavior is well understood
- **Example**: Correct a broken profile lookup while already refactoring profile management

## Pitfalls: Dry-Run Must Stay Offline
- **Situation**: Implementing CLI dry-run flows for Docker-backed commands
- **Lesson**: Do not require Docker CLI or daemon availability before rendering a dry-run command; only real execution paths should validate runtime dependencies
- **Example**: `llmr serve --dry-run` should still print the final `docker run ...` command even when Docker is stopped

## Pitfalls: Unit Conversion In Hardware Detection
- **Situation**: Translating OS-reported memory values into tuning heuristics
- **Lesson**: Normalize units before storing or comparing them
- **Example**: Convert Linux `/proc/meminfo` KiB to GiB and Windows adapter RAM bytes to MiB
