# Agent Instructions

## Code Change Approval

- Before modifying any source code, tests, scripts, or other code files, first describe the intended changes and obtain the user's explicit approval.
- Do not infer approval merely from a request to investigate, diagnose, review, or propose a fix.
- Read-only inspection and analysis may proceed without approval.

## Code Change Proposal Communication

- End every code modification proposal with a concise section that states what
  problems the proposal solves and what concrete advantages the proposed
  approach provides, including material safety or maintenance benefits.
- Apply this requirement only to proposals requesting approval for future code
  changes. Do not add that section to Git commit reports, completed-task
  summaries, status updates, or other non-proposal messages.

## LW Real-Deployment Remediation

- Use `.learnings/LW_REAL_DEPLOYMENT_ISSUES.md` as the authoritative issue order and status record.
- Work on only the single `LW-XXX` issue explicitly selected by the user.
- Do not bundle later issues, opportunistic refactors, or unrelated cleanup into an approved change.
- After verification, update only the selected issue's status and resolution evidence.

## Context Compaction Handoff

- Determine the current session's compaction count by running:

  ```bash
  python3 .agents/skills/inspect-context-compactions/scripts/inspect_context_compactions.py --threshold 5
  ```

- Read the JSON `status` before using the result. Only when `status=available`
  may `compaction_count` and `threshold_reached` be treated as exact. If the
  status is `unavailable` or `inconsistent`, do not infer a count; finish the
  current atomic operation, create or refresh a precautionary handoff before
  starting more substantial work, tell the user that exact inspection failed,
  and recommend continuing in a new session.
- Treat `threshold_reached=true` at the repository threshold of five verified
  context compactions as the handoff threshold. The inspector counts verified
  compactions regardless of whether they were triggered manually or
  automatically.
- At that threshold, finish any in-progress atomic operation, then create
  `.learnings/session_handoffs/YYYYMMDD-HHMM-<topic>.md` before starting more
  substantial work.
- The handoff document must record the current objective, completed work and
  commits, user decisions and approvals, applicable constraints, verification
  results, dirty or user-owned files that must be preserved, unresolved issues,
  blockers, and the exact next step.
- Tell the user that repeated compaction may reduce continuity, recommend
  continuing in a new session, link the handoff document, and provide a
  copyable prompt that instructs the new session to read `AGENTS.md`, the
  handoff document, and any authoritative task record before continuing.
- Fill and provide this prompt template:

  ```text
  请在 /home/lfr/rl_sar 继续上一会话的任务。开始前请完整阅读
  AGENTS.md、<handoff-document> 和 <authoritative-task-record>，然后运行
  git status --short 核对工作区并保留交接文档标记的用户文件。不要重复
  已完成或已提交的工作。当前应继续：<exact-next-step>。修改代码前必须
  按 AGENTS.md 先说明方案并获得我的明确审批。
  ```

- If the user explicitly chooses to remain in the current session, continue,
  but rerun the inspector after any subsequent compaction indication and
  refresh the handoff document whenever its verified count increases.
