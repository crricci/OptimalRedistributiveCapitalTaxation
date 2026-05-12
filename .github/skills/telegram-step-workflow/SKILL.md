---
name: telegram-step-workflow
description: "Telegram workflow for this workspace. Use when work must end with Telegram notification attempts, 10-minute command limits, controlled parallel execution, and process cleanup before handoff."
argument-hint: "Describe the task to perform under Telegram workflow"
user-invocable: true
---

# Telegram Step Workflow

## When To Use

Use this skill in this repository when the work must follow all of these operating rules:

- send a Telegram notification at the end of every significant step
- keep every single command, script, test, build, or launched process within 10 minutes unless the user explicitly approves more
- allow parallel execution only when it is safe for the machine
- avoid leaving unintended background, orphaned, or zombie processes behind
- end each completed step with the fixed step report structure

## Procedure

1. Work on the requested task normally, but treat every meaningful milestone as a step that must be closed explicitly.
2. Before launching multiple heavy commands in parallel, inspect the available logical processors with `nproc` or an equivalent command.
3. For CPU-bound parallel work, keep at least 2 logical processors free when available, and otherwise cap concurrent heavy processes to at most `max(1, floor(nproc / 2))`.
4. For memory-heavy, I/O-heavy, or solver-heavy workloads, use a more conservative limit whenever contention could materially slow or block the machine.
5. Set an explicit timeout whenever the tool supports it, and keep each single run within 10 minutes unless the user has approved more.
6. If a run times out, stop it, record that in the step summary, notify anyway, and propose a faster alternative or split plan.
7. Before the final in-chat message for each completed step, run `source ~/.bashrc` and then `python3 /home/cricci/vscode/notify_telegram.py --text "<summary>"`.
8. Do not add a manual prefix in `--text`; the notifier already prepends its own prefix.
9. Before handing control back, verify that no unintended background, orphaned, or zombie processes remain from your work.
10. If you started async terminals or child processes, inspect them and terminate or reap them unless the user explicitly asked to keep them running.
11. Do not consider the step complete until the notification was attempted and process cleanup has been checked.

## Step Summary Contents

Every step summary must include:

- objective
- actions performed
- files modified or analyzed
- commands, tests, or scripts run
- final outcome
- suggested next steps

## Final Step Message Format

Every final chat message for a completed step must contain these sections in this order:

- `STATO`
- `COSA HO FATTO`
- `FILE COINVOLTI`
- `COMANDI ESEGUITI`
- `PROSSIMO PASSO`
- `NOTIFICA TELEGRAM: inviata / fallita`
