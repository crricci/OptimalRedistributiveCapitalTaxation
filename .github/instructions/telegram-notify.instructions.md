---
description: "Use in this workspace when work steps must end with Telegram notification attempts, per-command 10-minute time limits, controlled parallel execution, process cleanup, and a fixed final step report structure."
name: "Workspace Telegram Notify And Timeout Rules"
applyTo: "**"
---
# Workspace Telegram Notify And Timeout Rules

Follow these rules as default behavior in this repository unless the user explicitly revokes them.

## End Of Step Workflow

Before stopping for the user's next command at the end of any significant step:

- Prepare a short but useful summary of the step.
- The summary must include: objective, actions performed, files modified or analyzed, commands/tests/scripts run, final outcome, and suggested next steps.
- Before the final in-chat step message, run `source ~/.bashrc` in the agent terminal.
- Then run `python3 /home/cricci/vscode/notify_telegram.py --text "<summary>"`.
- Do not add a manual prefix in `--text`; the script already prepends its own prefix.
- Do not consider the step finished until the notification was attempted.

## Notification Cadence

- Send a Telegram notification at the end of every significant step, not only at the end of the full session.
- If multiple meaningful steps happen back to back, notify after each one.

## Runtime Limits

- Any single code execution, test, script, simulation, build, long command, or launched process must have a maximum duration of 10 minutes.
- Always set an explicit timeout when the tool supports it.
- If a run may exceed 10 minutes, stop and ask the user for explicit approval before starting it.
- If needed, split work into multiple runs of at most 10 minutes each.

## Parallel Execution And Cleanup

- Independent commands may run in parallel when this reduces wall time.
- Before launching multiple CPU-bound or solver-heavy processes, inspect the available logical processors with `nproc` or an equivalent command.
- Do not saturate the machine: for heavy CPU-bound work, keep at least 2 logical processors free when available, and otherwise cap concurrent heavy processes to at most `max(1, floor(nproc / 2))`.
- For memory-heavy, I/O-heavy, or solver-heavy workloads, use a more conservative limit whenever contention could materially slow or block the machine.
- Before any final in-chat step message or handoff, verify that no unintended background, orphaned, or zombie processes remain from your work.
- If you started async terminals or child processes, inspect them and terminate or reap them unless the user explicitly asked to keep them running.
- Do not return control while known zombie processes created by your commands remain unresolved.

## Timeout Or Notification Failure Handling

- If a run hits the timeout, stop it, note that in the summary, notify anyway, and propose a faster alternative or a split plan.
- If the Telegram notifier cannot be executed, clearly say so, show the command that would have been run, and provide the ready-to-send summary.

## Final Step Message Format

Every final chat message for a completed step must contain these sections in this order:

- `STATO`
- `COSA HO FATTO`
- `FILE COINVOLTI`
- `COMANDI ESEGUITI`
- `PROSSIMO PASSO`
- `NOTIFICA TELEGRAM: inviata / fallita`

When the overall task is complete, still provide the normal final answer and call `task_complete`, but only after attempting the Telegram notification for that last step.