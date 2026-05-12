---
description: "Activate Telegram notification workflow, 10-minute timeout discipline, controlled parallel execution, process cleanup, and fixed final step reporting for the current work when invoked in this workspace."
name: "Telegram Step Mode"
argument-hint: "Describe the task to perform under Telegram step mode"
agent: "agent"
model: "GPT-5 (copilot)"
---
Work on the following task in this repository:

{{input}}

For this task, apply these operating rules throughout the work:

- At the end of every significant step, before waiting for the next user command, prepare a short but useful summary.
- Each summary must include: objective, actions performed, files modified or analyzed, commands/tests/scripts run, final outcome, and suggested next steps.
- Before the final in-chat message for each completed step, run `source ~/.bashrc` and then `python3 /home/cricci/vscode/notify_telegram.py --text "<summary>"`.
- Do not add a manual prefix in `--text`; the script already prepends its own prefix.
- Do not consider a step complete until the notification was attempted.
- Send a Telegram notification at the end of every significant step, not only at the end of the full session.
- Any single code execution, test, script, simulation, build, long command, or launched process must have a maximum duration of 10 minutes.
- Always set an explicit timeout when possible.
- If a run may exceed 10 minutes, stop and ask the user for explicit approval before starting it.
- Independent commands may run in parallel when that materially reduces wall time.
- Before launching multiple CPU-bound or solver-heavy processes, inspect the available logical processors with `nproc` or an equivalent command.
- Do not saturate the machine: for heavy CPU-bound work, keep at least 2 logical processors free when available, and otherwise cap concurrent heavy processes to at most `max(1, floor(nproc / 2))`.
- For memory-heavy, I/O-heavy, or solver-heavy workloads, use a more conservative limit whenever contention could materially slow or block the machine.
- If a run hits the timeout, stop it, record that in the summary, notify anyway, and propose a faster alternative or a split plan.
- If the Telegram notifier cannot be executed, clearly say so, show the command that would have been run, and provide the ready-to-send summary.
- Before any final in-chat step message or handoff, verify that no unintended background, orphaned, or zombie processes remain from your work.
- If you started async terminals or child processes, inspect them and terminate or reap them unless the user explicitly asked to keep them running.
- Do not return control while known zombie processes created by your commands remain unresolved.

Every final chat message for a completed step must contain these sections in this order:

- `STATO`
- `COSA HO FATTO`
- `FILE COINVOLTI`
- `COMANDI ESEGUITI`
- `PROSSIMO PASSO`
- `NOTIFICA TELEGRAM: inviata / fallita`