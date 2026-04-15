# Effective Todo Scoping Guide

This guide explains what makes a good todo item and how to scope tasks so Claude Code can work through them reliably.

## What is a "small" todo?

A small todo is a **single, atomic unit of work** — one change, one file, one outcome. You know it's done when you can tick a clear, objective box.

### Good examples (small and specific)

- Add logout button to `src/components/Header.tsx`
- Fix null check in `getUserById` at `db/users.ts:42`
- Write test for `/api/search` 400 response
- Update `fetchDestinations` to handle empty results

### Too big (needs splitting)

- Fix the auth flow
- Refactor the UI
- Set up the database
- Clean up the backend

Each of those should become 3–5 smaller, specific todos.

---

## Rules for scoping effectively

### 1. One file or one concept per todo

If a todo touches three files for three different reasons, split it. If it touches three files for *one* reason (e.g., renaming a shared type), it can stay together.

### 2. Order by dependency

Write todos in the order they must be completed. A todo that depends on another goes after it in the list.

### 3. Name it so success is obvious

Bad: `Fix fetching`
Good: `Update fetchDestinations in api/destinations.ts to return [] on 404 instead of throwing`

The good version has a clear before and after — you can verify it's done.

### 4. Mark complete immediately — never batch

The moment a task is done, mark it complete. Do not finish three todos and mark them all at the end. Immediate marking keeps the list accurate and prevents lost context.

### 5. Don't over-plan upfront

Create todos for what you know now. Add more as you discover them during work. A 15-item list created before reading any code is usually wrong by item 4.

### 6. One in-progress task at a time

Only one todo should be `in_progress` at any moment. Multiple in-progress todos are a sign that the scope needs to be broken down further.

---

## Quick reference

| Signal | What to do |
|---|---|
| Todo description is vague | Rewrite it with a specific file and outcome |
| Todo touches many unrelated files | Split into one todo per concern |
| You're unsure when the todo is "done" | Rewrite it so the done state is concrete |
| Multiple todos are in-progress | Finish the current one before starting the next |
| The list is more than ~8 items | Work through the first few before adding more |
