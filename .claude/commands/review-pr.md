# review-pr

Triage and resolve all unresolved review comments on a GitHub PR.

## Usage

```
/review-pr [PR number]
```

If no PR number is given, use the PR for the current branch (`gh pr view --json number`).

## Steps

1. **Fetch the PR snapshot.**
   ```
   gh pr view <number> --json number,title,baseRefName,headRefName,state
   ```
   Then fetch all unresolved review threads via GraphQL:
   ```graphql
   {
     repository(owner: "OWNER", name: "REPO") {
       pullRequest(number: N) {
         reviewThreads(first: 50) {
           nodes {
             id
             isResolved
             comments(first: 1) {
               nodes { databaseId author { login } body path line }
             }
           }
         }
       }
     }
   }
   ```
   Filter to `isResolved: false` nodes only.

2. **For each unresolved thread**, read the file at the referenced path and line,
   then recommend one of:
   - **resolve as irrelevant** — comment is factually wrong or does not apply to this codebase
   - **accept and implement the suggested solution** — suggestion is correct and fits as-is
   - **accept and implement a different solution** — issue is real but the suggestion is wrong/overcomplicated
   - **open a separate issue and resolve as out-of-scope** — valid concern but too large for this PR
   - **resolve as already treated by the code** — the code already handles the concern

   Present all recommendations together and **wait for the user to confirm** before making any changes.

3. **After user confirms**, implement all accepted changes in the working tree.
   Run `pytest` and the pre-commit hooks to verify nothing is broken.

4. **Commit** all changes in a single commit with a message that lists each COPILOT/reviewer
   item addressed (e.g. `fix: address PR review comments (COPILOT-1 taxonomy validation, ...)`).

5. **Push** the branch.

6. **Resolve threads** via GraphQL `resolveReviewThread` mutation for every accepted item:
   ```graphql
   mutation {
     resolveReviewThread(input: { threadId: "PRRT_..." }) {
       thread { isResolved }
     }
   }
   ```
   Verify each returns `isResolved: true`.

## Notes

- Always resolve threads via the GraphQL mutation — updating comment text with `PATCH /pulls/comments/:id`
  does NOT mark the thread as resolved on GitHub.
- For "resolve as irrelevant" or "already treated" items, resolve the thread without making code changes,
  and reply to the comment explaining why.
- Keep all fixes in a single commit when possible to keep the PR history clean.
