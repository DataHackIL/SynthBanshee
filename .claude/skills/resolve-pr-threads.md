# resolve-pr-threads

Resolve all open review threads on a GitHub PR after changes have been committed.

## Usage

```
/resolve-pr-threads <PR_NUMBER>
```

## Steps

1. Fetch all open review threads and their GraphQL node IDs:

```bash
gh api graphql -f query='{ repository(owner: "OWNER", name: "REPO") { pullRequest(number: PR_NUMBER) { reviewThreads(first: 50) { nodes { id isResolved comments(first: 1) { nodes { databaseId body } } } } } } }' \
  --jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | {id, comment: .comments.nodes[0].databaseId}'
```

2. For each unresolved thread ID, resolve it:

```bash
gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "PRRT_..."}) { thread { id isResolved } } }'
```

3. Confirm all threads are now resolved by re-running step 1 and verifying the output is empty.

## Notes

- Only resolve threads whose underlying concern has been addressed (code change committed, or explicitly triaged as out-of-scope/already treated).
- Unresolved threads after a fix imply unfinished work to reviewers — always close the loop.
- The `owner` and `repo` values can be inferred from `gh repo view --json owner,name`.
