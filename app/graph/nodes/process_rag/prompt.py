INSIGHT_PROMPT = """You are extracting ONE concrete technical insight for a short Discord reply.

Input:

original_posts: user messages (may be multiple)

query: retrieval query

rag_chunks: text returned from documentation search

Your job:

Return ONE insight sentence that directly helps the user.

Rules:

If rag_chunks contains a sentence that clearly addresses the user's issue, select it and lightly smooth grammar if needed (preserve meaning).

If rag_chunks does NOT contain relevant information for the user's problem, output a meta-insight in this format:

Lovable docs don't cover this — configuration happens directly in Supabase.

Do NOT add new technical information.

Do NOT summarize multiple ideas.

Do NOT explain or speculate.

Never output generic statements like “permissions need to be set” unless explicitly stated in docs.

Output ONLY one plain-text sentence.

Selection priority:

Directly answers the blocker or confusion.

Mentions access, permissions, roles, deployment checks, or integration limits.

Is specific and actionable.

Bad:

combining multiple chunks

adding explanations

inventing causes

Good:

lifting one sharp sentence from docs

or clearly stating that docs do not cover the issue.

Output:

<single technical sentence>
"""