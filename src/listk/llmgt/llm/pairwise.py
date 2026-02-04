"""Pairwise comparison prompt construction and response parsing.

This module provides utilities for constructing prompts for pairwise
document comparison and parsing LLM responses to extract winners.

Optimized for efficiency:
- Minimal prompts (no document snippets, just titles)
- Single-step generation with succinct reasoning
- Async batched comparisons for parallelism
- Regex-based answer extraction
"""

import asyncio
import random
import re
import uuid
from datetime import datetime, UTC
from typing import Callable, Optional

from ..data.models import Document, PairwiseComparison, Query
from .base import ComparisonResult, LLMBackend


# Dataset descriptions for context
DATASET_DESCRIPTIONS: dict[str, str] = {
    "scifact": (
        "SciFact: Scientific claim verification. Queries are scientific claims. "
        "Documents are paper abstracts. Relevant = provides evidence supporting or refuting the claim."
    ),
    "scidocs": (
        "SciDocs: Scientific document similarity. Queries are paper titles/abstracts. "
        "Documents are other paper abstracts. Relevant = topically similar or cite-worthy."
    ),
}


def get_dataset_description(dataset_name: str) -> str:
    """Get the description for a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    str
        Description of the dataset and relevance criteria.
    """
    return DATASET_DESCRIPTIONS.get(
        dataset_name,
        f"Document retrieval task for {dataset_name}. Relevant = helps answer the query."
    )


def format_document_snippet(doc: Document, max_chars: int = 1500) -> str:
    """Format a document for inclusion in a prompt.

    Extracts title and a snippet of the text content.

    Parameters
    ----------
    doc : Document
        The document to format.
    max_chars : int, optional
        Maximum characters for the text snippet. Default 1500.

    Returns
    -------
    str
        Formatted document string.
    """
    title_part: str = f"Title: {doc.title}\n" if doc.title else ""

    # Truncate text if necessary
    text: str = doc.text
    if len(text) > max_chars:
        # Try to truncate at a sentence boundary
        truncated: str = text[:max_chars]
        last_period: int = truncated.rfind(". ")
        if last_period > max_chars * 0.5:
            truncated = truncated[: last_period + 1]
        text = truncated + " [...]"

    return f"{title_part}Text: {text}"


def format_document_minimal(doc: Document) -> str:
    """Format a document minimally - just title or first sentence.

    Parameters
    ----------
    doc : Document
        The document to format.

    Returns
    -------
    str
        Minimal document representation.
    """
    if doc.title:
        return doc.title
    # Fall back to first sentence if no title
    first_sentence: str = doc.text.split(". ")[0]
    if len(first_sentence) > 200:
        first_sentence = first_sentence[:200] + "..."
    return first_sentence


# Document representation modes
DOC_MODE_TITLE = "title"
DOC_MODE_SUMMARY = "summary"
DOC_MODE_SNIPPET = "snippet"
DOC_MODES = [DOC_MODE_TITLE, DOC_MODE_SUMMARY, DOC_MODE_SNIPPET]


class DocumentSummarizer:
    """Async batch summarizer for documents.

    Generates succinct summaries of documents using an LLM.
    Caches summaries to avoid re-summarizing the same document.
    """

    def __init__(
        self,
        llm: LLMBackend,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize the summarizer.

        Parameters
        ----------
        llm : LLMBackend
            The LLM backend to use for summarization.
        max_concurrent : int, optional
            Maximum concurrent requests. Default 10.
        """
        self.llm: LLMBackend = llm
        self.max_concurrent: int = max_concurrent
        self._cache: dict[str, str] = {}
        self._loop = None

    async def summarize_async(self, doc: Document) -> str:
        """Summarize a single document asynchronously.

        Parameters
        ----------
        doc : Document
            The document to summarize.

        Returns
        -------
        str
            A succinct summary (1-2 sentences).
        """
        if doc.doc_id in self._cache:
            return self._cache[doc.doc_id]

        prompt = f"""Summarize this document in several sentences:

Title: {doc.title or 'N/A'}
Text: {doc.text[:2000]}

Summary:"""

        if hasattr(self.llm, 'generate_async'):
            summary = await self.llm.generate_async(
                prompt=prompt,
                system_prompt="You are a document summarizer.",
                max_tokens=500,
                temperature=0.0,
            )
        else:
            summary = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a document summarizer.",
                max_tokens=500,
                temperature=0.0,
            )

        self._cache[doc.doc_id] = summary.strip()
        return self._cache[doc.doc_id]

    async def summarize_batch_async(
        self,
        docs: list[Document],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> dict[str, str]:
        """Summarize multiple documents in parallel.

        Parameters
        ----------
        docs : list[Document]
            Documents to summarize.
        progress_callback : callable, optional
            Callback with count of completed summaries.

        Returns
        -------
        dict[str, str]
            Mapping from doc_id to summary.
        """
        # Filter out already cached docs
        uncached_docs = [d for d in docs if d.doc_id not in self._cache]

        if not uncached_docs:
            return {d.doc_id: self._cache[d.doc_id] for d in docs}

        # Build prompts
        system_prompt = "You are a document summarizer. Be concise."
        prompts: list[tuple[str, Optional[str]]] = []

        for doc in uncached_docs:
            prompt = f"""Summarize this document in 1-2 sentences (max 100 words):

Title: {doc.title or 'N/A'}
Text: {doc.text[:2000]}

Summary:"""
            prompts.append((prompt, system_prompt))

        # Batch generate
        if hasattr(self.llm, 'generate_batch_async'):
            summaries = await self.llm.generate_batch_async(
                prompts, max_tokens=100, temperature=0.0
            )
        else:
            # Fallback to sequential
            summaries = []
            for prompt, sys in prompts:
                if hasattr(self.llm, 'generate_async'):
                    s = await self.llm.generate_async(prompt, sys, max_tokens=100)
                else:
                    s = self.llm.generate(prompt, sys, max_tokens=100)
                summaries.append(s)
                if progress_callback:
                    progress_callback(len(summaries))

        # Cache results
        for doc, summary in zip(uncached_docs, summaries):
            self._cache[doc.doc_id] = summary.strip()

        if progress_callback:
            progress_callback(len(docs))

        return {d.doc_id: self._cache[d.doc_id] for d in docs}

    def summarize_batch(
        self,
        docs: list[Document],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> dict[str, str]:
        """Summarize multiple documents (sync wrapper).

        Parameters
        ----------
        docs : list[Document]
            Documents to summarize.
        progress_callback : callable, optional
            Callback with count of completed summaries.

        Returns
        -------
        dict[str, str]
            Mapping from doc_id to summary.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.summarize_batch_async(docs, progress_callback)
                )
                return future.result()
        else:
            if not hasattr(self, '_loop') or self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop.run_until_complete(
                self.summarize_batch_async(docs, progress_callback)
            )

    def get_summary(self, doc_id: str) -> Optional[str]:
        """Get cached summary for a document.

        Parameters
        ----------
        doc_id : str
            Document ID.

        Returns
        -------
        str or None
            Cached summary, or None if not cached.
        """
        return self._cache.get(doc_id)


def build_system_prompt(dataset_name: str) -> str:
    """Build the system prompt for pairwise comparison.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    str
        System prompt string.
    """
    description: str = get_dataset_description(dataset_name)

    return f"""You are an expert relevance judge for information retrieval evaluation.

{description}

Your task is to compare two documents and determine which one is MORE RELEVANT to a given query. You must:
1. Carefully read the query and both documents
2. Consider how directly each document addresses the query
3. Provide clear reasoning for your decision
4. Make a definitive choice - you MUST pick either A or B

Even if both documents seem equally relevant or irrelevant, you must choose the one that is slightly better. There are no ties."""


def build_system_prompt_minimal(dataset_name: str) -> str:
    """Build a minimal system prompt.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    str
        Minimal system prompt.
    """
    description: str = get_dataset_description(dataset_name)
    return f"Relevance judge. {description} Compare docs A vs B. Pick the more relevant one. Be brief."


def build_user_prompt(
    query: Query,
    doc_a: Document,
    doc_b: Document,
    doc_a_snippet: str,
    doc_b_snippet: str,
) -> str:
    """Build the user prompt for pairwise comparison (full version).

    Parameters
    ----------
    query : Query
        The query to judge relevance against.
    doc_a : Document
        First document.
    doc_b : Document
        Second document.
    doc_a_snippet : str
        Formatted snippet for document A.
    doc_b_snippet : str
        Formatted snippet for document B.

    Returns
    -------
    str
        Formatted user prompt.
    """
    return f"""Compare the following two documents for relevance to the query.

QUERY: {query.text}

DOCUMENT A:
{doc_a_snippet}

DOCUMENT B:
{doc_b_snippet}

Instructions:
1. Analyze how each document relates to the query
2. Explain your reasoning step by step
3. End your response with your final answer on a new line in exactly this format:
   ANSWER: A
   or
   ANSWER: B

Which document is more relevant to the query?"""


def build_user_prompt_minimal(
    query: Query,
    doc_a: Document,
    doc_b: Document,
) -> str:
    """Build a minimal user prompt - just query and doc titles.

    Parameters
    ----------
    query : Query
        The query.
    doc_a : Document
        First document.
    doc_b : Document
        Second document.

    Returns
    -------
    str
        Minimal user prompt.
    """
    title_a: str = format_document_minimal(doc_a)
    title_b: str = format_document_minimal(doc_b)

    return f"""Query: {query.text}

A: {title_a}
B: {title_b}

Which is more relevant? Brief reason, then answer with ANSWER: A or ANSWER: B"""


def format_document_for_prompt(
    doc: Document,
    mode: str,
    summaries: Optional[dict[str, str]] = None,
    max_snippet_chars: int = 1500,
) -> str:
    """Format a document based on the specified mode.

    Parameters
    ----------
    doc : Document
        The document to format.
    mode : str
        One of 'title', 'summary', or 'snippet'.
    summaries : dict[str, str], optional
        Pre-computed summaries (required for 'summary' mode).
    max_snippet_chars : int, optional
        Max chars for snippet mode. Default 1500.

    Returns
    -------
    str
        Formatted document representation.
    """
    if mode == DOC_MODE_TITLE:
        return format_document_minimal(doc)
    elif mode == DOC_MODE_SUMMARY:
        if summaries and doc.doc_id in summaries:
            title = doc.title or "Untitled"
            return f"{title}\n{summaries[doc.doc_id]}"
        else:
            # Fallback to title if no summary
            return format_document_minimal(doc)
    elif mode == DOC_MODE_SNIPPET:
        return format_document_snippet(doc, max_snippet_chars)
    else:
        raise ValueError(f"Unknown doc mode: {mode}. Use one of {DOC_MODES}")


def build_user_prompt_with_mode(
    query: Query,
    doc_a: Document,
    doc_b: Document,
    mode: str,
    summaries: Optional[dict[str, str]] = None,
    max_snippet_chars: int = 1500,
) -> str:
    """Build user prompt with specified document representation mode.

    Parameters
    ----------
    query : Query
        The query.
    doc_a : Document
        First document.
    doc_b : Document
        Second document.
    mode : str
        One of 'title', 'summary', or 'snippet'.
    summaries : dict[str, str], optional
        Pre-computed summaries for 'summary' mode.
    max_snippet_chars : int, optional
        Max chars for 'snippet' mode. Default 1500.

    Returns
    -------
    str
        User prompt.
    """
    repr_a = format_document_for_prompt(doc_a, mode, summaries, max_snippet_chars)
    repr_b = format_document_for_prompt(doc_b, mode, summaries, max_snippet_chars)

    if mode == DOC_MODE_TITLE:
        return f"""Query: {query.text}

A: {repr_a}
B: {repr_b}

Which is more relevant? Brief reason, then answer with ANSWER: A or ANSWER: B"""
    else:
        return f"""Query: {query.text}

DOCUMENT A:
{repr_a}

DOCUMENT B:
{repr_b}

Which document is more relevant? Brief reason, then answer with ANSWER: A or ANSWER: B"""


def parse_winner(response: str) -> Optional[str]:
    """Parse the winner from an LLM response.

    Looks for 'ANSWER: A' or 'ANSWER: B' pattern.

    Parameters
    ----------
    response : str
        The LLM's response.

    Returns
    -------
    str or None
        'A' or 'B' if found, None otherwise.
    """
    # Look for explicit ANSWER: pattern (case insensitive)
    pattern: str = r"ANSWER:\s*([AB])\b"
    matches: list[str] = re.findall(pattern, response, re.IGNORECASE)

    if matches:
        # Return the last match (in case LLM gives multiple)
        return matches[-1].upper()

    # Fallback: look for standalone A or B at end of response
    lines: list[str] = response.strip().split("\n")
    last_line: str = lines[-1].strip().upper()

    if last_line in ("A", "B"):
        return last_line
    if last_line.endswith(" A") or last_line.endswith(" A."):
        return "A"
    if last_line.endswith(" B") or last_line.endswith(" B."):
        return "B"

    # Last resort: look for "Document A" or "Document B" anywhere
    if "document a" in response.lower() and "document b" not in response.lower():
        return "A"
    if "document b" in response.lower() and "document a" not in response.lower():
        return "B"

    return None


def run_pairwise_comparison(
    llm: LLMBackend,
    query: Query,
    doc_a: Document,
    doc_b: Document,
    dataset_name: str,
    randomize_order: bool = True,
    max_snippet_chars: int = 1500,
    minimal: bool = True,
) -> PairwiseComparison:
    """Run a single pairwise comparison.

    Parameters
    ----------
    llm : LLMBackend
        The LLM backend to use.
    query : Query
        The query to judge relevance against.
    doc_a : Document
        First document (canonical order).
    doc_b : Document
        Second document (canonical order).
    dataset_name : str
        Name of the dataset for context.
    randomize_order : bool, optional
        Whether to randomize presentation order. Default True.
    max_snippet_chars : int, optional
        Maximum characters for document snippets. Default 1500.
    minimal : bool, optional
        Use minimal prompts (titles only). Default True.

    Returns
    -------
    PairwiseComparison
        The comparison result with all metadata.

    Raises
    ------
    ValueError
        If the LLM response cannot be parsed.
    """
    # Optionally randomize presentation order
    if randomize_order and random.random() < 0.5:
        presented_a, presented_b = doc_b, doc_a
        presented_order: list[str] = [doc_b.doc_id, doc_a.doc_id]
        order_swapped: bool = True
    else:
        presented_a, presented_b = doc_a, doc_b
        presented_order = [doc_a.doc_id, doc_b.doc_id]
        order_swapped = False

    # Build prompts based on mode
    if minimal:
        system_prompt: str = build_system_prompt_minimal(dataset_name)
        user_prompt: str = build_user_prompt_minimal(query, presented_a, presented_b)
    else:
        system_prompt = build_system_prompt(dataset_name)
        snippet_a: str = format_document_snippet(presented_a, max_snippet_chars)
        snippet_b: str = format_document_snippet(presented_b, max_snippet_chars)
        user_prompt = build_user_prompt(
            query, presented_a, presented_b, snippet_a, snippet_b
        )

    # Get LLM response
    result: ComparisonResult = llm.compare_documents(
        query=query,
        doc_a=presented_a,
        doc_b=presented_b,
        dataset_description=get_dataset_description(dataset_name),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Parse winner
    winner_label: Optional[str] = parse_winner(result.raw_response)
    if winner_label is None:
        # Try to extract from reasoning
        winner_label = parse_winner(result.reasoning)

    if winner_label is None:
        raise ValueError(
            f"Could not parse winner from LLM response: {result.raw_response[:500]}"
        )

    # Map winner label back to actual document ID
    if winner_label == "A":
        winner_doc: Document = presented_a
    else:
        winner_doc = presented_b

    return PairwiseComparison(
        comparison_id=str(uuid.uuid4()),
        query_id=query.query_id,
        doc_a_id=doc_a.doc_id,
        doc_b_id=doc_b.doc_id,
        presented_order=presented_order,
        winner_id=winner_doc.doc_id,
        reasoning=result.reasoning,
        model=llm.name,
        timestamp=datetime.now(UTC),
    )


def generate_all_pairs(doc_ids: list[str]) -> list[tuple[str, str]]:
    """Generate all unique pairs from a list of document IDs.

    Parameters
    ----------
    doc_ids : list[str]
        List of document IDs.

    Returns
    -------
    list[tuple[str, str]]
        List of (doc_a_id, doc_b_id) pairs, where doc_a_id < doc_b_id
        lexicographically to ensure consistent ordering.
    """
    pairs: list[tuple[str, str]] = []
    n: int = len(doc_ids)

    for i in range(n):
        for j in range(i + 1, n):
            # Maintain consistent ordering for deduplication
            a, b = doc_ids[i], doc_ids[j]
            if a > b:
                a, b = b, a
            pairs.append((a, b))

    return pairs


def get_completed_pairs(
    comparisons: list[PairwiseComparison],
) -> set[tuple[str, str]]:
    """Extract the set of completed pairs from existing comparisons.

    Parameters
    ----------
    comparisons : list[PairwiseComparison]
        List of completed comparisons.

    Returns
    -------
    set[tuple[str, str]]
        Set of (doc_a_id, doc_b_id) tuples that have been compared.
    """
    completed: set[tuple[str, str]] = set()

    for comp in comparisons:
        # Normalize ordering
        a, b = comp.doc_a_id, comp.doc_b_id
        if a > b:
            a, b = b, a
        completed.add((a, b))

    return completed


# Models that require higher max_tokens (reasoning models)
REASONING_MODELS = {"o1", "o1-mini", "o3-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.2"}


class LLMComparator:
    """Comparator class for sorting documents using LLM pairwise comparisons.

    This class wraps an LLM backend and provides a comparison function
    that can be used with sorting algorithms.

    Attributes
    ----------
    llm : LLMBackend
        The LLM backend to use for comparisons.
    query : Query
        The query to judge relevance against.
    corpus : dict[str, Document]
        Mapping from doc_id to Document objects.
    dataset_name : str
        Name of the dataset for context.
    doc_mode : str
        Document representation mode: 'title', 'summary', or 'snippet'.
    summaries : dict[str, str]
        Pre-computed document summaries (for 'summary' mode).
    comparisons : list[PairwiseComparison]
        List of all comparisons made (for logging/resumability).
    comparison_count : int
        Number of comparisons performed.
    max_tokens : int
        Maximum tokens for LLM response.
    """

    def __init__(
        self,
        llm: LLMBackend,
        query: Query,
        corpus: dict[str, Document],
        dataset_name: str,
        max_snippet_chars: int = 1500,
        randomize_order: bool = True,
        minimal: bool = True,
        doc_mode: str = DOC_MODE_TITLE,
        summaries: Optional[dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize the comparator.

        Parameters
        ----------
        llm : LLMBackend
            The LLM backend to use.
        query : Query
            The query to judge relevance against.
        corpus : dict[str, Document]
            Document corpus mapping.
        dataset_name : str
            Name of the dataset.
        max_snippet_chars : int, optional
            Maximum snippet length. Default 1500.
        randomize_order : bool, optional
            Whether to randomize presentation order. Default True.
        minimal : bool, optional
            Use minimal prompts (deprecated, use doc_mode). Default True.
        doc_mode : str, optional
            Document representation: 'title', 'summary', or 'snippet'. Default 'title'.
        summaries : dict[str, str], optional
            Pre-computed summaries for 'summary' mode.
        max_tokens : int, optional
            Max tokens for LLM response. Auto-detected based on model if not specified.
        """
        self.llm: LLMBackend = llm
        self.query: Query = query
        self.corpus: dict[str, Document] = corpus
        self.dataset_name: str = dataset_name
        self.max_snippet_chars: int = max_snippet_chars
        self.randomize_order: bool = randomize_order
        self.minimal: bool = minimal
        self.doc_mode: str = doc_mode
        self.summaries: dict[str, str] = summaries or {}
        self.comparisons: list[PairwiseComparison] = []
        self.comparison_count: int = 0
        # Cache for comparison results: (doc_a_id, doc_b_id) -> winner_id
        self._cache: dict[tuple[str, str], str] = {}

        # Auto-detect max_tokens based on model type
        if max_tokens is not None:
            self.max_tokens: int = max_tokens
        elif llm.model_id in REASONING_MODELS or llm.name in REASONING_MODELS:
            # Reasoning models need more tokens for their extended thinking
            self.max_tokens = 1000
        else:
            self.max_tokens = 150

    def compare(self, doc_a_id: str, doc_b_id: str) -> int:
        """Compare two documents by relevance.

        Parameters
        ----------
        doc_a_id : str
            First document ID.
        doc_b_id : str
            Second document ID.

        Returns
        -------
        int
            -1 if doc_a is more relevant, 1 if doc_b is more relevant.
        """
        # Check cache first (normalize key ordering)
        cache_key: tuple[str, str] = (min(doc_a_id, doc_b_id), max(doc_a_id, doc_b_id))
        if cache_key in self._cache:
            winner_id: str = self._cache[cache_key]
            return -1 if winner_id == doc_a_id else 1

        doc_a: Document = self.corpus[doc_a_id]
        doc_b: Document = self.corpus[doc_b_id]

        comparison: PairwiseComparison = run_pairwise_comparison(
            llm=self.llm,
            query=self.query,
            doc_a=doc_a,
            doc_b=doc_b,
            dataset_name=self.dataset_name,
            randomize_order=self.randomize_order,
            max_snippet_chars=self.max_snippet_chars,
            minimal=self.minimal,
        )

        self.comparisons.append(comparison)
        self.comparison_count += 1
        self._cache[cache_key] = comparison.winner_id

        # Return -1 if doc_a wins (should come first), 1 if doc_b wins
        return -1 if comparison.winner_id == doc_a_id else 1

    async def compare_batch_async(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[int]:
        """Compare multiple document pairs in parallel.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            List of (doc_a_id, doc_b_id) pairs to compare.

        Returns
        -------
        list[int]
            List of comparison results (-1 if doc_a wins, 1 if doc_b wins).
        """
        # Filter out cached pairs and track their indices
        uncached_pairs: list[tuple[int, str, str]] = []
        results: list[Optional[int]] = [None] * len(pairs)

        for i, (doc_a_id, doc_b_id) in enumerate(pairs):
            cache_key: tuple[str, str] = (min(doc_a_id, doc_b_id), max(doc_a_id, doc_b_id))
            if cache_key in self._cache:
                winner_id: str = self._cache[cache_key]
                results[i] = -1 if winner_id == doc_a_id else 1
            else:
                uncached_pairs.append((i, doc_a_id, doc_b_id))

        if not uncached_pairs:
            return results  # All cached

        # Build prompts for uncached pairs
        system_prompt: str = build_system_prompt_minimal(self.dataset_name)
        prompts_data: list[tuple[int, str, str, str, list, Document, Document]] = []

        for idx, doc_a_id, doc_b_id in uncached_pairs:
            doc_a: Document = self.corpus[doc_a_id]
            doc_b: Document = self.corpus[doc_b_id]

            # Randomize presentation order
            if self.randomize_order and random.random() < 0.5:
                presented_a, presented_b = doc_b, doc_a
                presented_order = [doc_b_id, doc_a_id]
            else:
                presented_a, presented_b = doc_a, doc_b
                presented_order = [doc_a_id, doc_b_id]

            # Build prompt based on doc_mode
            user_prompt: str = build_user_prompt_with_mode(
                self.query,
                presented_a,
                presented_b,
                mode=self.doc_mode,
                summaries=self.summaries,
                max_snippet_chars=self.max_snippet_chars,
            )
            prompts_data.append((idx, doc_a_id, doc_b_id, user_prompt, presented_order, presented_a, presented_b))

        # Check if LLM supports async batch
        if hasattr(self.llm, 'generate_batch_async'):
            # Batch all prompts together
            prompt_tuples: list[tuple[str, Optional[str]]] = [
                (data[3], system_prompt) for data in prompts_data
            ]
            responses: list[str] = await self.llm.generate_batch_async(
                prompt_tuples, max_tokens=self.max_tokens, temperature=0.0
            )
        else:
            # Fallback to sequential async
            async def single_generate(prompt: str) -> str:
                if hasattr(self.llm, 'generate_async'):
                    return await self.llm.generate_async(prompt, system_prompt, max_tokens=self.max_tokens)
                else:
                    return self.llm.generate(prompt, system_prompt, max_tokens=self.max_tokens)

            responses = await asyncio.gather(*[
                single_generate(data[3]) for data in prompts_data
            ])

        # Parse responses and update cache
        for (idx, doc_a_id, doc_b_id, _, presented_order, presented_a, presented_b), response in zip(prompts_data, responses):
            winner_label: Optional[str] = parse_winner(response)
            if winner_label is None:
                winner_label = "A"  # Default to first presented

            # Map winner label back to actual document ID
            if winner_label == "A":
                winner_id = presented_order[0]
            else:
                winner_id = presented_order[1]

            # Update cache
            cache_key: tuple[str, str] = (min(doc_a_id, doc_b_id), max(doc_a_id, doc_b_id))
            self._cache[cache_key] = winner_id

            # Create comparison record
            comparison = PairwiseComparison(
                comparison_id=str(uuid.uuid4()),
                query_id=self.query.query_id,
                doc_a_id=doc_a_id,
                doc_b_id=doc_b_id,
                presented_order=presented_order,
                winner_id=winner_id,
                reasoning=response,
                model=self.llm.name,
                timestamp=datetime.now(UTC),
            )
            self.comparisons.append(comparison)
            self.comparison_count += 1

            # Set result
            results[idx] = -1 if winner_id == doc_a_id else 1

        return results

    def compare_batch(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[int]:
        """Compare multiple document pairs in parallel (sync wrapper).

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            List of (doc_a_id, doc_b_id) pairs to compare.

        Returns
        -------
        list[int]
            List of comparison results.
        """
        # Use existing event loop if available, otherwise create one
        # This avoids the "Event loop is closed" error from repeated asyncio.run()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're already in an async context - this shouldn't happen in normal usage
            # but handle it gracefully
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.compare_batch_async(pairs))
                return future.result()
        else:
            # Create a new event loop and keep it open for reuse
            if not hasattr(self, '_loop') or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop.run_until_complete(self.compare_batch_async(pairs))


def merge_sort_with_llm(
    doc_ids: list[str],
    comparator: LLMComparator,
    progress_callback: Optional[callable] = None,
) -> list[str]:
    """Sort document IDs by relevance using merge sort with LLM comparisons.

    Merge sort is used because it has guaranteed O(n log n) complexity
    and is stable.

    Parameters
    ----------
    doc_ids : list[str]
        List of document IDs to sort.
    comparator : LLMComparator
        Comparator instance for LLM-based comparisons.
    progress_callback : callable, optional
        Callback function called after each comparison with current count.

    Returns
    -------
    list[str]
        Document IDs sorted by relevance (most relevant first).
    """
    if len(doc_ids) <= 1:
        return doc_ids.copy()

    def merge(left: list[str], right: list[str]) -> list[str]:
        """Merge two sorted lists."""
        result: list[str] = []
        i: int = 0
        j: int = 0

        while i < len(left) and j < len(right):
            cmp_result: int = comparator.compare(left[i], right[j])
            if progress_callback:
                progress_callback(comparator.comparison_count)

            if cmp_result <= 0:  # left[i] is more relevant or equal
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        # Append remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def sort(arr: list[str]) -> list[str]:
        """Recursively sort using merge sort."""
        if len(arr) <= 1:
            return arr

        mid: int = len(arr) // 2
        left: list[str] = sort(arr[:mid])
        right: list[str] = sort(arr[mid:])
        return merge(left, right)

    return sort(doc_ids)


def quicksort_with_llm(
    doc_ids: list[str],
    comparator: LLMComparator,
    progress_callback: Optional[callable] = None,
) -> list[str]:
    """Sort document IDs by relevance using quicksort with LLM comparisons.

    Uses median-of-three pivot selection for better average performance.

    Parameters
    ----------
    doc_ids : list[str]
        List of document IDs to sort.
    comparator : LLMComparator
        Comparator instance for LLM-based comparisons.
    progress_callback : callable, optional
        Callback function called after each comparison with current count.

    Returns
    -------
    list[str]
        Document IDs sorted by relevance (most relevant first).
    """
    if len(doc_ids) <= 1:
        return doc_ids.copy()

    arr: list[str] = doc_ids.copy()

    def compare_and_track(a: str, b: str) -> int:
        """Compare and optionally call progress callback."""
        result: int = comparator.compare(a, b)
        if progress_callback:
            progress_callback(comparator.comparison_count)
        return result

    def partition(low: int, high: int) -> int:
        """Partition array around pivot."""
        # Median-of-three pivot selection
        mid: int = (low + high) // 2

        # Sort low, mid, high
        if compare_and_track(arr[mid], arr[low]) < 0:
            arr[low], arr[mid] = arr[mid], arr[low]
        if compare_and_track(arr[high], arr[low]) < 0:
            arr[low], arr[high] = arr[high], arr[low]
        if compare_and_track(arr[high], arr[mid]) < 0:
            arr[mid], arr[high] = arr[high], arr[mid]

        # Use mid as pivot, move to high-1
        arr[mid], arr[high - 1] = arr[high - 1], arr[mid]
        pivot: str = arr[high - 1]

        i: int = low
        for j in range(low, high - 1):
            if compare_and_track(arr[j], pivot) < 0:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1

        arr[i], arr[high - 1] = arr[high - 1], arr[i]
        return i

    def quicksort_recursive(low: int, high: int) -> None:
        """Recursively sort array."""
        if low < high:
            # Use insertion sort for small subarrays
            if high - low < 10:
                for i in range(low + 1, high + 1):
                    key: str = arr[i]
                    j: int = i - 1
                    while j >= low and compare_and_track(arr[j], key) > 0:
                        arr[j + 1] = arr[j]
                        j -= 1
                    arr[j + 1] = key
            else:
                pivot_idx: int = partition(low, high)
                quicksort_recursive(low, pivot_idx - 1)
                quicksort_recursive(pivot_idx + 1, high)

    quicksort_recursive(0, len(arr) - 1)
    return arr


def quicksort_batched(
    doc_ids: list[str],
    comparator: LLMComparator,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> list[str]:
    """Sort document IDs using quicksort with batched parallel comparisons.

    This version batches all pivot comparisons in each partition step,
    allowing them to run in parallel via async API calls.

    Parameters
    ----------
    doc_ids : list[str]
        List of document IDs to sort.
    comparator : LLMComparator
        Comparator instance for LLM-based comparisons.
    progress_callback : callable, optional
        Callback function called after each batch with current count.

    Returns
    -------
    list[str]
        Document IDs sorted by relevance (most relevant first).
    """
    if len(doc_ids) <= 1:
        return doc_ids.copy()

    arr: list[str] = doc_ids.copy()

    def partition_batched(low: int, high: int) -> int:
        """Partition array around pivot using batched comparisons."""
        if high - low < 2:
            # Too small to partition meaningfully
            if high > low:
                pairs = [(arr[low], arr[high])]
                results = comparator.compare_batch(pairs)
                if progress_callback:
                    progress_callback(comparator.comparison_count)
                if results[0] > 0:  # arr[high] is more relevant
                    arr[low], arr[high] = arr[high], arr[low]
            return low

        # Use middle element as pivot
        mid: int = (low + high) // 2
        pivot: str = arr[mid]
        arr[mid], arr[high] = arr[high], arr[mid]  # Move pivot to end

        # Batch compare all elements to pivot
        elements: list[str] = [arr[i] for i in range(low, high)]
        pairs: list[tuple[str, str]] = [(elem, pivot) for elem in elements]

        results: list[int] = comparator.compare_batch(pairs)
        if progress_callback:
            progress_callback(comparator.comparison_count)

        # Partition based on results
        less: list[str] = []
        greater: list[str] = []
        for elem, cmp_result in zip(elements, results):
            if cmp_result < 0:  # elem is more relevant than pivot
                less.append(elem)
            else:
                greater.append(elem)

        # Rebuild array: less + pivot + greater
        idx: int = low
        for elem in less:
            arr[idx] = elem
            idx += 1
        pivot_pos: int = idx
        arr[pivot_pos] = pivot
        idx += 1
        for elem in greater:
            arr[idx] = elem
            idx += 1

        return pivot_pos

    def quicksort_recursive(low: int, high: int) -> None:
        """Recursively sort array."""
        if low < high:
            pivot_idx: int = partition_batched(low, high)
            quicksort_recursive(low, pivot_idx - 1)
            quicksort_recursive(pivot_idx + 1, high)

    quicksort_recursive(0, len(arr) - 1)
    return arr


def sort_documents_with_llm(
    doc_ids: list[str],
    llm: LLMBackend,
    query: Query,
    corpus: dict[str, Document],
    dataset_name: str,
    algorithm: str = "merge",
    max_snippet_chars: int = 1500,
    randomize_order: bool = True,
    minimal: bool = True,
    doc_mode: str = DOC_MODE_TITLE,
    summaries: Optional[dict[str, str]] = None,
    max_tokens: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> tuple[list[str], list[PairwiseComparison]]:
    """Sort documents by relevance using LLM pairwise comparisons.

    This is the main entry point for O(n log n) sorting using LLM.

    Parameters
    ----------
    doc_ids : list[str]
        List of document IDs to sort.
    llm : LLMBackend
        The LLM backend to use.
    query : Query
        The query to judge relevance against.
    corpus : dict[str, Document]
        Document corpus mapping.
    dataset_name : str
        Name of the dataset.
    algorithm : str, optional
        Sorting algorithm: "merge" or "quick". Default "merge".
    max_snippet_chars : int, optional
        Maximum snippet length. Default 1500.
    randomize_order : bool, optional
        Whether to randomize doc presentation order. Default True.
    minimal : bool, optional
        Use minimal prompts. Default True.
    doc_mode : str, optional
        Document representation: 'title', 'summary', or 'snippet'. Default 'title'.
    summaries : dict[str, str], optional
        Pre-computed document summaries for 'summary' mode.
    max_tokens : int, optional
        Max tokens for LLM response. Auto-detected based on model if not specified.
    progress_callback : callable, optional
        Callback function called after each comparison.

    Returns
    -------
    tuple[list[str], list[PairwiseComparison]]
        Sorted document IDs (most relevant first) and list of all comparisons made.
    """
    comparator: LLMComparator = LLMComparator(
        llm=llm,
        query=query,
        corpus=corpus,
        dataset_name=dataset_name,
        max_snippet_chars=max_snippet_chars,
        randomize_order=randomize_order,
        minimal=minimal,
        doc_mode=doc_mode,
        summaries=summaries,
        max_tokens=max_tokens,
    )

    if algorithm == "merge":
        sorted_ids: list[str] = merge_sort_with_llm(
            doc_ids, comparator, progress_callback
        )
    elif algorithm == "quick":
        sorted_ids = quicksort_with_llm(doc_ids, comparator, progress_callback)
    elif algorithm == "quick_batched":
        sorted_ids = quicksort_batched(doc_ids, comparator, progress_callback)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'merge', 'quick', or 'quick_batched'.")

    return sorted_ids, comparator.comparisons
