# Towards Retrieval Augmented Generation in High-Energy and Astroparticle Physics

Official code repository for the paper:

**Towards Retrieval Augmented Generation in High-Energy and Astroparticle Physics**  
Jacky Kumar and Sajan Kumar

📄 **Paper:** https://arxiv.org/abs/2610.00891

## Overview

The rapidly growing literature in high-energy and astroparticle physics makes it increasingly difficult to perform comprehensive literature searches, identify relevant previous work, and determine open research directions.

This project develops an open-source **Retrieval Augmented Generation (RAG)** pipeline designed specifically for the high-energy physics and astroparticle physics literature.

Given a scientific query, the pipeline retrieves relevant papers from a large database of arXiv articles and uses a Large Language Model (LLM) to generate a concise, citation-grounded literature report.

The system is designed to assist researchers with tasks such as:

- focused literature searches,
- identifying relevant previous work,
- generating concise literature reviews,
- gap and novelty analysis,
- identifying potential research directions,
- and supporting peer review.

The goal is not to replace expert scientific judgment, but to provide an AI-assisted tool for efficiently navigating the rapidly growing scientific literature.

---

## Pipeline

The overall workflow is

```text
Scientific Query
       |
       v
+------------------+
| Dense Retrieval  |
|    Specter2      |
+------------------+
       |
       |             +------------------+
       +------------>| Sparse Retrieval |
                     |      BM25         |
                     +------------------+
                              |
                              v
                     +------------------+
                     | Reciprocal Rank  |
                     | Fusion (RRF)     |
                     +------------------+
                              |
                              v
                     +------------------+
                     | Cross-Encoder    |
                     | Reranking        |
                     +------------------+
                              |
                              v
                     +------------------+
                     | LLM Relevance    |
                     | Judge            |
                     +------------------+
                              |
                              v
                     +------------------+
                     | LLM Report       |
                     | Synthesis        |
                     +------------------+
                              |
                              v
                 Citation-grounded report
```

---

## Knowledge Base

The knowledge base is constructed from arXiv papers belonging to the

- `hep-ph` — High Energy Physics - Phenomenology
- `astro-ph.HE` — High Energy Astrophysical Phenomena

categories.

Paper titles and abstracts are embedded and stored in a vector database together with the corresponding metadata.

The dataset used in the paper contains approximately **250,000 scientific articles** related to these fields.

---

## Retrieval Architecture

The retrieval system combines complementary semantic and lexical search strategies.

### Dense Retrieval

Scientific papers are embedded using **SPECTER2**, an embedding model specifically designed for scientific documents.

Each document is represented using its

```text
Title + Abstract
```

and stored in a **ChromaDB** vector database.

Queries are independently embedded and compared with the stored paper embeddings using cosine similarity.

### Sparse Retrieval

In parallel, the pipeline performs lexical retrieval using **BM25**.

BM25 is particularly useful for queries containing specific terminology, particle names, models, observables, experiments, or other technical keywords that should be matched explicitly.

### Reciprocal Rank Fusion

The dense and sparse retrieval results are combined using **Reciprocal Rank Fusion (RRF)**.

This allows the system to exploit both

```text
semantic similarity
        +
keyword matching
```

without requiring the scores produced by the two retrieval methods to be directly comparable.

### Cross-Encoder Reranking

The fused candidate papers are reranked using the cross-encoder

```text
ms-marco-MiniLM-L-6-v2
```

which jointly processes the scientific query and candidate document to obtain a more accurate relevance score.

The paper demonstrates that this multi-stage retrieval strategy substantially improves retrieval performance compared with dense retrieval alone.

---

## LLM-based Relevance Evaluation

Retrieval is followed by an additional LLM-based relevance evaluation.

The LLM first acts as a **judge**, determining whether each retrieved paper is genuinely relevant to the scientific question.

Only the papers passing this relevance stage are supplied to the final report-generation step.

This separates the pipeline into two distinct reasoning stages:

```text
Retrieve papers
      |
      v
Judge relevance
      |
      v
Synthesize scientific report
```

---

## Report Generation

The selected papers are provided as context to an open-source Large Language Model.

In the paper, we use **Google DeepMind's Gemma-4-E4B**.

The model synthesizes the retrieved scientific literature into a concise report while grounding its statements in the retrieved papers and providing references to the original work.

A typical query could be:

```text
What are the current theoretical explanations of the
B-meson flavor anomalies?
```

The system retrieves relevant papers, evaluates their relevance, and produces a compact literature report with citations.

---

## Evaluation

We construct dedicated retrieval benchmarks for both

```text
hep-ph
astro-ph.HE
```

and evaluate the system using metrics including

- Recall@k
- Mean Reciprocal Rank (MRR)

Three retrieval configurations are compared:

```text
Dense
Dense + Sparse (Hybrid)
Dense + Sparse + Cross-Encoder Reranking
```

The results demonstrate that the complete **Hybrid + Rerank** pipeline significantly outperforms dense retrieval alone.

For example, for `hep-ph` claim-verification queries, Recall@10 improves from approximately

```text
Dense                 : 0.391
Hybrid                : 0.893
Hybrid + Rerank       : 0.958
```

demonstrating the importance of combining semantic retrieval, lexical retrieval, and reranking.

---

## Main Components

The project combines several modern information-retrieval and language-model techniques:

| Component | Method |
|---|---|
| Scientific embeddings | SPECTER2 |
| Vector database | ChromaDB |
| Dense retrieval | Cosine similarity |
| Sparse retrieval | BM25 |
| Retrieval fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Relevance filtering | LLM-as-a-Judge |
| Report generation | Gemma-4-E4B |
| Literature source | arXiv |

---

## Applications

The framework can be used for several scientific tasks:

### Literature Review

Generate focused summaries of research on narrowly defined topics.

### Gap Analysis

Determine what has already been studied and identify potentially unexplored directions.

### Novelty Analysis

Compare a research idea against the existing literature.

### Research Assistance

Quickly identify important papers when entering a new research topic.

### Peer Review

Assist referees and editors in identifying related literature and checking the positioning of submitted work.

---

## Citation

If you use this code or framework in your research, please cite:

```bibtex
@article{Kumar:2026rag,
    author = {Kumar, Jacky and Kumar, Sajan},
    title = {Towards Retrieval Augmented Generation in High-Energy and Astroparticle Physics},
    eprint = {2610.00891},
    archivePrefix = {arXiv},
    primaryClass = {astro-ph.HE},
    year = {2026}
}
```

---

## Paper

For details of the methodology, benchmarks, retrieval evaluation, and example generated reports, see:

**J. Kumar and S. Kumar,  
"Towards Retrieval Augmented Generation in High-Energy and Astroparticle Physics"**

arXiv:2610.00891

https://arxiv.org/abs/2610.00891

---

## License

Please see the repository license for information regarding use and distribution.
