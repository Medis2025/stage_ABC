# HPO-MoRE Candidate Pipeline

## Stage 1 / Stage 2 / Stage 3 Architecture

---

# 1. Overview

The HPO-MoRE Candidate Pipeline is a structured, multi-stage system for constructing a literature-grounded phenotype corpus and semantic representation space for Human Phenotype Ontology (HPO) terms.

The pipeline is organized into three sequential stages:

* **Stage 1 — Evidence Construction**
* **Stage 2 — Representation & Structural Modeling**
* **Stage 3 — Supervision-Oriented Corpus Generation**

The overall data flow is:

[
\text{HPO Terms}
\rightarrow
\text{Literature Evidence}
\rightarrow
\text{Phenotype Embeddings}
\rightarrow
\text{Cluster Topology}
\rightarrow
\text{Structured Training Corpus}
]

The design principle is to stabilize semantic structure before generating supervision, thereby reducing noise amplification and synthetic bias.

---

# 2. Stage 1 — Evidence Construction

## 2.1 Objective

Construct phenotype-specific evidence pools grounded in biomedical literature using automated retrieval and weak filtering mechanisms.

---

## 2.2 Input

For each phenotype ( h ):

* HPO ID
* Canonical name
* Optional synonyms

---

## 2.3 Workflow

### 2.3.1 Query Generation

Generate a multi-scale query set:

[
Q_h = { q_1, q_2, ..., q_k }
]

Query strategies include:

* Exact phenotype name
* Synonym expansion
* Pattern-based augmentation
* LLM-guided paraphrase expansion

---

### 2.3.2 Literature Retrieval

Using PubMed and PMC APIs:

[
D_h = \text{Retrieve}(Q_h)
]

Documents include:

* Title
* Abstract
* Full text (if available)
* Metadata (PMID, PMC ID)

---

### 2.3.3 Candidate Span Extraction

From each document:

[
C_h = { c_1, c_2, ..., c_n }
]

Extraction strategies:

* Sentence segmentation
* Context window expansion
* Pattern-based filtering

---

### 2.3.4 Weak Evidence Filtering

Each candidate span is evaluated via an LLM-based support classifier:

[
f(c_i, h) \rightarrow {\text{support}, \text{not_support}}
]

Each evidence unit stores:

* Text span
* Source reference
* Confidence score
* Explanation note

---

## 2.4 Output of Stage 1

A structured evidence pool:

[
\mathcal{E} = { E_{h_1}, E_{h_2}, ..., E_{h_N} }
]

Each ( E_h ) contains validated, phenotype-specific evidence spans.

This stage ensures literature grounding before representation learning.

---

# 3. Stage 2 — Representation & Structural Modeling

## 3.1 Objective

Transform heterogeneous evidence pools into stable phenotype-level embeddings and construct a global semantic topology.

---

## 3.2 Evidence Categorization

Evidence may be partitioned into categories:

* Definition evidence
* Domain literature evidence
* Medium-confidence evidence
* Weak evidence

---

## 3.3 Embedding Construction

Each evidence span is encoded:

[
\mathbf{e}_i = \text{Encoder}(e_i)
]

Pooling per phenotype:

[
\mathbf{v}_h =
\text{Fuse}\Big(
\text{Pool}(E_h^{def}),
\text{Pool}(E_h^{domain}),
\text{Pool}(E_h^{medium}),
\text{Pool}(E_h^{weak})
\Big)
]

Result:

[
\mathbf{v}_h \in \mathbb{R}^d
]

---

## 3.4 Similarity Computation

Cosine similarity:

[
\text{sim}(h_i, h_j) =
\frac{\mathbf{v}*{h_i} \cdot \mathbf{v}*{h_j}}
{|\mathbf{v}*{h_i}| |\mathbf{v}*{h_j}|}
]

---

## 3.5 Approximate Nearest Neighbor Construction

Using HNSW indexing:

* Efficient kNN graph construction
* Sublinear retrieval complexity
* Memory-efficient large-scale ontology support

---

## 3.6 Clustering

Cluster phenotype embeddings:

[
\mathcal{C} = { C_1, C_2, ..., C_K }
]

Cluster assignments can be frozen for downstream sampling.

---

## 3.7 Output of Stage 2

* Embedding matrix ( V \in \mathbb{R}^{N \times d} )
* kNN adjacency structure
* Cluster assignments
* Frozen semantic topology

This stage constructs a phenotype semantic manifold grounded in literature evidence.

---

# 4. Stage 3 — Supervision-Oriented Corpus Generation

## 4.1 Objective

Generate structured, weakly supervised training corpora guided by the semantic topology constructed in Stage 2.

---

## 4.2 Cluster-Aware Phenotype Set Sampling

For a seed phenotype ( h_s ):

1. Sample same-cluster phenotypes.
2. Sample cross-cluster phenotypes.
3. Construct phenotype sets:

[
S = { h_s, h_{i_1}, ..., h_{i_k} }
]

This balances semantic coherence and diversity.

---

## 4.3 LLM-Based Clinical Narrative Generation

Given phenotype set ( S ):

[
g(S) \rightarrow x
]

Where ( x ) is a clinical-style narrative describing co-occurring phenotypes.

---

## 4.4 Structural Gating Constraints

Generation is validated using:

* Minimum phenotype hits per line
* Mandatory seed inclusion
* Coverage ratio threshold
* Retry with keyword forcing

These constraints enforce semantic fidelity and reduce hallucination.

---

## 4.5 Final Output

A structured corpus:

[
\mathcal{T} = { (x_j, S_j) }
]

Each item contains:

* Generated text
* Associated phenotype set
* Cluster metadata
* Coverage statistics

This corpus can be used to train:

* Dual-encoder phenotype linking models
* Contrastive span-to-ID systems
* Rerankers
* Router-first multi-head NER architectures

---

# 5. Design Principles

The pipeline enforces:

1. Evidence grounding before representation learning.
2. Structural stabilization before supervision generation.
3. Separation between retrieval noise and embedding learning.
4. Topology-aware synthetic supervision.

This reduces:

* Supervision drift
* Synthetic overfitting
* Mode collapse in embedding space
* Hallucinated training signals

---

# 6. Theoretical Interpretation

The pipeline constructs a structured semantic transformation:

[
\text{Ontology}
\rightarrow
\text{Evidence Graph}
\rightarrow
\text{Semantic Manifold}
\rightarrow
\text{Supervised Corpus}
]

Rather than training directly on noisy span-label pairs, it first stabilizes the phenotype representation geometry, then derives supervision from structural properties of that geometry.

---

# 7. Scalability & Reproducibility

The architecture supports:

* Full-ontology scaling
* Modular encoder replacement
* Frozen cluster versioning
* Deterministic sampling
* Reproducible run configuration

---

# 8. Summary

The HPO-MoRE Candidate Pipeline consists of:

* **Stage 1:** Literature-grounded evidence construction
* **Stage 2:** Phenotype representation and structural modeling
* **Stage 3:** Structure-guided supervision corpus generation

Together, they form a closed semantic construction loop:

[
\text{HPO} \rightarrow
\text{Evidence} \rightarrow
\text{Representation} \rightarrow
\text{Topology} \rightarrow
\text{Supervision}
]

This enables scalable, weakly supervised phenotype modeling grounded in biomedical literature.
