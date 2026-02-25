# Phase A – Stage 3 Pipeline Report

**From Rule-Based Recall to Semantic Locking and Constraint Framing**

---

## 0. Pipeline Overview

This system decomposes clinical text processing into **recall → narrowing → locking → generation**, explicitly separating responsibilities to ensure both coverage and controllability.

```
Phase A    : Rule-based Candidate Recall
Stage 2   : Structural Filtering & Candidate Narrowing
Stage 3   : Semantic Locking & Constraint Framing
Stage 4   : Case Generation / Ontology Reasoning
```

The combined objective of **Phase A – Stage 3** is:

> **To progressively eliminate ambiguity caused by co-occurrence while preserving recall, and to construct a stable, auditable semantic constraint frame before any case-level generation or reasoning.**

---

## 1. Phase A: Rule-Based Candidate Recall

### 1.1 Objective

Phase A is designed as a **recall-first** stage. Its purpose is not semantic understanding, but exhaustive signal capture.

> **Phase A aims to capture all textual signals that *might* correspond to case-level facts, without attempting to interpret them.**

Design principles:

* Maximize recall
* Prefer false positives over false negatives
* Defer semantic decisions to later stages

---

### 1.2 Input

* PubMed Central (PMC) full-text XML
* Abstracts, case sections, and narrative paragraphs

---

### 1.3 Techniques Used in Phase A

#### (1) Rule-Based High-Recall Extraction

Explicit, interpretable rules are used to extract candidate signals, including:

* **Measurements**

  * ECG intervals (ms)
  * Blood pressure (mmHg)
  * Heart rate (bpm)
  * Laboratory values (mmol/L, mg/dL)
  * Percentage values (e.g., EF)

* **Temporal / Frequency Expressions**

  * Durations (days, weeks, months)
  * Frequencies (every X days, episodes)

* **Reference Ranges**

  * Numeric lower–upper bounds appearing in context

These rules ensure coverage without asserting meaning.

---

#### (2) Context Localization (Sentence / Clause Level)

To avoid cross-sentence contamination, Phase A localizes context using:

* Sentence-level segmentation
* Clause-first extraction for measurements

This guarantees that each candidate carries only the **minimal sufficient context** required for downstream interpretation.

---

#### (3) Anchor Signals (Weak Semantics)

Each candidate records weak semantic anchors such as:

* ECG
* EF / LVEF
* LAB
* BP

Anchors do **not** imply conclusions; they serve as directional hints for later stages.

---

### 1.4 Output of Phase A

Phase A outputs a **raw candidate set**, where each candidate includes:

* Surface text span
* Structured value and unit
* Anchor hits
* Broad semantic label options
* Localized context

Key property:

> **Phase A output explicitly allows ambiguity, redundancy, and noise.**

---

## 2. Stage 2: Structural Filtering & Candidate Narrowing

### 2.1 Objective

Stage 2 acts as the **final rule-based firewall** before semantic modeling.

> **Its goal is to reduce noise while preserving all candidates that are structurally plausible clinical facts.**

---

### 2.2 Core Responsibilities

#### (1) Candidate Filtering

Remove candidates that are:

* Purely numeric without anchors
* Clearly non-clinical temporal or percentage mentions

Retain candidates supported by:

* Valid units
* Anchor signals
* Localized clinical context

---

#### (2) Candidate Space Narrowing

Broad semantic options from Phase A are narrowed. For example:

```
["QTc", "QT", "PR", "QRS", "OTHER"]
→
["QTc", "QT"]
```

Principles:

* Only shrink the candidate set
* Never select a final label

---

### 2.3 Output of Stage 2

Stage 2 produces **structurally valid but semantically unresolved candidates**, which are the *only* inputs permitted to reach Stage 3.

---

## 3. Stage 3: Semantic Locking & Constraint Framing

Stage 3 performs the final transition from **possible interpretations** to **fixed semantic facts**.

---

### 3.1 Core Objective

> **Resolve ambiguity arising from co-occurrence and construct a semantic constraint frame that bounds downstream case generation.**

Stage 3 does not discover new information; it fixes interpretation.

---

### 3.2 Stage 3 Pipeline Structure

Stage 3 consists of three sequential submodules.

---

#### 3.2.1 Phase B: Semantic Decision

**Purpose**
Select the most appropriate semantic label from a pre-narrowed candidate set.

**Technique**

* Lightweight LLM (e.g., Qwen)
* Used strictly as a **constrained classifier**, not a generator

**Decisions Made**

* Best semantic label (within candidates)
* Validity of the candidate as a clinical fact
* Polarity (HIGH / LOW / NORMAL / UNKNOWN)

**Output Constraint**

* Plain-text index and flags only
* JSON reconstructed externally for determinism

---

#### 3.2.2 Semantic Framing (Constraint Construction)

This is the **core engineering contribution** of Stage 3.

Each resolved fact is assigned:

##### Semantic Slot

Defines the category of the fact:

* ECG
* LAB
* CARDIAC_FUNCTION
* VITAL
* TEMPORAL

##### Scope

Defines applicability within the case:

* patient
* episode
* treatment
* followup

> Slot and scope prevent downstream generation from misplacing facts.

---

#### 3.2.3 Conflict Resolution & Canonicalization

To ensure stability before generation, Stage 3 applies deterministic rules:

* Deduplication of equivalent facts
* Conflict resolution (e.g., nearest or most abnormal value)
* Isolation of temporal expressions from measurements

No LLM is used in this step.

---

### 3.3 Output of Stage 3

Stage 3 outputs a **Case Constraint Frame**:

> A structured, auditable set of resolved facts that strictly bounds downstream generation.

This frame is not text; it is a semantic control object.

---

## 4. Why the Full Phase A – Stage 3 Pipeline Is Necessary

| Missing Stage | Consequence                                        |
| ------------- | -------------------------------------------------- |
| Phase A       | Incomplete case evidence due to low recall         |
| Stage 2       | Noise directly contaminates semantic modeling      |
| Stage 3       | Co-occurrence ambiguity propagates into generation |

---

## 5. One-Sentence Summary

> *We employ a staged pipeline from rule-based recall to semantic locking. Phase A maximizes coverage of potential clinical facts, Stage 2 structurally narrows the candidate space, and Stage 3 resolves co-occurring ambiguity while constructing a constraint frame that bounds downstream case generation, ensuring fidelity, controllability, and auditability.*
