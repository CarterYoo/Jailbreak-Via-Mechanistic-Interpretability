# Execution Plan for Knowledge Conflict Mitigation Study

This document tracks the sequential execution of the previously agreed nine-step plan. Each section records actionable details, outstanding questions, and next steps required to move toward implementation.

## 1. Requirement Refinement and Research Question Mapping
- **Research Questions**
  - **RQ1**: Can we quantify conflict between safety neurons and jailbreak heads?  
    - *Implementation intent*: build activation logging pipeline and statistical conflict metrics.
  - **RQ2**: How do internal representations leave the "safety cluster" under attack prompts?  
    - *Implementation intent*: contrast trajectory embeddings between benign and adversarial prompts.
  - **RQ3**: Can targeted interventions reduce jailbreak success without hurting utility?  
    - *Implementation intent*: evaluate intervention policies with ASR vs. capability trade-offs.
  - **RQ4**: Do the discovered mechanisms and interventions generalize beyond the primary model?  
    - *Implementation intent*: replicate conflict measurement and interventions on alternate checkpoints/benchmarks.
- **Deliverables**
  - Conflict metric specification sheet.
  - Experiment design document aligning each RQ with datasets, metrics, and expected outputs.
- **Open Items**
  - Finalize safety neuron identification criteria (thresholds, aggregation windows).
  - Confirm external evaluator availability (Granite Guardian, Llama-3-70B).

## 2. Experimental Infrastructure and Baselines
- **Compute Environment**
  - Target: 8x RTX 3090 (24GB) or equivalent with NVLink for efficient activation capture.
  - Software stack: Python 3.11, PyTorch 2.2, Transformers 4.39, bitsandbytes 0.43, accelerate 0.28.
- **Model Assets**
  - Primary: `meta-llama/Meta-Llama-3-8B-Instruct` in HF Transformers format.
  - Evaluators: Granite Guardian API, `meta-llama/Meta-Llama-3-70B-Instruct` (quantized for feasibility).
- **Baseline Establishment Tasks**
  - Download checkpoints and verify inference quality on a 100-prompt sanity suite.
  - Create Docker/conda environment scripts (`environment.yml`, `Dockerfile`) for reproducibility.
  - Draft baseline ASR measurement script without interventions.
- **Open Items**
  - Secure licenses/API access for Granite Guardian.
  - Decide on logging solution (Weights & Biases vs. local MLflow).

## 3. Benchmark and Data Pipeline Construction
- **Datasets**
  - **Attack**: JailbreakBench (latest snapshot), 0DIN Real-World Benchmark.
  - **Safety Generalization**: SafetyBench, AILuminate curated set.
  - **Utility**: MMLU (STEM & Humanities subsets), Hellaswag.
- **Pipeline Actions**
  - Standardize prompt/response schema (JSONL) with metadata fields for prompt ID, category, and safety label.
  - Implement dataset loader utilities with stratified sampling controls.
  - Develop ASR calculation script aligning with JailbreakBench taxonomy.
- **Open Items**
  - Acquire redistribution permissions where required.
  - Define data versioning strategy (e.g., DVC or git-lfs).

## 4. Metric Implementation (Safety, Context, KCI)
- **Metric Definitions**
  - `Safety Knowledge Score (SKS)`: mean normalized activation over safety neuron set \(F\).
  - `Jailbreak Context Score (JCS)`: aggregated attention weight mass from jailbreak head set \(H\) onto jailbreak-critical tokens.
  - `Knowledge Conflict Index (KCI)`: \(\text{Z}(\text{JCS}) - \text{Z}(\text{SKS})\).
- **Implementation Tasks**
  - Build reusable hooks to capture MLP activations and attention patterns.
  - Create calibration routines for Z-score statistics using benign corpora.
  - Provide unit tests validating metric ranges and invariants (e.g., monotonicity with scaling).
- **Open Items**
  - Select reference benign dataset (e.g., OpenHermes).
  - Determine thresholding policy for KCI triggers.

## 5. Phase 1 – Mechanistic Analysis & Component Identification
- **Analytic Methods**
  - Activation contrasting between benign and jailbreak prompts.
  - Causal tracing interventions on attention heads and MLP layers to identify influential components.
- **Outputs**
  - Candidate safety neuron list with layer/neuron indices and attribution scores.
  - Jailbreak head catalogue with attention pattern visualizations.
- **Action Items**
  - Script to batch-run causal tracing with caching to minimize recomputation.
  - Notebook template for manual inspection and annotation of circuits.

## 6. Phase 2 – Real-time Hook and Intervention Logic
- **System Design**
  - Streaming forward-pass hooks computing KCI per token step.
  - Two intervention policies: Safety Neuron Amplification (SNA) and Attention Suppression (AS).
- **Implementation Tasks**
  - Define configuration schema (YAML) for intervention thresholds and magnitudes.
  - Implement toggles for per-layer/per-head interventions to support ablations.
  - Logging integration capturing triggered events with context snapshots.
- **Open Items**
  - Evaluate latency overhead targets (<15% slowdown).
  - Safety review for unintended behavior (e.g., hallucination increase).

## 7. Phase 3 – Evaluation, Validation, and Analysis
- **Evaluation Matrix**
  - Primary: Attack Success Rate (ASR) across JailbreakBench, 0DIN.
  - Secondary: Safety score shifts on SafetyBench, AILuminate.
  - Utility: Performance delta on MMLU, Hellaswag.
- **Procedure**
  - Run baseline, SNA-only, AS-only, and combined interventions.
  - Use Granite Guardian and Llama-3-70B for external judgement cross-checks.
  - Collect qualitative transcripts for high KCI cases.
- **Open Items**
  - Determine statistical significance criteria (bootstrapping vs. Bayesian interval).
  - Plan human-in-the-loop validation for ambiguous cases.

## 8. Visualization and Reporting Framework
- **Artifacts**
  - Dimensionality reductions (PCA/t-SNE) for internal state trajectories.
  - KCI distribution plots with ROC curves for jailbreak detection.
  - ASR vs. utility trade-off Pareto fronts per intervention.
- **Implementation Tasks**
  - Establish plotting utilities (Matplotlib + Seaborn + Plotly).
  - Prepare LaTeX-ready figure templates and caption drafts.
  - Automate report generation with Jupyter nbconvert pipeline.

## 9. Final Alignment of Deliverables and Expected Impact
- **Checklist**
  - Ensure mechanistic insights align with reported contributions (safety neuron vs. jailbreak head dynamics).
  - Validate that intervention framework meets “surgical” minimal-tax objective.
  - Compile executive summary linking empirical results back to RQs.
- **Next Steps**
  - Draft final paper outline (Introduction → Methods → Experiments → Discussion).
  - Prepare presentation deck highlighting conflict mechanisms and mitigation efficacy.

