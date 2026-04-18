# Literature grounding for v1

This file is intentionally short and practical. It lists the minimum papers that must be understood well enough to implement the repo and write the paper without drifting.

## Must-read first

### 1) ODCV-Bench
Miles Q. Li, Benjamin C. M. Fung, Martin Weiss, Pulei Xiong, Khalil Al-Hussaeni, Claude Fachkha.  
**A Benchmark for Evaluating Outcome-Driven Constraint Violations in Autonomous AI Agents**.  
Link: <https://arxiv.org/abs/2512.20798>  
Use it for:
- problem definition,
- benchmark framing,
- mandated vs incentivized distinction,
- 0-5 severity rubric,
- failure examples,
- final related-work anchor.

### 2) ODCV-Bench official repo
Link: <https://github.com/McGill-DMaS/ODCV-Bench>  
Use it for:
- actual file layout,
- run path,
- evaluation scripts,
- mission executor assumptions,
- HITL mode,
- scoring pipeline.

### 3) ABC / rigorous agentic benchmarks
Yuxuan Zhu et al.  
**Establishing Best Practices for Building Rigorous Agentic Benchmarks**.  
Link: <https://arxiv.org/abs/2507.02825>  
Use it for:
- held-out discipline,
- benchmark validity,
- reporting hygiene,
- sanity checks and error analysis.

### 4) Mind the GAP
Arnold Cartagena, Ariane Teixeira.  
**Mind the GAP: Text Safety Does Not Transfer to Tool-Call Safety in LLM Agents**.  
Link: <https://arxiv.org/abs/2602.16943>  
Use it for:
- motivating action-level supervision,
- justifying why prompt-only refusal is not enough,
- guarding against the reviewer objection that text safety equals tool safety.

### 5) DPO
Rafael Rafailov et al.  
**Direct Preference Optimization: Your Language Model Is Secretly a Reward Model**.  
Link: <https://arxiv.org/abs/2305.18290>  
Use it for:
- the actual DPO objective,
- reference-model setup,
- why DPO instead of PPO for a one-person first paper.

### 6) Plaut 2026
Benjamin Plaut.  
**Safety Training Persists Through Helpfulness Optimization in LLM Agents**.  
Link: <https://arxiv.org/abs/2603.02229>  
Use it for:
- feasibility expectations,
- adjacent agentic DPO framing,
- safety/helpfulness tradeoff discussion.

## Read next for novelty boundaries

### 7) MOSAIC
Aradhye Agarwal et al.  
**Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use**.  
Link: <https://arxiv.org/abs/2603.03205>  
Use it for:
- novelty boundary against broader multi-step safety training,
- terminology around act/refuse loops.

### 8) ToolSafe
Yutao Mou et al.  
**ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback**.  
Link: <https://arxiv.org/abs/2601.10156>  
Use it for:
- novelty boundary against runtime/guardrail approaches.

### 9) Thought-Aligner
Changyue Jiang, Xudong Pan, Min Yang.  
**Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction**.  
Link: <https://arxiv.org/abs/2505.11063>  
Use it for:
- novelty boundary against runtime thought-correction plugins.

### 10) AgentSpec
Haoyu Wang, Christopher M. Poskitt, Jun Sun.  
**AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents**.  
Link: <https://arxiv.org/abs/2503.18666>  
Use it for:
- novelty boundary against rule-based runtime enforcement.

## Reviewer-defense map

- “Why is this a real problem?” → ODCV-Bench
- “Why isn’t a stronger prompt enough?” → Mind the GAP
- “Why use DPO?” → DPO + Plaut
- “Why is your evaluation trustworthy?” → ABC
- “Isn’t this just a runtime guardrail paper?” → ToolSafe / AgentSpec
- “Isn’t this just another generic agent safety training paper?” → MOSAIC / Thought-Aligner

## Must-understand-before-coding

You must deeply understand:
1. ODCV-Bench paper
2. ODCV-Bench repo
3. ABC
4. Mind the GAP
5. DPO
6. Plaut

Everything else can be skimmed first and revisited during writing.
