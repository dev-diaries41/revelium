# Revelium

Revelium is a business-focused LLM analytics platform that turns real user prompts into actionable insights, optimized templates, cost analysis, and compliance monitoring.

## Design choices
* Model: MiniLM-6 quant onnxruntime model
* Max token length: MiniLM-l6 tokenizer used max length of 512 instead of 182
    - Pro: This improves cluster accuracy by avoiding loss of context when embedding large prompts
    - Con: This increases indexing time (batch embed generation and storage) by up to 4x
    - Bench marks: 50 prompts of character length ~ 500, using the 512 tokenizer length takes ~3s, on CPU with 16 cores
* Since cluster accuracy is pivotal for maximising accuracy and usefulness of some of the features below, 512 is used instead of 128 even though in increases processing time.
    
## Features

1. **Cluster Prompts & Label by Use Case**

   * Automatically groups similar prompts and labels them to reveal the **most popular business use cases**.

2. **Token Usage by Cluster**

   * Tracks **token consumption and cost per use case**, helping teams identify expensive workflows and optimize budgets.

3. **Optimized Prompt Templates**

   * Analyses prompts for performance and cost efficiency, then **generates reusable, optimized templates** for faster adoption and reduced costs.

4. **(Optional) Prompt Playground**

   * Allows users to **test prompts interactively**, compare outputs across multiple LLM providers, and validate templates in real time.

4. **(Optional) Prompt abuse Detector**
    * Identifies prompt which are against rules and auto-bans, warns, or escalates to manual review
    * This is useful for industries with strict regulations
    
    