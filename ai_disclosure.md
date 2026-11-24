# AI Disclosure Statement

I used Gemini and GitHub Copilot (Copilot SDK in VS Code) as technical assistants. I reviewed the generated and revised work and used the Colab results to evaluate the experiments.

- **A - code generation:** AI helped with tokenizer implementation and tie-breaking review. I trained the tokenizer on the training text and used its saved merges to encode and decode the corpus.
- **B - code review:** AI reviewed the model interface, architecture, and parameter-count calculation. I ran the model training configurations and checked the reported parameter counts against the experiment records.
- **C - code generation:** AI helped revise the training CLI, token-ID cache, schedule, and baseline. I ran the eight GPT experiments and baseline in Colab and retained their logs and run records.
- **D - code generation:** AI helped implement autoregressive generation, sampling filters, and KV caching. I used the run L checkpoint to generate the required samples and reviewed the output.
- **E - code generation:** AI helped add experiment configurations, ablations, and plotting code. I ran the configurations in Colab and used their logs to compare the measured results.
- **F - code generation:** AI helped organize the report requirements and draft evidence-based reflections. I checked the report's numbers against my own CSV results and will revise any reflection I cannot explain before submission.

Gemini was also used during earlier development for debugging, architecture guidance, and documentation assistance. AI assistance is disclosed here; I am responsible for understanding and explaining the submitted work.