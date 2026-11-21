# AI Disclosure Statement

I used Gemini and GitHub Copilot (Copilot SDK in VS Code) as technical assistants. I reviewed the changes and will verify that I can explain the submitted code.

## Assistance by part

- **A - code generation:** Copilot helped resolve the tokenizer's tie-breaking behavior. I will check it against the tokenizer tests and review the implementation.
- **B - code review:** Copilot identified the required attention class name and grader marker. I will verify the model architecture and parameter count.
- **C - code generation:** Copilot helped revise the training CLI, token-ID cache, schedule, and baseline. I will run the pipeline and inspect its logs.
- **D - code generation:** Copilot helped implement autoregressive generation with sampling filters and a KV cache. I will test cached and uncached generation and review the implementation.
- **E - concepts:** Copilot helped organize the remaining experiment work. I will run the experiments and report the measurements from my own logs.
- **F - concepts:** Copilot helped identify the report and disclosure requirements. I will write the final reflections using my own run results and commit history.

Gemini was also used for earlier debugging, architecture guidance, and documentation assistance. I will update this disclosure if I use additional AI assistance for the experiments or report.