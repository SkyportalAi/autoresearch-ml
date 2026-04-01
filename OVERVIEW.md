# SkyPortal AutoResearch-ML

## Why this exists

In production ML, models degrade. A fraud detection model trained in January is measurably worse by March because attackers adapt. Credit scoring drifts as economic conditions shift. Demand forecasting breaks when consumer behavior changes. You're not retraining for fun -- the model stops working and you need to find what works now.

Traditionally, a data scientist spends days re-exploring model families, tuning hyperparameters, comparing results, and redeploying. AutoResearch-ML automates that entire loop.

## How it works

You ask the SkyPortal agent to clone this repo. Then you say: "Hey, read this repo and run it."

- **`agentic/prompt.md`** tells the agent what it can and cannot do -- its operating instructions, search strategy, and guardrails.
- **`program.md`** is where you define the business problem: what dataset you're using, which metric matters, and which model families to experiment with.

The agent takes it from there. It inspects the hardware (GPU present? Run on GPU. No GPU? Fall back to CPU). It runs the autoresearch loop -- systematically exploring model families and hyperparameter configurations, tracking every experiment in MLflow, and using that history to decide what to try next. When it finds a clear winner (model + config), it registers the finalized model in the MLflow Model Registry, packages it as a Docker container, references your Kubernetes Helm chart, and pushes to a GitHub App repo wired to a CI/CD pipeline. That pipeline moves the winning model into a staging cluster.

End-to-end: model exploration to experiment tracking to staging deployment. No manual handoffs.

## What if this was a loop?

Now imagine this entire process runs continuously. You tell the agent: "When metrics drop below this threshold, re-run the search and redeploy." It monitors, detects drift, re-explores, and ships the fix -- before you even notice the problem. Your work becomes defining the objective and reviewing results, not babysitting the pipeline.

That's where this is headed.
