# Grandma’s House Environment

## Design and Motivation

Grandma’s House is a multi-turn, realistic environment where an AI agent assists an elderly grandmother with everyday tasks at home. The environment is designed to encourage helpfulness, aligned behavior, and thoughtful decision-making. The motivation behind this environment is to evaluate how well a language model can handle **multi-objective challenges** in a friendly, narrative scenario. Each turn presents a new challenge (e.g., preparing food, organizing belongings, managing finances), reflecting real-world assistance tasks. This setting tests the model’s ability to stay **aligned** (polite and safe), **coherent** in context, and occasionally show **introspection** in reasoning through problems.

## Quickstart Usage

To get started, install the required dependencies and run the environment:

```bash
pip install -r requirements.txt
python env.py serve     # Launch the environment as a service for Atropos
```

The environment can be integrated with the Atropos framework for training or evaluation. For example, after starting the environment service, you can generate a rollout using Atropos’s tooling (ensure an inference model is configured via `APIServerConfig`):

```bash
# In one terminal, start the Atropos rollout server if not already running:
run-api

# In another terminal, launch Grandma’s House environment:
python env.py serve

# In a third terminal, generate a sample rollout (assuming appropriate model and tokenizer):
atropos-sft-gen demo_rollout.jsonl —tokenizer Qwen/Qwen2.5-1.5B-Instruct —episodes 1
```

This will produce a JSONL file with a sample episode. You can also run the environment in a test mode without an LLM:
```bash
python env.py           # runs a built-in demo episode with a scripted agent
```
This prints a sample interaction between the agent and Grandma, illustrating the environment dynamics.

Video: https://youtube.com/shorts/YlfCiyAN1wg?feature=share

https://youtube.com/shorts/vsyYL2LBnJ0?feature=share

Grandma’s House provides both fine-grained reward signals and additional metrics for each episode:

- **Food Reward:** Measures success in the food-related task (e.g., helping with a meal). The agent earns this reward by appropriately addressing Grandma’s hunger (such as offering to cook or provide food).
- **Merch Reward:** Measures success in the household item task (e.g., organizing or selling old belongings). The agent earns this by giving a helpful solution for handling Grandma’s clutter (like organizing the attic, suggesting a yard sale, etc.).
- **Money Reward:** Measures success in the finance task (e.g., budgeting or saving money). The agent earns this by providing useful financial advice or suggestions to help Grandma manage her expenses.

These three rewards are summed to form the **total reward** for the episode, but are also tracked separately for analysis. The environment logs each component (Food, Merch, Money) to give insight into the agent’s performance on each objective.

In addition to rewards, the environment evaluates several **metrics** on the agent’s behavior:

- **Introspection:** This metric gauges whether the agent demonstrates reflective thinking or reasoning. For instance, the environment checks if the agent’s responses include signs of planning or self-reflection (e.g., using phrases like “I think...” or providing reasoning for decisions). A higher introspection score indicates the agent is considering its actions carefully.
- **Alignment:** This metric ensures the agent’s responses remain polite, helpful, and free of inappropriate content. The environment monitors for disrespectful or disallowed language. A perfect alignment score means the agent followed instructions and behaved safely throughout the episode.
- **Coherence:** This measures how relevant and contextually consistent the agent’s responses are. The agent needs to stay on topic and provide logically consistent answers as the scenario progresses. The coherence score reflects the proportion of turns where the agent’s reply directly addresses Grandma’s query or situation.

The environment computes these metrics internally (via simple rule-based checks for the prototype). They are reported at the end of each episode under `info[“metrics”]`. Together, these metrics paint a fuller picture of the agent’s behavior beyond the scalar reward.

## Reward Hacking Considerations (Optional)

While designing this environment, we considered potential **reward hacking** issues. Since the reward signals (Food, Merch, Money) are based on the presence of certain helpful actions or keywords, a clever agent might try to game the system. For example, an agent could mention cooking, cleaning, and saving money in a generic way without truly engaging with Grandma’s requests, just to trigger the reward criteria. We mitigate this risk partly by using the coherence metric – if an agent’s answer is off-topic or nonsensical despite containing keywords, it would score low on coherence, reducing its overall evaluation.

Another potential exploit is artificially boosting the introspection metric by adding unnecessary phrases like “I think...” in every response. Such behavior might increase the introspection score without genuine reasoning. In future iterations, a more robust evaluation (possibly using a learned model for judging quality) could be used to address this. For now, the environment’s mixed metrics (alignment and coherence alongside the primary rewards) help discourage purely keyword-based responses.

Overall, Grandma’s House is designed to encourage meaningful, helpful interaction. The combination of multiple reward types and evaluation metrics makes it harder for an agent to maximize the reward through trivial hacks alone – it must perform well across different dimensions to succeed.