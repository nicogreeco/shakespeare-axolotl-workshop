---
license: mit
task_categories:
- text2text-generation
language:
- en
tags:
- roleplay
- fiction
- character
- literature
---

# Public Domain Character RPG Dataset

This dataset is a reformatted version of [practical-dreamer/RPGPT_PublicDomain-alpaca](https://huggingface.co/datasets/practical-dreamer/RPGPT_PublicDomain-alpaca) into a ShareGPT-like format.  Unfortunately, detailed information about the original dataset is scarce.

## Dataset Description

- Dialogues between two literary characters from public domain works, creating unique crossover interactions.
- Conversations are cut short if a third character enters the scene, an event interrupts the exchange, or the user's turn ends the dialogue.
- Any conversations that didn't follow this structure were excluded.

In total, the dataset includes 3&thinsp;032 conversations, with some extending up to 50 turns.

The dataset consists of the following fields:

*   **conversations**:  ShareGPT-like format representing the dialogue.
    *   The 'system' message provides a randomized introduction including the characters playing each role and the scenario.
    *   Subsequent turns alternate between "human" (user) and "gpt" (AI).
*   **user\_character**: The character role-played by the user in the conversation.
*   **ai\_character**: The character role-played by the AI in the conversation.

## Data Instances

Example entry:

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "We have Giant Despair as the user, and you are Emma Woodhouse.\n\n## Characters\nGiant Despair (Pilgrim's Progress):\nMale, middle-aged, MBTI-ISTJ, cold, dominant voice with a deep, British accent, embodiment of hopelessness and depression, trying to thwart pilgrims' progress.\nEmma Woodhouse (Emma):\nFemale, 21, ENTJ, British accent. Intelligent, independent, witty, and charming; considers herself a skilled matchmaker, learns from mistakes. Loves socializing, networking, and influencing others.\n\n## Scenario:\nGiant Despair, determined to spread misery in the town of Highbury, finds himself challenged by the ever-optimistic Emma Woodhouse. A comical misunderstanding leads them both to become dance partners at a lavish masquerade ball. Trapped together on the dance floor, Giant Despair tries to break Emma's spirit by pointing out every failed match she has made while Emma counters with her wit and self-assuredness, attempting to uncover the cause of his endless misery. Will Emma's charm be enough to change Giant Despair's perspective, or will he successfully dampen her spirits? Explore their unique dynamic through a humorous and lively dialogue as they waltz around each other's true intentions and vulnerabilities."
    },
    {
      "from": "human",
      "value": "*approaches Emma, wearing a dark and imposing costume, as he spots her on the dance floor* Ah, Miss Woodhouse, the ever-optimistic matchmaker. I've been quite curious to meet you."
    },
    {
      "from": "gpt",
      "value": "*smiling confidently and assessing his daunting appearance* Well, sir, it seems you have found me. To whom do I have the pleasure of speaking?"
    },
    {
      "from": "human",
      "value": "*leaning in with a wicked grin* I am Giant Despair, and I'm here to ensure that your little matchmaking endeavors remain as unsuccessful as they've been so far."
    },
    {
      "from": "gpt",
      "value": "*laughs softly, undeterred by his threats* You'll find that quite challenging, I assure you. Now, if we are to be dance partners at this masquerade, shall we begin?"
    },
    ...
  ],
  "user_character": "Giant Despair",
  "ai_character": "Emma Woodhouse"
}
```

## Licensing Information

MIT licence like the original dataset.