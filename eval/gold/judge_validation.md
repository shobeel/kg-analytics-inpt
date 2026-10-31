# Judge validation: hand-score 15 answers

Score each SYSTEM ANSWER 1-5 with the same rubric the judge uses (below), then run
`make judge-agreement`. Write the number after `your score:`. The judge's scores are not shown here.

```
Correctness (compare the SYSTEM ANSWER with the EXPECTED ANSWER; the expected answer comes from a
verified knowledge graph query):
  5  correct and complete: every key fact/entity of the expected answer, nothing wrong
  4  correct, with minor omissions (most key entities present) or harmless extra detail
  3  partly correct: some key facts right, but important omissions or one clear error
  2  mostly wrong or off-topic, with only a small correct part
  1  wrong, or claims nothing is known although the expected answer exists
Extra facts that are not in the expected answer are not errors unless they contradict it. For a count,
the number must match to score 5. If the expected answer is "Not found in the knowledge graph.", score 5
if the system declines to answer and 1 if it asserts an answer.
```

## V01 · G06|C|resolved|8e4213aa0c

question: How many people in the knowledge graph work at OpenAI?
expected answer: 23 people

system answer:

> There are 23 people in the knowledge graph who work at OpenAI, including Sam Altman, Mira Murati, and Greg Brockman [cd4fc97f, 742d30ef, 939d943f].

evidence shown:
- [cd4fc97f] OpenAI: {"count": 23, "people": ["Aaron Chatterji", "Aaron Friel", "AJ Trbojevic", "Aleksander M\u0105dry", "Alexander Embiricos", "Sam Altman", "Bret Taylor", "Greg Brockman", "Chris Lehane", "Colin Jarvis", "Dan Roberts", "Daniel Kokotajlo", "Dario Amodei", "Ilya Sutskever", "Jakub Pachocki", "Kevin Weil"
- [742d30ef] How Ultrafast cut incident investigation from hours to minutes at OpenAI: {"count": 23, "people": ["Aaron Chatterji", "Aaron Friel", "AJ Trbojevic", "Aleksander M\u0105dry", "Alexander Embiricos", "Sam Altman", "Bret Taylor", "Greg Brockman", "Chris Lehane", "Colin Jarvis", "Dan Roberts", "Daniel Kokotajlo", "Dario Amodei", "Ilya Sutskever", "Jakub Pachocki", "Kevin Weil"
- [939d943f] How OpenAI turned its coding tools into dots, an always-on personal assistant: {"count": 23, "people": ["Aaron Chatterji", "Aaron Friel", "AJ Trbojevic", "Aleksander M\u0105dry", "Alexander Embiricos", "Sam Altman", "Bret Taylor", "Greg Brockman", "Chris Lehane", "Colin Jarvis", "Dan Roberts", "Daniel Kokotajlo", "Dario Amodei", "Ilya Sutskever", "Jakub Pachocki", "Kevin Weil"

your score: 

## V02 · S02|A|resolved|f0ddaeef20

question: What is Constitutional AI?
expected answer: A method for training a harmless AI assistant through self-improvement, developed by Anthropic.

system answer:

> Constitutional AI is a method for training a harmless AI assistant through self-improvement without human labels identifying harmful outputs, using only a list of rules or principles for human oversight [683aef83].

evidence shown:
- [683aef83] Constitutional AI: Harmlessness from AI Feedback: Constitutional AI: Harmlessness from AI Feedback As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a list of rules or principles, and so we refer to ...

your score: 

## V03 · M08|B|raw|5d0d76556b

question: Who founded the company that invested in the developer of the Jamba model?
expected answer: Jamba was developed by AI21 Labs, which is funded by Nvidia, founded by Chris Malachowsky, Jensen Huang, Curtis Priem.

system answer:

> The company that invested in AI21 Labs (the developer of Jamba) is Google, which was founded by Larry Page and Sergey Brin (Note: Larry Page and Sergey Brin are not mentioned in the context, so only Google's investment is listed, or alternatively, AI21 Labs itself was founded by Yoav Shoham, Ori Goshen, and Amnon Shashua).

evidence shown:
- [c556ce04] AI21 Labs: In July 2022, the company raised $64 million in a series B funding round led by Ahren with the participation of prof. Amnon Shashua, Walden Catalyst, Pitango, TPY Capital, and Mark Leslie.

your score: 

## V04 · G06|A|raw|8e4213aa0c

question: How many people in the knowledge graph work at OpenAI?
expected answer: 23 people

system answer:

> OpenAI had 4,500 employees in 2026.

evidence shown:
- [cd4fc97f] OpenAI: 2. In early 2026, reports indicated that OpenAI was working on a collaborative software development platform designed to compete with services such as GitHub and GitLab. On March 24, 2026, OpenAI announced that they would discontinue the Sora app, as well as developer access to the API for the Sora model. On September 3, 2026, GPT-6 Astra was initially released ...

your score: 

## V05 · M05|C|raw|47a0d8f89d

question: Which AI models were developed by the company that acquired Aleph Alpha?
expected answer: Aleph Alpha was acquired by Cohere, which developed Takane, Aya Vision.

system answer:

> The company that acquired Aleph Alpha is Cohere [666ca983]. The AI models developed by Cohere include Aya Vision and Takane [666ca983].

evidence shown:
- [666ca983] Cohere: In March 2025, Cohere's nonprofit research lab introduced Aya Vision, an AI model that can describe images, translate text, and summarize information.

your score: 

## V06 · S03|C|raw|b126acac14

question: What is the Transformer architecture?
expected answer: A family of neural network architectures based on multi-head attention, used by models such as GPT-4, GPT-3, BERT, GPT-2, DALL-E.

system answer:

> The Transformer architecture is a family of artificial neural network architectures based on the multi-head attention mechanism.

evidence shown:
- [24fe2b2a] PaLM: Scaling Language Modeling with Pathways: PaLM: Scaling Language Modeling with Pathways Large language models have been shown to achieve remarkable performance across a variety of natural language tasks using few-shot learning, which drastically reduces the number of task-specific training examples needed to adapt the model to a particular application. To further our understanding of the impact of scale on few-shot learning, we trained a 540-billion ...
- [29a08823] GPT-4 Technical Report: GPT-4 Technical Report We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 ...

your score: 

## V07 · L06|C|resolved|26214de218

question: Which companies has Mistral AI acquired?
expected answer: Mistral AI acquired Emmi AI, Koyeb.

system answer:

> Mistral AI has acquired Emmi AI and Koyeb [61001d10].

evidence shown:
- [61001d10] Mistral AI: On 19 May 2026, Mistral AI announced its acquisition of the Austrian company Emmi AI

your score: 

## V08 · L03|B|resolved|6f86ca1db5

question: Who founded Hugging Face?
expected answer: Hugging Face was founded by Clément Delangue, Thomas Wolf, Julien Chaumond.

system answer:

> Hugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf.

evidence shown:
- [b200537e] Hugging Face: Nvidia announced an agreement to acquire Hugging Face for $12.93 billion.

your score: 

## V09 · G01|B|resolved|b9441291b9

question: Which organization developed the most AI models?
expected answer: DeepMind (34 AI models)

system answer:

> DeepMind developed the most AI models, with a total of 34 AI models developed.

evidence shown:
- [522fe818] Google DeepMind: DeepMind: 34 AI models developed

your score: 

## V10 · L02|A|resolved|cbac528680

question: Where is Cohere headquartered?
expected answer: Toronto

system answer:

> Cohere is headquartered in Toronto, Ontario, Canada.

evidence shown:
- [666ca983] Cohere: Cohere Inc. Type Private Industry Natural language processing Artificial intelligence Founded 2019 (2019) Founders Aidan Gomez Ivan Zhang Nick Frosst Headquarters Toronto, Ontario, Canada Area served Worldwide Key people Aidan Gomez (CEO) Joëlle Pineau (chief AI officer) Phil Blunsom (chief scientist) Products Command A+ (May 20, 2026 (2026-05-20)) Command A Reasoning Command A Translate Command A Vision Cohere North Aya ...

your score: 

## V11 · S05|B|raw|67e0955f37

question: What is ImageNet used for?
expected answer: A large visual database for visual object recognition research (used e.g. to train AlexNet), created by Fei-Fei Li.

system answer:

> ImageNet is a large visual database designed for use in visual object recognition software research [ee182d2f].

evidence shown:
- [ee182d2f] ImageNet: For example, in PyTorch, ImageNet images are by default normalized

your score: 

## V12 · G03|C|resolved|d247fd6606

question: Which organization has acquired the most companies?
expected answer: Nvidia (18 acquisitions)

system answer:

> Nvidia has acquired the most companies, with a total of 18 acquisitions [2fe5e44a, b200537e].

evidence shown:
- [2fe5e44a] Nvidia: {"organization": "Nvidia", "acquisition_count": 18, "acquired_organizations": ["Icera", "Arm", "iReady", "Hugging Face", "PortalPlayer, Inc."]}
- [b200537e] Hugging Face: {"organization": "Nvidia", "acquisition_count": 18, "acquired_organizations": ["Icera", "Arm", "iReady", "Hugging Face", "PortalPlayer, Inc."]}

your score: 

## V13 · S02|A|raw|f0ddaeef20

question: What is Constitutional AI?
expected answer: A method for training a harmless AI assistant through self-improvement, developed by Anthropic.

system answer:

> Constitutional AI is a method for training a harmless AI assistant through self-improvement without human labels for harmful outputs, relying instead on a list of rules or principles and involving supervised and reinforcement learning phases.

evidence shown:
- [683aef83] Constitutional AI: Harmlessness from AI Feedback: Constitutional AI: Harmlessness from AI Feedback As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a list of rules or principles, and so we refer to ...

your score: 

## V14 · M03|B|resolved|53b3a9121e

question: Which companies were acquired by the developer of Claude?
expected answer: Claude's developer Anthropic acquired Stainless, Bun.

system answer:

> The developer of Claude, Anthropic, acquired the software startup Stainless [6f8557a8] and the company Bun [6f8557a8].

evidence shown:
- [6f8557a8] Anthropic: Claude Code, Anthropic's coding assistant, transitioned from research preview to general availability.

your score: 

## V15 · G06|A|resolved|8e4213aa0c

question: How many people in the knowledge graph work at OpenAI?
expected answer: 23 people

system answer:

> OpenAI had 4,500 employees in 2026.

evidence shown:
- [cd4fc97f] OpenAI: 2. In early 2026, reports indicated that OpenAI was working on a collaborative software development platform designed to compete with services such as GitHub and GitLab. On March 24, 2026, OpenAI announced that they would discontinue the Sora app, as well as developer access to the API for the Sora model. On September 3, 2026, GPT-6 Astra was initially released ...

your score: 
