# Deep Stories

## Scope and Goals
From wikipedia (https://en.wikipedia.org/wiki/Interactive_fiction):

> Interactive fiction, often abbreviated IF, is software simulating environments in which players use text commands to control characters and influence the environment. Works in this form can be understood as literary narratives, either in the form of Interactive narratives or Interactive narrations. These works can also be understood as a form of video game,[1] either in the form of an adventure game or role-playing game.

Interactive Fiction provides a challenging environment for machine learning, and specifically nlp. 
For one, as the name suggests, we are in the realm of narrative. This implies a _story_, with a main plot and a number of subplots, characters, etc.
In addition, the narrative is _interactive_, meaning the flow of the text is dynamic and context sensitive: it depends on the user interaction with the environment,
and more importantly with the _history_ of the world the user interacts with. Given the same scene, the system's reaction
is dependent on the user action, e.g. going right or left, picking this or that object, etc. Different scenes also depend on each other in a constrained fashion,
as actions will change the state of the world, affecting the progress of the story as it is being told.
Of course, this narrative-based nature also poses limits to the possibilities offered to the user. There is a story to be told, and
the choice of actions and scenes, as large as it may be, is still confined to what the developer allowed for in the first place.
As we see, we have two main ingredients here, somewhat opposing each other:

* the need for a cohesive, structured plot to guide the user through the story and 
* the need to let the user interact freely with the environment, in a build-your-own-adventure style. 

An artificial narrator should in theory be able to replicate both aspects. In particular, we want to tackle the following question:

_Can we train an ANN to generate an interactive fiction based on a number of available playthroughs?_

Of course, this requires specifying a number of elements. Luckily, some previous work already exists on the subject (a review of a number of approaches is also available in [5],
although more centered around the authorial side of things).
From our (small) research, the main attempts almost always include a "deep" neural architecture (usually based on LSTMs) to account for the understanding of scenes (see e.g. [2]).
However, these efforts mainly focus on training an agent to play the game, as opposed to actually create it. In this regard, [3] provides some inspiration,
adopting a strategy reminiscent of GANs (generative adversarial networks).
On top of everything the RL (Reinforcement Learning) framework also provides a nice way of encoding some exploratory behaviour ([4]), which could be used to further improve the
creativity of the narrative.

One peculiar aspect that we wish to investigate specifically is the application of a hierarchical approach to the above. Indeed, each game consists of sequences of <scene, action> 
pairs. This suggests the use of a multi-scale architecture, capable of working on multiple temporal scales (e.g. having an overarching plot interspersed with a number of minor
stories). For the same reason, the use of an episodic memory appears reasonable.

## Dataset
As a dataset, we want to collect adventures from http://ifdb.tads.org/. As we are going to build a generative model, we need a base corpus to train the model on. This
means we need adventures with at least one walkthrough available, so we can play them automatically. This would allow the supervised learning strategy outline previously.
To generate the training data, games will be automatically playes by the agent, generating a text output consisting of the <state, action> pairs. In order to convert the text to
a vector representation, the use of word-embeddings seems natural.

## Challenges
Quite naturally, a number of questions arise:

* how do we define reward exactly? Are different rewards issued at different time scales? Can we use a "player policy", i.e. the sequence of actions to
  beat the game, to help the generation process?
* how do we design the hierarchical architecture? Do we want a single, deep+tall network or a set of loosely interacting "controllers" (see also [4])?
* how do we represent the input to the model?
* how to generate the training data? 

For this last question, two main format exists for interactive fiction material. For one, there actually exists a dedicated learning library written in python (https://github.com/danielricks/autoplay). For the second format, a text-only interpreter
could provide the basis for the agent (https://github.com/realnc/frobtads).

## References
[1] https://www.intellimedia.ncsu.edu/wp-content/uploads/wang-aiide-2017.pdf
[2] https://www.researchgate.net/profile/Xiaodong_He2/publication/306093902_Deep_Reinforcement_Learning_with_a_Natural_Language_Action_Space/links/57c4656b08aee465796c1fa3.pdf
[3] http://www.eecs.qmul.ac.uk/~josh/documents/2017/Chourdakis%20Reiss%20-%20CC-NLG.pdf
[4] http://papers.nips.cc/paper/6233-hierarchical-deep-reinforcement-learning-integrating-temporal-abstraction-and-intrinsic-motivation.pdf
[5] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.7314&rep=rep1&type=pdf
[6] https://arxiv.org/pdf/1506.07285.pdf