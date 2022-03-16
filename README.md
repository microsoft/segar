[![Flake8](https://github.com/microsoft/segar/actions/workflows/flake8.yml/badge.svg?branch=main)](https://github.com/microsoft/segar/actions/workflows/flake8.yml)
[![CodeQL](https://github.com/microsoft/segar/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/microsoft/segar/actions/workflows/codeql-analysis.yml)
[![.github/workflows/sphinx.yml](https://github.com/microsoft/segar/actions/workflows/sphinx.yml/badge.svg)](https://github.com/microsoft/segar/actions/workflows/sphinx.yml)
[![Test environments](https://github.com/microsoft/segar/actions/workflows/test_environments.yml/badge.svg)](https://github.com/microsoft/segar/actions/workflows/test_environments.yml)

# Overview
**The Sandbox Environment for Generalizable Agent Research** (SEGAR) is a suite of research 
tools for studying generalization in settings that involve Learning from Interactive Data (LfID). 
LfID broadly is any settings where an agent is trained on interactive data to perform 
optimally on a sequential decision-making task, including offline / online RL, IL, Goal-Conditioned 
RL, Meta-RL, etc, and we believe SEGAR will be useful in any of these settings.
SEGAR comes with the following features:
- A controllable physics simulator that can be used to model dynamics under a 
flexible set of physical phenomena the designer chooses, such as those involving charge, magnetism, 
collisions, etc.
- The experiment designer has rich control over defining benchmark MDPs, making it easy to
construct custom environments and tasks. 
- Providing transparent and intuitive control over variation in the environment, implemented in 
  a way to allow the researcher to precisely specify the settings that an agent is trained or tested in.
- Control is in terms of distributions of parameters that determine that task, which also makes it possible to quantitatively measure between tasks, sets of tasks, and 
their distributions.
- Adds much needed accountability to LfID generalization research, as these measures can be 
used to directly gauge the nature and difficulty of the generalization tasks.

![Putt Putt](resources/readme-images/puttputt_example.gif)
![Billiards](resources/readme-images/billiards_example.gif)
![Invisiball](resources/readme-images/invisiball_example.gif)

Details on code and tutorials can be found 
[here](https://github.com/microsoft/segar/tree/main/segar).

## Background and Motivation
In settings that involve training on i.i.d. data, such as classification, regression, generation 
tasks, etc, regardless of the domain of the problem, generalization is a central criteria to an 
algorithm's success.
Success of resulting models hinges on good performance across variation in the data, whether 
representing in-distribution or out-of-distribution tasks.

i.i.d. supervised learning settings rely heavily on annotation for formulation of experiments to 
create splits that are representative of some task the designer has in mind.
Beyond the common criteria of generalizing from a limited number of training examples, 
annotation can challenge a model's ability to generalize from a few examples (e.g., few-shot 
learning), from no examples (e.g., zero-shot learning), etc.

For interactive settings like RL or IL, annotation is far less available nor used.
Benchmarks such as Procgen, Robosuite, CausalWorld, etc all carry different traits that make it 
substantially easier to evaluate whether algorithms generate generalizable agents or not than 
the most popular benchmark of all, Atari.
However, we believe that existing benchmarks are missing one or more of the existing 
critical or useful features:

- Access to and control over all underlying factors of the environment, their distributions, and 
their samples.
- Access to and control over all transition functions or rules of the environment.
- An interface for creating interactive tasks that is intuitive, extensible, and customizable.
- A simulator that is inexpensive to run on accessible hardware. 
- A framework for measuring experiments, such as on distributions between sets of tasks (e.g., 
  across train/test sets).
  
We developed the SEGAR to bring all of these components into one place.
SEGAR is designed to be customizable and community-driven, allowing for measurable 
extensibility on generalization experimentation.
While SEGAR is customizable, it comes with a built-in "physics", such as those involving charge,
magnetism, collisions, etc., as well as built-in tasks for studying generalization and measures 
on those tasks which we believe to be a useful start for evaluating generalization experiments 
in LfID.
It provides a simple API for creating and evaluating experiments for research towards 
generalizable agents along numerous axes, from the structure of the state space, the rules by 
which states change, what and how the agent sees (e.g., visual features for pixel-based 
observations, partial observability, etc), and the overall task structure.
Finally, as we believe that studying the representation of interactive agents will be crucial 
towards developing better algorithms that yield generalizable agents, we provide built-in 
measures on the agent's representation as well based on the mutual information between its 
representations and the underlying factors.
Note that, while we provide these built-in tasks and measures, SEGAR is designed from the 
ground up to be customizable and extensible in nearly any way, allowing the research community 
to design new tasks with novel components and measures.

## What does SEGAR offer that isn't already offered in benchmark *X*?

There are a number of interesting and useful benchmarks available for 
training representations through interaction. SEGAR's key distinguishing feature -- which we 
believe to be crucial for making substantial progress in generalization in LfID -- is **giving 
researchers full  control over constructing the training and test task distributions for 
answering specific representation learning questions the researchers are interested in**. Namely:

* __SEGAR allows an experiment designer to inspect and modify all components of an environment MDP.__ This includes:

    * __The evironment MDP's generative factors.__ For environments in popular existing benchmarks, e.g., 
      [Procgen](https://github.com/openai/procgen) and [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment), 
      their generative factors such as the number of objects and their properties aren't easily available for modification or even inspection.
      This makes it difficult to understand what information the agent should be able to capture in a representation.

    * __MDP dynamics.__ The evironment MDP's dynamics is parameterized by its generative factors. In SEGAR, users can easily inspect the
      dynamics model to understand how it is affected by changes in the generative factors and choose these factors' values to induce
      dynamics with a desired behavior. This makes SEGAR useful in studying causal inference.

    * __Reward function.__ Reward functions defines tasks in an environment, and SEGAR allows an experiment designer to easily construct them. This makes SEGAR highly customizable, 
      allowing researchers to construct tasks that depend on subsets of factors of the environment, and contrasts with
      most existing benchmarks used in RepL literature, which make defining new tasks non-trivial if at all possible.

    * __Visual features.__  We believe that providing observations to the 
      agent based on visual features that are understandable by humans may 
      unintentionally introduce experimental challenges, as neural networks 
      can leverage low-level visual cues to "cheat" at tasks that were 
      intended to require higher-level reasoning. In SEGAR, the visual 
      features are treated as a transparent and controllable variable in 
      building experiments, such that the researcher can control how the 
      underlying factors are expressed in terms of pixels, if at all.

* __SEGAR allows defining training and test task distributions for generalization experiments__. Such experiments are a major motivation behind SEGAR's design,
  and SEGAR makes them easy to set up by specifying distributions over generative factors' values for training and test tasks. In popular benchmarks such as
  Procgen and [MetaWorld](https://github.com/rlworkgroup/metaworld), these distributions are usually predefined and fixed, so an experiment designer has no control
  over the generalization gap that a learned representation is expected to bridge. 

* __SEGAR is lightweight__. Some physics-based benchmark suites, e.g., [dm_control](https://github.com/deepmind/dm_control), also provide a significant degree of control
  over environment design. However, 3D physics simulation makes them computationally expensive. SEGAR is based on the insight that most questions
  about representation learning from images are applicable just as well in a set of lightweight 
  2D environments that can be tested on quickly to make progress in generalization and related 
  areas.



## Research Questions and Topics


### How can we train generalizable agents?

We informally say that a representation generalizes well from a training to a test task 
distribution if it allows learning (or transferring, in the case of zero-shot generalization) a 
near-optimal test distribution policy that is linear in the representation's features. Some of 
the relevant questions that SEGAR facilitates studying are: 

- How do we learn representations that generalize across specific kinds 
  of differences between training and test task distributions?

- What assumptions do we need to make on the collection of online interactive data 
  and coverage of offline interactive data in order to enable generalization 
  across various differences between training and test task distributions?

- What properties of training and test task distribution determine whether we need to
  infer statically unobservable state features? E.g., inferring the charge of colliding metal 
  balls may or may not be important for a generalizable representation for predicting their 
  future positions, depending on how small their masses can be in the test task distribution.


### When and how does interactivity help in representation learning and visa-versa?

- What properties of an environment and a task in it determine whether we need to
  infer statically unobservable state features? 

- What properties of a task distribution necessitate learning *all* statically unobservable 
  features in the MDP's generative model, such as masses and charges of all objects?

- Does *interactive* RepL provide an edge to models ultimately meant for *static* tasks such as 
  classification and segmentation, versus traditional RepL methods that use static i.
  i.d. data?
  
- How do representations trained on static data differ from those learned 
  from interactive data?

- Can we develop better algorithms for learning statically unobservable 
  features relevant to sequential decision-making?
  
- How can we most effectively learn the part of the underlying causal graph of the MDP,
  including statically unobservable variables, relevant to learning (near-)optimal policies?

- How can we exploit RepL to learn better policies, better interventions, and 
  better exploration policies for agent learning?

- What are the roles of priors in unsupervised representation learning when 
  data is drawn from interactions that depend on said representations?



  

# Installation
- Create and activate a conda virtual environment:
  ```
  conda create -n segar pip python=3.9
  conda activate segar
  ```

- Install from PyPI for the public release:
  ```
  pip install segar
  ```
- Or clone the current repo and install it for the latest release:
  ```
  git clone https://github.com/microsoft/segar.git
  cd segar
  pip install -e .
  ```

- For running RL samples (using rllib):
  ```
  pip install -e '.[rl]'
  ```


- For pytorch installation:
  - On CPU-only machines:
  ```
  conda install pytorch=1.7.0 torchvision cpuonly -c pytorch
  ```
  
  - On machines with a GPU:
  ```
  conda install pytorch=1.7.0 torchvision cudatoolkit=10.1 -c pytorch
  ```

For tutorials on the features of SEGAR, including running experiments, 
creating environments, etc, please see `segar/README.md`.


## Gym Environments Included

Segar is designed to be extensible and for users to design their own environments. 
To help users get started, it includes a handful of OpenAI-Gym compatible environments for training RL agents 
to play puttputt/minigolf.

The agent controls the (player-)ball by applying a 2D continuous force vector. The goal is to navigate the ball to the
goal through obstacles. 

- **Action space:** `(2,), np.float32, [-1,1]` - 2D force vector to apply to the ball
- **Observation space:** `(64,64,3), np.uint8, [0,255]` - top-down color image.
- **Reward:** Current distance between ball and goal.

The environments follow the format:
    
    Segar-TASK-DIFFICULTY-OBSERVATION-v0

...where we have

- `TASK`: one of `empty`, `objectsxN`, or `tilesxN` and `N` is one of `1`, `2`, `3`
  - `TASK`:`empty` is an empty square arena with one goal and one ball.
  - `TASK`:`objectsxN` is a square arena with a goal, a ball, and `N` other random objects (like other non-player balls 
or magnets)
  - `TASK`:`tilesxN` is a square arena with a goal, a ball, and `N` random patches of random materials (like sand or 
lava)
- `DIFFICULTY`: one of `easy`, `medium`, or `hard`. Generally, in the easy case, the ball and goal are in fixed
positions, in the medium case, the ball starts at the bottom of the arena and the goal is always at the top, and in 
the hard case, the ball and goal can be anywhere, uniformly.
- `OBSERVATION`: currently only `rgb`. We will include different modalities later. This corresponds to the observation
space `64x64x3` in `np.uint8` (i.e. a color image of 64x64 pixels)

So for example, we have `Segar-empty-easy-rgb-v0` or `Segar-tilesx3-hard-rgb-v0`. [The complete list of current 
environments](./segar/__init__.py) can be found at the bottom of the file [`segar/__init__.py`](./segar/__init__.py). 
All environments are registered when you call `import segar`. 


More details about the [specs of each environment](./segar/envs/env.py) can be found in the env.py file: 
[`segar/envs/env.py`](./segar/envs/env.py)

## Contributing

This project welcomes contributions and suggestions.  Most contributions 
require you to agree to a Contributor License Agreement (CLA) declaring 
that you have the right to, and actually do, grant us the rights to use 
your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine 
whether you need to provide a CLA and decorate the PR appropriately (e.g., 
status check, comment). Simply follow the instructions provided by the bot. 
You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct]
(https://opensource.microsoft.com/codeofconduct/). For more information see 
the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) 
or contact [opencode@microsoft.com] (mailto:opencode@microsoft.com) with 
any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or 
services. Authorized use of Microsoft trademarks or logos is subject to and 
must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project 
must not cause confusion or imply Microsoft sponsorship. Any use of 
third-party trademarks or logos are subject to those third-party's policies.
