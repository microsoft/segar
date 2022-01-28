# Overview
**Robo Putt-Putt** (RPP) is a suite of research tools for studying *Interactive 
Representation Learning* (IRepL). IRepL is the study of the interplay between an agent's 
representation learning in an environment and the agent's interactions with this environment. 
As a field of study, IRepL researches two key questions: *why* and *how* do interactions help representation learning (RepL)?
IRepL is distinct from but related to Reinforcement Learning (RL), focusing on how 
advances in RL can benefit representation learning and visa versa. 

Robo Putt-Putt provides a setting for studying IRepL in the context of visual observations.
It comes with a physics simulator that models dynamics under phenomena such 
as charge, magnetism, collisions, etc.
As such, RPP is relevant to researching RepL and RL for robotics , 
although we anticipate lessons learned with RPP to be useful in a number of 
domains.
RPP gives an experiment designer rich control over defining benchmark MDPs, making it easy to construct 
custom environments and tasks. Finally, RPP was *designed with generalization in mind*, so the generative factors of 
RPP environments (such as the charge, mass, or position of objects) can be 
distributional. 
This allows an experimenter to set and quantify generalization difficulty in terms of 
a distance between training and test task distributions.

![Putt Putt](rpp/assets/examples/puttputt_example.gif)
![Billiards](rpp/assets/examples/billiards_example.gif)
![Invisiball](rpp/assets/examples/invisiball_example.gif)

Details on code and tutorials can be found [here](https://github.com/microsoft/roboputtputt/tree/main/rpp).

## Background and Motivation
Throughout most of ML history, progress in areas that deal 
with high-dimensional weakly structured inputs -- chiefly computer vision 
(CV) and natural language processing (NLP) -- has relied on feature 
engineering. 
A major role of constructing low-dimensional features has been in 
*generalizing* knowledge across tasks, thereby enabling successful 
use of CV and NLP technology even in settings with scarce task-specific 
data. 
Deep learning has revolutionized CV and NLP by supplanting feature 
engineering with its more scalable and data-driven version, 
*representation learning*. 


Encouraged by the impact of representation learning on CV, researchers have 
been attempting to replicate it in two related areas -- robotics and RL with visual observations. Despite some successes, the effect of 
representation learning on these fields has been more modest. 
In our opinion, this status quo is for the following reasons:

- Successful decision-making in RL and robotics often relies on knowing 
  latent features such as object mass as fragility. Such **statically 
  unobservable features** can't be detected from a single image, because 
  they are only weakly reflected in object appearance, or not all.
  
- Many representation learning techniques, and even representations such as 
  pretrained ResNet, that have been tried in RL and robotics were 
  transplanted without modification from computer vision literature and 
  operated on individual static images. As a result, these approaches fail 
  to extract crucial statically unobservable features.
  
- **Interactive representation learning** (IRepL) exploits 
  *multimodality* in robotics and RL data by extracting features from 
  sequences of observation images and actions and, in principle, can derive 
  statically unobservable features. However, despite a lot of recent IRepL literature, IRepL research is still in its infancy. Connections 
  between changes in statically unobservable features and representation 
  generalization are poorly understood as well.
  
- Progress towards such an understanding has been checked by the lack of 
  benchmarks where researchers can vary the values of crucial unobservable 
  features across tasks in a controlled fashion, measure the effect of 
  these changes on generalization, and play with the degree of correlation 
  between observable and unobservable features.  
  
Thus, a key step towards making progress towards representation learning in 
sequential decision-making scenarios is creating a benchmarks that would 
enable controlled, informative experiments in this research ares. 
**RoboPuttPutt**, the environment suite we present here, serves exactly 
that purpose.

## What does RPP offer that isn't already offered in benchmark *X*?

There are a number of interesting and useful benchmarks available for 
training representations through interaction. RPP's key distinguishing feature -- which we 
believe to be crucial for making substantial progress in IRepL -- is **giving researchers full 
control over constructing the training and test task distributions for answering specific 
representation learning questions the researchers are interested in**. Namely:

* __RPP allows an experiment designer to inspect and modify all components of an environment MDP.__ This includes:

    * __The evironment MDP's generative factors.__ For environments in popular existing benchmarks, e.g., 
      [Procgen](https://github.com/openai/procgen) and [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment), 
      their generative factors such as the number of objects and their properties aren't easily available for modification or even inspection.
      This makes it difficult to understand what information the agent should be able to capture in a representation.

    * __MDP dynamics.__ The evironment MDP's dynamics is parameterized by its generative factors. In RPP, users can easily inspect the
      dynamics model to understand how it is affected by changes in the generative factors and choose these factors' values to induce
      dynamics with a desired behavior. This makes RPP useful in studying causal inference.

    * __Reward function.__ Reward functions defines tasks in an environment, and RPP allows an experiment designer to easily construct them. This makes RPP highly customizable, 
      allowing researchers to construct tasks that depend on subsets of factors of the environment, and contrasts with
      most existing benchmarks used in RepL literature, which make defining new tasks non-trivial if at all possible.

    * __Visual features.__  We believe that providing observations to the 
      agent based on visual features that are understandable by humans may 
      unintentionally introduce experimental challenges, as neural networks 
      can leverage low-level visual cues to "cheat" at tasks that were 
      intended to require higher-level reasoning. In RPP, the visual 
      features are treated as a transparent and controllable variable in 
      building experiments, such that the research can control how the 
      underlying factors are expresses in terms of pixels, if at all.

* __RPP allows defining training and test task distributions for generalization experiments__. Such experiments are a major motivation behind RPP's design,
  and RPP makes them easy to set up by specifying distributions over generative factors' values for training and test tasks. In popular benchmarks such as
  Procgen and [MetaWorld](https://github.com/rlworkgroup/metaworld), these distributions are usually predefined and fixed, so an experiment designer has no control
  over the generalization gap that a learned representation is expected to bridge. 

* __RPP is lightweight__. Some physics-based benchmark suites, e.g., [dm_control](https://github.com/deepmind/dm_control), also provide a significant degree of control
  over environment design. However, 3D physics simulation makes them computationally expensive. RPP is based on the insight that most questions
  about representation learning from images are applicable just as well in a set of lightweight 2D environments that can be tested on quickly to make progress in IRepL and related areas.



## Research Questions and Topics


### When and how does interactivity help in representation learning?

- What properties of an environment and a task in it determine whether we need to
  infer statically unobservable state features (and hence use IRepL)? 

- What properties of a task distribution necessitate learning *all* statically unobservable features in the MDP's generative model, such as masses and charges of all objects?

- Does *interactive* RepL provide an edge to models ultimately meant for *static* tasks such as 
  classification and segmentation, versus traditional RepL methods that use static i.
  i.d. data?
  
- How do representations trained on static data differ from those learned 
  from interactive data?
  


### How can IRepL help in RL?

- Can we develop better algorithms for learning statically unobservable 
  features relevant to sequential decision-making?
  
- How can we most effectively learn the part of the underlying causal graph of the MDP, including statically unobservable variables, relevant to learning (near-)optimal policies?

- How can we exploit RepL to learn better policies, better interventions, and 
  better exploration policies for agent learning?

- What are the roles of priors in unsupervised representation learning when 
  data is drawn from interactions that depend on said representations?

- How can IRepL make the most use of *offline* interaction data and thereby help offline RL algorithms?


### How can IRepL help with generalization?

We informally say that a representation generalizes well from a training to a test task distribution if it allows learning (or transferring, in the case of zero-shot generalization) a near-optimal test distribution policy that is linear in the representation's features. Some of the relevant questions that RPP facilitates studying are: 

- How do we learn representations that generalize across specific kinds 
  of differences between training and test task distributions?

- What assumptions do we need to make on the collection of online interactive data 
  and coverage of offline interactive data in order to enable generalization 
  across various differences between training and test task distributions?

- What properties of training and test task distribution determine whether we need to
  infer statically unobservable state features (and hence use IRepL)? E.g., inferring 
  the charge of colliding metal balls may or may not be important for a generalizable representation for predicting their future positions,
  depending on how small their masses can be in the test task distribution.

  

# Installation
- Create and activate a conda virtual environment:
  ```
  conda create -n rpp pip python=3.6
  conda activate rpp
  ```
- Clone the current repo and install it in editable mode:
  ```
  git clone https://github.com/microsoft/roboputtputt.git
  cd roboputtputt
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

For tutorials on the features of RPP, including running experiments, 
creating environments, etc, please see `rpp/README.md`.


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
