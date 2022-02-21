# The Sandbox Environment for Generalizable Agent Research Code and Tutorials

**The Sandbox Environment for Generalizable Agent Research** (SEGAR) is a suite of research tools for doing *Interactive 
Representation Learning* (IRepL). IRepL is the study of the interplay between 
representation learning and interactions. As a study, two key outcomes of 
IReL are answers to *why* and *how* interactions improve representations. 
IReL is distinct from Reinforcement Learning (RL) and Representation 
Learning (RepL), though dependent on these studies, focusing on how 
advances in RL can help RepL and visa versa. 

SEGAR provides opportunities for studying IReL in the context of vision.
SEGAR comes with a physics simulator that models dynamics under forces such 
as charge, magnetism, collisions, etc.
As such, SEGAR is readily relatable to RepL and RL in robotics domains, 
though we anticipate lessons learned here will be useful in a number of 
domains.
SEGAR also provides a fully-composable MDP, making it easy to construct 
custom environments and tasks.
Finally, SEGAR has *generalization in mind*, such that the components of 
every environment (such as the charge, mass, or position of objects) can be 
distributional. 
This allows the experimenter to quantify a generalization task in terms of 
the distance between distributions.

![Putt Putt](../resources/readme-images/puttputt_example.gif)
![Billiards](../resources/readme-images/billiards_example.gif)
![Invisiball](../resources/readme-images/invisiball_example.gif)

Project code documentation can be found [here](https://animated-train-17411965.pages.github.io/).

## The simulator

The simulator is a core component of the SEGAR research suite.
It controls:

* The underlying factors of the environment: the objects and tiles along
with their affordances
* The dynamics, conditioned on those objects and tiles, e.g., magnetism via
Lorentz's law.
* The transition function of those factors.

As such, the simulator is *separated* from the semantics of the MDP, the
task, the reward, etc. Its job is to only to manage the underlying factors
and to simulate physics.

For a detailed tutorial see [Simulator Tutorial](https://github.com/microsoft/segar/tree/main/segar/sim).

## The MDP

The MDP is split into the following functional components:

* __Simulator object__(`segar.sim.core.Simulator`) controls the underlying
state space $\mathcal{S}$ and the transition function $\mathcal{P}$.
* __Observations objects__(`segar.mdps.observations.Observations`), make up
the observation function and corresponding space $(\mathcal{Z}, \mathcal{O})$.
* __Task objects__(`segar.mdps.tasks.Task`) control the initialization $s_0$
(`segar.mdps.initializations.Initialization`), the reward $\mathcal{R}$, and
the actions $\mathcal{A}$. The task also embodies the semantics of the MDP,
controlling what states correspond to reward, what observations the agent
sees, etc.
* __MDP objects__ put this all together and coordinate all of the components.

For a detailed tutorial see [MDP Tutorial](https://github.com/microsoft/segar/tree/main/segar/mdps).

## Configurations
Configurations provide convenience for  creating environments, as well as 
provide a way to sample and compare environments from distributions.

For a detailed tutorial see [Configuration Tutorial](https://github.com/microsoft/segar/tree/main/segar/configs).

## Built-in Tasks
Several built-in tasks are available in SEGAR: PuttPutt, Invisiball, and 
Billiards.
For a detailed tutorial see [Tasks Tutorial](https://github.com/microsoft/segar/tree/main/segar/tasks).

## Rendering
# Rendering Tutorial

SEGAR provides flexible, transparent, and customizable rendering capabilities 
for generating visual features. We did this because we believe that making 
the visual features such allows the researcher to make better conclusions w.
r.t. agents and models trained on pixels.

For a detailed tutorial see [Tasks Tutorial](https://github.com/microsoft/segar/tree/main/segar/rendering).

## Running RL agents
TODO
