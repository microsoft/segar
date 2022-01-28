# Robo PuttPutt Code and Tutorials

**Robo Putt-Putt** (RPP) is a suite of research tools for doing *Interactive 
Representation Learning* (IReL). IReL is the study of the interplay between 
representation learning and interactions. As a study, two key outcomes of 
IReL are answers to *why* and *how* interactions improve representations. 
IReL is distinct from Reinforcement Learning (RL) and Representation 
Learning (RepL), though dependent on these studies, focusing on how 
advances in RL can help RepL and visa versa. 

RPP provides opportunities for studying IReL in the context of vision.
RPP comes with a physics simulator that models dynamics under forces such 
as charge, magnetism, collisions, etc.
As such, RPP is readily relatable to RepL and RL in robotics domains, 
though we anticipate lessons learned here will be useful in a number of 
domains.
RPP also provides a fully-composable MDP, making it easy to construct 
custom environments and tasks.
Finally, RPP has *generalization in mind*, such that the components of 
every environment (such as the charge, mass, or position of objects) can be 
distributional. 
This allows the experimenter to quantify a generalization task in terms of 
the distance between distributions.

![Putt Putt](assets/examples/puttputt_example.gif)
![Billiards](assets/examples/billiards_example.gif)
![Invisiball](assets/examples/invisiball_example.gif)

Project code documentation can be found [here](https://animated-train-17411965.pages.github.io/).

## The simulator

The simulator is a core component of the RPP research suite.
It controls:
* The underlying factors of the environment: the objects and tiles along
with their affordances
* The dynamics, conditioned on those objects and tiles, e.g., magnetism via
Lorentz's law.
* The transition function of those factors.

As such, the simulator is *separated* from the semantics of the MDP, the
task, the reward, etc. Its job is to only to manage the underlying factors
and to simulate physics.

For a detailed tutorial see [Simulator Tutorial](https://github.com/microsoft/roboputtputt/tree/main/rpp/sim).

## The MDP

The MDP is split into the following functional components:

* __Simulator object__(`rpp.sim.core.Simulator`) controls the underlying
state space $\mathcal{S}$ and the transition function $\mathcal{P}$.
* __Observations objects__(`rpp.mdps.observations.Observations`), make up
the observation function and corresponding space $(\mathcal{Z}, \mathcal{O})$.
* __Task objects__(`rpp.mdps.tasks.Task`) control the initialization $s_0$
(`rpp.mdps.initializations.Initialization`), the reward $\mathcal{R}$, and
the actions $\mathcal{A}$. The task also embodies the semantics of the MDP,
controlling what states correspond to reward, what observations the agent
sees, etc.
* __MDP objects__ put this all together and coordinate all of the components.

For a detailed tutorial see [MDP Tutorial](https://github.com/microsoft/roboputtputt/tree/main/rpp/mdps).

## Configurations
Configurations provide convenience for  creating environments, as well as 
provide a way to sample and compare environments from distributions.

For a detailed tutorial see [Configuration Tutorial](https://github.com/microsoft/roboputtputt/tree/main/rpp/config).

## Built-in Tasks
Several built-in tasks are available in RPP: PuttPutt, Invisiball, and 
Billiards.
For a detailed tutorial see [Tasks Tutorial](https://github.com/microsoft/roboputtputt/tree/main/rpp/tasks).

## Rendering
# Rendering Tutorial

RPP provides flexible, transparent, and customizable rendering capabilities 
for generating visual features. We did this because we believe that making 
the visual features such allows the researcher to make better conclusions w.
r.t. agents and models trained on pixels.

For a detailed tutorial see [Tasks Tutorial](https://github.com/microsoft/roboputtputt/tree/main/rpp/rendering).

## Running RL agents
TODO
