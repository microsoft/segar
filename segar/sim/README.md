This tutorial covers using the simulator in RPP.


```python
import pprint

import numpy as np

from rpp.sim import Simulator


pp = pprint.PrettyPrinter(indent=4)
```

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

Creating a simulator is the first step in using RPP:


```python
sim = Simulator()
```

## Adding objects

The simulator by itself isn't interesting: we need to add objects and tiles
to it.

Let's add a single ball to the middle of the field.


```python
pp.pprint(sim.things)
sim.add_ball(position=np.array([0., 0.]))
pp.pprint(sim.things)
```

    {'global_friction': Entity(id=ID(global_friction))}
    {0: Object(id=ID(0)), 'global_friction': Entity(id=ID(global_friction))}


Now we have a single object in the middle of the field. Note that by
default the simulator has a square field with boundaries at (-1., 1.). Let's
look at the state of the object:


```python
pp.pprint(sim.thing_states[0])
```

    {   StoredEnergy: StoredEnergy(0.0),
        Position: Position([0. 0.]),
        Velocity: Velocity([0. 0.]),
        Acceleration: Acceleration([0. 0.]),
        Visible: Visible(True),
        Alive: Alive(True),
        Done: Done(False),
        InfiniteEnergy: InfiniteEnergy(False),
        Mobile: Mobile(True),
        Order: Order(0),
        Size: Size(0.2),
        Mass: Mass(1.0),
        Density: Density(1.0),
        Charge: Charge(0.0),
        Magnetism: Magnetism(0.0),
        Collides: Collides(None),
        ID: ID(0),
        Label: Label(object),
        Text: Text(B),
        Shape: Shape(Circle(r=0.1)),
        'cl': 'Object'}


The state contains all attributes necessary to intantiate an object, but a
number of these will be useful for building MDPs (see the `rpp/mdps/README.md`
for more details).

It's good to be able to see the state of objects, but visualization can be
very useful to see what's going on. For this, we borrow a renderer from the
rendering module (see `rpp/rendering/README.md` for details).


```python
from rpp.rendering.rgb_rendering import RGBRenderer
import PIL
from matplotlib.pyplot import imshow

renderer = RGBRenderer(annotation=True)
renderer.reset(sim)
img_arr = renderer()

image = PIL.Image.fromarray(img_arr)
imshow(image)
```




    <matplotlib.image.AxesImage at 0x7ff09cfc92e0>




    
![png](../../resources/readme-images/segar/sim/README_9_1.png)
    


## Adding more objects and tiles

That's a very simple arena, but we expected that. Let's add some more things.


```python
from rpp.factors import Hexagon, RandomConvexHull, Mobile, Size, Charge, Shape

sim.add_magnet(position=np.array([-0.5, 0.5]), text='M', initial_factors={Mobile: True, Size:0.3})
sim.add_sand(position=np.array([0.1, -0.1]), text='S', initial_factors={Shape: Hexagon(0.4)})
sim.add_magma(position=np.array([-0.5, 0.5]), text='G', initial_factors={Shape: RandomConvexHull(0.3)})
sim.add_charger(position=np.array([0.3, 0.7]), text='C', initial_factors={Charge:1.0, Mobile: True})

pp.pprint(sim.state)
renderer.reset(sim)
img_arr = renderer()

image = PIL.Image.fromarray(img_arr)
imshow(image)
```

    {   'boundaries': (-1, 1),
        'parameters': {   Friction: Friction(0.05),
                          MinMass: MinMass(0.1),
                          MaxVelocity: MaxVelocity(10.0),
                          MinVelocity: MinVelocity(0.0001),
                          Gravity: Gravity(10.0),
                          WallDamping: WallDamping(0.025),
                          <class 'rpp.types.Time'>: 0.01},
        'safe_mode': False,
        'save_path': None,
        'scaling_factor': 2.8284271247461903,
        'things': {   0: {   StoredEnergy: StoredEnergy(0.0),
                             Position: Position([0. 0.]),
                             Velocity: Velocity([0. 0.]),
                             Acceleration: Acceleration([0. 0.]),
                             Visible: Visible(True),
                             Alive: Alive(True),
                             Done: Done(False),
                             InfiniteEnergy: InfiniteEnergy(False),
                             Mobile: Mobile(True),
                             Order: Order(0),
                             Size: Size(0.2),
                             Mass: Mass(1.0),
                             Density: Density(1.0),
                             Charge: Charge(0.0),
                             Magnetism: Magnetism(0.0),
                             Collides: Collides(None),
                             ID: ID(0),
                             Label: Label(object),
                             Text: Text(B),
                             Shape: Shape(Circle(r=0.1)),
                             'cl': 'Object'},
                      1: {   StoredEnergy: StoredEnergy(0.0),
                             Position: Position([-0.5  0.5]),
                             Velocity: Velocity([0. 0.]),
                             Acceleration: Acceleration([0. 0.]),
                             Visible: Visible(True),
                             Alive: Alive(True),
                             Done: Done(False),
                             InfiniteEnergy: InfiniteEnergy(False),
                             Mobile: Mobile(True),
                             Order: Order(0),
                             Size: Size(0.3),
                             Mass: Mass(1.0),
                             Density: Density(1.0),
                             Charge: Charge(0.0),
                             Magnetism: Magnetism(1.0),
                             Collides: Collides(None),
                             ID: ID(1),
                             Label: Label(magnet),
                             Text: Text(M),
                             Shape: Shape(Circle(r=0.15)),
                             'cl': 'Magnet'},
                      2: {   Position: Position([ 0.1 -0.1]),
                             Visible: Visible(True),
                             Order: Order(0),
                             Size: Size(0.4),
                             Heat: Heat(0.0),
                             Friction: Friction(0.4),
                             Floor: Floor(None),
                             ID: ID(2),
                             Label: Label(sand),
                             Text: Text(S),
                             Shape: Shape(Hexagon(p=[[-0.2         0.        ]
     [-0.1        -0.17320508]
     [ 0.1        -0.17320508]
     [ 0.2         0.        ]
     [ 0.1         0.17320508]
     [-0.1         0.17320508]])),
                             'cl': 'SandTile'},
                      3: {   Position: Position([-0.5  0.5]),
                             Visible: Visible(True),
                             Order: Order(0),
                             Size: Size(0.3),
                             Heat: Heat(1.0),
                             Friction: Friction(0.1),
                             Floor: Floor(None),
                             ID: ID(3),
                             Label: Label(magma),
                             Text: Text(G),
                             Shape: Shape(RandomConvexHull(p=[[ 0.17190495 -0.31753388]
     [ 0.3567939  -0.11447178]
     [ 0.27574046  0.18417291]
     [ 0.15424706  0.29786891]
     [ 0.08204527  0.32354599]
     [-0.12679405  0.20864738]
     [-0.20936721  0.1404314 ]
     [-0.23014032  0.11565415]
     [-0.26089685  0.04421944]
     [-0.28800921 -0.2072203 ]
     [-0.01132443 -0.31665488]])),
                             'cl': 'MagmaTile'},
                      4: {   StoredEnergy: StoredEnergy(0.0),
                             Position: Position([0.3 0.7]),
                             Velocity: Velocity([0. 0.]),
                             Acceleration: Acceleration([0. 0.]),
                             Visible: Visible(True),
                             Alive: Alive(True),
                             Done: Done(False),
                             InfiniteEnergy: InfiniteEnergy(False),
                             Mobile: Mobile(True),
                             Order: Order(0),
                             Size: Size(0.2),
                             Mass: Mass(1.0),
                             Density: Density(1.0),
                             Charge: Charge(1.0),
                             Magnetism: Magnetism(0.0),
                             Collides: Collides(None),
                             ID: ID(4),
                             Label: Label(charger),
                             Text: Text(C),
                             Shape: Shape(Circle(r=0.1)),
                             'cl': 'Charger'},
                      'global_friction': {   Friction: Friction(0.05),
                                             ID: ID(global_friction),
                                             'cl': 'Entity'}},
        'timer': 0,
        'walls': {0: {'boundaries': (-1, 1), 'damping': WallDamping(0.025)}}}





    <matplotlib.image.AxesImage at 0x7ff07808b1f0>




    
![png](../../resources/readme-images/segar/sim/README_11_2.png)
    


Note: we must reset the renderer every time we add new objects or tiles..

Let's add one more object, but misplace it on top of another. The simulator
has convenient methods for fixing overlaps.


```python
sim.add_ball(position=np.array([0.05, 0.05]), text='X', initial_factors={Charge: -0.5})
renderer.reset(sim)

img_arr = renderer()
image = PIL.Image.fromarray(img_arr)
imshow(image)
```




    <matplotlib.image.AxesImage at 0x7ff0687e18e0>




    
![png](../../resources/readme-images/segar/sim/README_13_1.png)
    



```python
sim.fix_overlaps()

img_arr = renderer()
image = PIL.Image.fromarray(img_arr)
imshow(image)
```




    <matplotlib.image.AxesImage at 0x7ff09d163d60>




    
![png](../../resources/readme-images/segar/sim/README_14_1.png)
    


## Moving the simulator forward

Now that we have some objects and tiles, let's get things moving. The
simulator method `step` will move the simulator forward. Let's apply some
force to one of the objects.


```python
pp.pprint(sim.things)
sim.add_force(5, np.array([0.3, 0.7]))

for _ in range(9):
    sim.step()

img_arr = renderer()
image = PIL.Image.fromarray(img_arr)
imshow(image)
```

    {   0: Object(id=ID(0)),
        1: Magnet(id=ID(1)),
        2: SandTile(id=ID(2)),
        3: MagmaTile(id=ID(3)),
        4: Charger(id=ID(4)),
        5: Object(id=ID(5)),
        'global_friction': Entity(id=ID(global_friction))}





    <matplotlib.image.AxesImage at 0x7ff078296f10>




    
![png](../../resources/readme-images/segar/sim/README_16_2.png)
    


Looks like things have moved. We can even go back if we want.
Note: we have to reset the renderer because it references old object copies
kept in the history.


```python
for _ in range(9):
    sim.rewind()

renderer.reset(sim)

img_arr = renderer()
image = PIL.Image.fromarray(img_arr)
imshow(image)
```




    <matplotlib.image.AxesImage at 0x7ff09d026040>




    
![png](../../resources/readme-images/segar/sim/README_18_1.png)
    


Now let's get things really moving. There are some convenient functions in
the tools module for generating and viewing trajectories.

```python
from IPython.display import Image

from rpp.tools.sample_trajectories import rollout_sim_only, save_gif

renderer.reset(sim)
trajectories, imgs = rollout_sim_only(sim, renderer=renderer)

save_gif(imgs, out_path='../../resources/readme-images/segar/sim/trajectory_sample.gif')
Image(open('../../resources/readme-images/segar/sim/trajectory_sample.gif', 'rb').read())
```




    
![png](../../resources/readme-images/segar/sim/README_20_0.png)
    



Finally, the sim has rules. You can view the rules by checking the `rules` method:


```python
pp.pprint(sim.rules)
```

    [   rpp.rules.transitions.Differential[rpp.factors.arrays.Position] <- move([Position, Velocity, MinVelocity]),
        rpp.rules.transitions.Aggregate[rpp.factors.arrays.Acceleration] <- lorentz_law([(Position, Velocity, Charge, Magnetism), (Position, Velocity, Charge, Mass, Acceleration)]),
        rpp.rules.transitions.Aggregate[rpp.factors.arrays.Acceleration] <- apply_friction([(Mass, Velocity, Acceleration), (Friction,), Gravity]),
        rpp.rules.transitions.SetFactor[rpp.factors.number_factors.Mass] <- apply_burn([(Mass, Mobile), (Heat,)]),
        typing.Tuple[rpp.rules.transitions.SetFactor[rpp.factors.arrays.Velocity], rpp.rules.transitions.SetFactor[rpp.factors.arrays.Acceleration]] <- stop_condition([(Mobile, Alive, Velocity, Acceleration)]),
        typing.Tuple[rpp.rules.transitions.SetFactor[rpp.factors.number_factors.Mass], rpp.rules.transitions.SetFactor[rpp.factors.arrays.Velocity], rpp.rules.transitions.SetFactor[rpp.factors.bools.Alive], rpp.rules.transitions.SetFactor[rpp.factors.arrays.Acceleration], rpp.rules.transitions.SetFactor[rpp.factors.bools.Visible]] <- kill_condition([Mass, Velocity, Visible, Acceleration, Alive, MinMass]),
        typing.Tuple[rpp.rules.transitions.SetFactor[rpp.factors.bools.Done], rpp.rules.transitions.SetFactor[rpp.factors.bools.Visible], rpp.rules.transitions.SetFactor[rpp.factors.bools.Mobile]] <- consume([(Mobile, Done, Visible), (Consumes,)]),
        rpp.rules.transitions.Differential[rpp.factors.arrays.Velocity] <- accelerate([Velocity, Acceleration])]


The rules can be added and removed from the sim. Let's remove Lorentz law, which controls how objects move according to charge and magnetism:

```python
sim.remove_rule('lorentz_law')
trajectories, imgs = rollout_sim_only(sim, renderer=renderer)

save_gif(imgs, out_path='../../resources/readme-images/segar/sim/trajectory_sample2.gif')
Image(open('../../resources/readme-images/segar/sim/trajectory_sample2.gif', 'rb').read())
```




    
![png](../../resources/readme-images/segar/sim/README_24_0.png)
    



And there you go. The simulator and related object, tile, and shape API is
simple and easily extensible, so they should provide a clear way to build
easy-to-interpret experiments for Interactive Representation Learning.

The next suggested README is in `mdps`.
