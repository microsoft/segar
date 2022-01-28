# Rules and relations

RPP provides a framework on the simulator rules, which determine how states transition from one to another over subsequent time steps. RPP comes with a set of built-in rules for the simulator, but these can be changed at the user's disgression.

### Rules
Rules are functions from one set of factors to another. Consider the following built-in rules:


```python
from rpp.rules import lorentz_law, move, apply_friction
print(lorentz_law)
print(move)
print(apply_friction)
```

    rpp.rules.transitions.Aggregate[rpp.factors.arrays.Acceleration] <- lorentz_law([(Position, Velocity, Charge, Magnetism), (Position, Velocity, Charge, Mass, Acceleration)])
    rpp.rules.transitions.Differential[rpp.factors.arrays.Position] <- move([Position, Velocity, MinVelocity])
    rpp.rules.transitions.Aggregate[rpp.factors.arrays.Acceleration] <- apply_friction([(Mass, Velocity, Acceleration), (Friction,), Gravity])


These rules all apply on sets of (sets of) factors (see the factors README for more details). If the rule contains a single set of factors, this applies to a single Entity (see the things README for more details). If a rule contains multiple sets, then the rule applies to multiple things.

Rules apply automatic pattern matching, such that a rule will not return a valid change in factor unless the input pattern matches the factors contained in the inputs and the parameters:


```python
from rpp.factors import Position, Charge, Mass, Velocity, Friction
from rpp.sim import Simulator
from rpp.things import Object, Tile
from rpp.parameters import Gravity, MinVelocity

sim = Simulator()  # Must initialize sim before creating things.

o = Object(initial_factors={Charge: 0.1, Mass: 1.0, Position: [-0.5, 0.5]}, 
           sim=sim)
print(o.keys())
o2 = Object(initial_factors={Charge: 0.2, Mass: 2.0, Velocity: [1.0, 1.0]},
            sim=sim)
t = Tile(initial_factors={Friction: 1.0},
         sim=sim)
print(t.keys())
g = Gravity(1.0)  # Gravity parameter
min_vel = MinVelocity(1e-5)  # Min velocity parameter, for allowing objects to "stop"
```

    dict_keys([Charge, Mass, Position, Shape, Size, Visible, Order, Label, Text, ID, Collides, Mobile, Velocity, Density, Magnetism, StoredEnergy, InfiniteEnergy, Alive, Done, Acceleration])
    dict_keys([Friction, Shape, Size, Position, Visible, Order, Label, Text, ID, Floor, Heat])


Note that the objects contain factors needed for the lorentz law, but not the tile. The tile contains the factors needed for friction. If we pass the wrong things to lorentz law it will return `DidNotMatch`.


```python
print(lorentz_law(o))
print(lorentz_law(o, t))
```

    DidNotMatch(rpp.rules.transitions.Aggregate[rpp.factors.arrays.Acceleration] <- lorentz_law([(Position, Velocity, Charge, Magnetism), (Position, Velocity, Charge, Mass, Acceleration)]), (Object(id=ID(0)),))
    DidNotMatch(rpp.rules.transitions.Aggregate[rpp.factors.arrays.Acceleration] <- lorentz_law([(Position, Velocity, Charge, Magnetism), (Position, Velocity, Charge, Mass, Acceleration)]), (Object(id=ID(0)), Tile(id=ID(2))))


However, if we pass the correct inputs:


```python
l_apply = lorentz_law(o, o2)
print(l_apply, type(l_apply))
f_apply = apply_friction(o2, t, g)
print(f_apply, type(f_apply))
```

    Acceleration([0. 0.]) += [ 0.01414214 -0.01414214] <class 'rpp.rules.transitions.Aggregate'>
    Acceleration([0. 0.]) += [-0.35355339 -0.35355339] <class 'rpp.rules.transitions.Aggregate'>


The output of these applications are `Aggregate` objects, which inform the sim to aggregate all rules that apply to the target factor as the new value. In this case, what is aggregating is the acceleration of the object, corresponding to the addition of forces.

Rules can also return differentials over time, such as what happens when we apply velocity to change position:


```python
m_apply = move(o2, min_vel)
print(m_apply, type(m_apply))
```

    Position([0. 0.]) += Δt [1. 1.] <class 'rpp.rules.transitions.Differential'>


This rule application is a `Differential`, which says that the position will change in the direction of the velocity, integrated over the time interval, assuming that the velocity is constant over said interval.

Finally, there is a `SetFactor` application, which informs the sim of a new value for a factor. 
Given a set of applications provided from the rules, the sim will decide which rules to apply and when.
Roughly speaking:
1) The sim will first apply all valid rules to everything but position and velocity.
2) Then the sim will apply all rules to velocity
3) Finally, the sim will apply rules to positions, correcting for any collisions that might occur.
* The sim will choose, if the different rules apply to the same factor, which rules to apply. `SetFactor` is applied over `Aggregate` and `Differential`. Additional rule conditions can increase the weight of a rule application over others.

### Rule design

RPP allows users to design their own rules, then add them to the sim. Let's design a custom rule: 


```python
from rpp.rules import TransitionFunction, Differential

@TransitionFunction
def fast_loses_mass(m: Mass, v: Velocity) -> Differential[Velocity]:
    m_new = m * (1. - max(v.norm(), 1.))  # Scale the mass by the velocity
    return Differential[Velocity](m, m_new)

print(fast_loses_mass)
```

    rpp.rules.transitions.Differential[rpp.factors.arrays.Velocity] <- fast_loses_mass([Mass, Velocity])


This is a strange rule, but it helps demonstrate the flexibility for creating custom rules in RPP. Let's see how it would be applied:


```python
fast_loses_mass(o2)
```




    Mass(2.0) += Δt -0.8284271247461903



Note: Some care needs to be taken when applying rules, as they may cause factors to do unusual things (such as become negative).

Finally, we can add the rule to our sim:


```python
sim.add_rule(fast_loses_mass)
print(sim.rules)
```

    [rpp.rules.transitions.Differential[rpp.factors.arrays.Position] <- move([Position, Velocity, MinVelocity]), rpp.rules.transitions.Aggregate[rpp.factors.arrays.Acceleration] <- lorentz_law([(Position, Velocity, Charge, Magnetism), (Position, Velocity, Charge, Mass, Acceleration)]), rpp.rules.transitions.Aggregate[rpp.factors.arrays.Acceleration] <- apply_friction([(Mass, Velocity, Acceleration), (Friction,), Gravity]), rpp.rules.transitions.SetFactor[rpp.factors.number_factors.Mass] <- apply_burn([(Mass, Mobile), (Heat,)]), typing.Tuple[rpp.rules.transitions.SetFactor[rpp.factors.arrays.Velocity], rpp.rules.transitions.SetFactor[rpp.factors.arrays.Acceleration]] <- stop_condition([(Mobile, Alive, Velocity, Acceleration)]), typing.Tuple[rpp.rules.transitions.SetFactor[rpp.factors.number_factors.Mass], rpp.rules.transitions.SetFactor[rpp.factors.arrays.Velocity], rpp.rules.transitions.SetFactor[rpp.factors.bools.Alive], rpp.rules.transitions.SetFactor[rpp.factors.arrays.Acceleration], rpp.rules.transitions.SetFactor[rpp.factors.bools.Visible]] <- kill_condition([Mass, Velocity, Visible, Acceleration, Alive, MinMass]), typing.Tuple[rpp.rules.transitions.SetFactor[rpp.factors.bools.Done], rpp.rules.transitions.SetFactor[rpp.factors.bools.Visible], rpp.rules.transitions.SetFactor[rpp.factors.bools.Mobile]] <- consume([(Mobile, Done, Visible), (Consumes,)]), rpp.rules.transitions.Differential[rpp.factors.arrays.Velocity] <- accelerate([Velocity, Acceleration]), rpp.rules.transitions.Differential[rpp.factors.arrays.Velocity] <- fast_loses_mass([Mass, Velocity])]


Now verify that this rule indeed applies to the mass:


```python
print('before: ', o[Mass], o2[Mass])
sim.step()
print('after: ', o[Mass], o2[Mass])
```

    before:  Mass(1.0) Mass(2.0)
    after:  Mass(1.0) Mass(1.991715728752538)


There is additional functionality in rules, including adding conditions. See the source code for more details.


```python

```
