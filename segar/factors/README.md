# Factors tutorial

This tutorial covers the atomic units of The Sandbox Environment for Generalizable Agent Research (SEGAR): the factors.
Factors make up the underlying state space, and their values make up the underlying states.
Factors are _types_, and many of them inherit from built-in types, such as `float`, `int`, `bool`, etc.

### Factors
Here are some basic examples of creating and manipulating factors:


```python
from segar.factors import Charge, Mass, Alive, Position, Factor
c1 = Charge(0.3)
c2 = Charge(0.3)
m = Mass(0.3)
a = Alive(True)
p = Position([0.1, 0.1])
```


```python
print(c1, c2)
```

    Charge(0.3) Charge(0.3)


Factors can take on values, so they can be compared and operated on, but they are _unique_


```python
print(c1 == c2, c1 is c2, c1 == m, c1 + m, p + m)
```

    True False True 0.6 Position([0.4 0.4])


The base type `Factor` is a generic type:


```python
f = Factor[float](0.4)
print(f, f + c1)
```

    Factor(0.4) Charge(0.7)


### Factor Containers
In order to be useable by the sim, sets factors must be encapsulated in a container called a `FactorContainer`:


```python
from factors import FactorContainer
fc = FactorContainer({Charge: 0.3, Position: [.1, .3], Alive: False})
print(fc)
```

    FactorContainer({Charge: Charge(0.3), Position: Position([0.1 0.3]), Alive: Alive(False)})


Factor containers are basically dictionaries with Factor type keys and Factor values. We can check if a FactorContainer contains a factor or not:


```python
print(fc.has_factor(Charge), fc.has_factor(Mass))
```

    True False


Changing factors of their containers is protected:


```python
try:
    fc[Charge] = 0.4
    raise RuntimeError('Should not get here')
except ValueError as e:
    print('Error message:', e)

```

    Error message: Factor in-place operations are protected.


But in-place operations can be allowed:


```python
with fc.in_place():
    fc[Charge] += 0.4
print(fc[Charge])
```

    Charge(0.8)


### Random factor generation
SEGAR comes with factor generation from random numbers. This can help generate distributions of factors used for initialization, or even can be used to generate stochastic transition functions:


```python
from segar.factors import UniformNoise, GaussianNoise, GaussianNoise2D
u1 = UniformNoise(0., 1.)  # Uniform from 0 to 1
u2 = UniformNoise(0.1, 1.)  # Uniform from 0.1 to 1
g1 = GaussianNoise(0., 0.1)  # Normal distribution with 0 mean and 0.1 std
g2 = GaussianNoise(0.1, 0.2)  # Normal distribution with 0.1 mean and 0.2 std
g2D = GaussianNoise2D([0., 0.1], [0.1, 0.2])  # 2D normal distribution

print(u1.sample(), g1.sample(), g2D.sample())
```

    Factor(0.2933396462141178) Factor(-0.17201698297461065) Factor([0.13358935 0.15078861])


SEGAR allows the researcher to compare distributions (via the Wasserstein distance), ultimately to measure how different different sets of tasks are (say between train and test in a generalization experiment).


```python
print(u1.distance(u1), u1.distance(u2), u1.distance(g1), g1.distance(g1), g1.distance(g2))
```

    0.0011885052012289158 0.05644332784973156 0.5370324519869282 0.0 0.02715728752538099


Note that the distance of `u1` with itself is not measured as zero. This is because we are using a sample-based approximation for the Wasserstein-2 in most cases. More accurate measurement of the W2 can be accomplished by increasing the number of samples used. This increases computational cost; however, as these measures are only intended to be used to compare tasks, this can be done separate of running agents for analysis and results.


```python

```
