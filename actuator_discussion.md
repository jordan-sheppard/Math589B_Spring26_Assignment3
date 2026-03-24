# Discussion of Actuator Location

There is some code under `src/python/scan_actuator.py` that compares locations of the actuator on the membrane. Here is a summary of what it prints:

```
location=(0.500,0.500)  min|beta|=3.000e-32  zero-like modes=27
location=(0.370,0.610)  min|beta|=1.060e-01  zero-like modes=0
location=(0.250,0.500)  min|beta|=3.000e-32  zero-like modes=21
location=(0.210,0.290)  min|beta|=1.243e-01  zero-like modes=0

Tiny couplings can also be found by scanning a dense grid of actuator locations.
Best coarse-grid location by maximin coupling: (0.700, 0.700), score=1.910e-01
```

What this tells us is that an actuator location is better if it has more control over more the modes; that is, if the inner product of the eigenfunctions of this domain with the actuator delta function doesn't go to zero ever. The best locations are the ones who have the most control over the modes with the least amount of movement.

What that means is this: when the actuator is placed at zeroes of the sine and/or cosine eigenfunctions on this square domain, what happens is that no matter what the actuator does or where it is positioned, it has no control over those modes or any of the higher frequency modes - no matter what position the actuator is in, it will have no effect on the value of the eigenmodes at that location, which will all be zero regardless. This means it can't eliminate these modes, which is the entire goal (we are hoping to eliminate the energy from propagating modes, AKA damp those modes entirely).

Thus, the optimal actuator position is one which is not a zero of any modes used up to a certain practical point of truncation. This gives the actuator control of every mode that will vibrate the damped/viscous membrane. According to the simulation, this occurs around (0.7, 0.7), though the position used in the simulations of (0.37, 0.61) is also very good. 

The actuator placed at (0.5, 0.5) misses most of the higher frequency modes, and thus, higher frequencies will still vibrate in the membrane. 