# This repo provides code for Example 3.1, based on the original code from https://thaipduong.github.io/SE3HamDL/.

## Dependencies
Our code is tested with Ubuntu 18.04 and Python 3.7, Python 3.8. It depends on the following Python packages: 

```torchdiffeq 0.1.1, torchdiffeq 0.2.3```

```gym 0.18.0, gym 1.21.0```

```gym-pybullet-drones: https://github.com/utiasDSL/gym-pybullet-drones```

```torch 1.4.0, torch 1.9.0, torch 1.11.0```

```numpy 1.20.1```

```scipy 1.5.3```

```matplotlib 3.3.4```

```pyglet 1.5.27``` (pendulum rendering not working with pyglet >= 2.0.0)

***Notes: The NaN error during training with ```torch 1.10.0``` or newer has been fixed!!!!!!!!!. However, training might be slower since we switch to float64. To use the float32 version with torch 1.9.0, run ``` git checkout float32_tensors```.***

## Demo with pendulum
Run ```python ./examples/pendulum/train_pend_SO3.py``` to train the model with data collected from the pendulum environment. It might take some time to train. A pretrained model is stored in ``` ./examples/pendulum/data/pendulum-so3ham_ode-rk4-5p.tar ```

Run ```python ./examples/pendulum/analyze_pend_SO3.py``` to plot the generalized mass inverse M^-1(q), the potential energy V(q), and the control coefficient g(q)
<p float="left">
<img src="figs/pendulum/M_x_all.png" height="180">
<img src="figs/pendulum/V_x.png" height="180">
<img src="figs/pendulum/g_x.png" height="180">
</p>

Run ```python ./examples/pendulum/rollout_pend_SO3.py``` to verify that our framework respect energy conservation and SE(3) constraints by construction, and plots phase portrait of a trajectory rolled out from our dynamics.
<p float="left">
<img src="figs/pendulum/hamiltonian.png" height="170">
<img src="figs/pendulum/SO3_constraints.png" height="170">
<img src="figs/pendulum/phase_portrait.png" height="170">
</p>

Run ```python ./examples/pendulum/control_pend_SO3.py``` to test the energy-based controller with the learned dynamics.
<p float="left">
<img src="figs/pendulum/pendulum_state_control.gif" width="400">
<img src="figs/pendulum/pendulum_animation_control.gif" width="300">
</p>

