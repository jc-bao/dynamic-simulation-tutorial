---
theme: seriph
highlighter: shiki
class: 'text-center'
title: Physics-based Simulator Tutorial
author: Chaoyi Pan, Xiaofeng Guo, Nikhil Sobanbabu
# background: './assets/joint.gif'
layout: cover
---
# 📚 Physics-based Simulator Tutorial      

Chaoyi Pan, Xiaofeng Guo, Nikhil Sobanbabu

---

# 💁 Agenda

## 1️⃣ Can simulate: rigid body

Chaoyi Pan: basic dynamics of rigid body simulation

## 2️⃣ Hard to simulate: soft body

Xiaofeng Guo: advanced contact, soft body dynamics

## 3️⃣ Failed to simulate: unmodelled effects

Nikhil Sobanbabu: defects in sim, sim2real, real2sim

---
layout: center
---

# 1️⃣ Rigid Body Dynamics

---

# Simulation sometimes can go weird…

### Why my simulator blow up?
### Why my robot sink into the floor?
### Why my block slip away from robot’s hand?
### Why the simulation is not deterministic?

---

# Today's roadmap

<v-clicks depth="2">

## Basics: 

- Coordinate: minimum/generalized coordinate, maximum/cartesian coordinate

## Discretization, integration: Explicit, RK, Implicit

- Multi-rigid-body System: 
- Least-action Principle and Euler Lagrange Equation
- Lagrangian Dynamics and Hamilton Dynamics
- Simulation with equality constraints / external force

## Contact

- Basic Contact Dynamics: smoothing method, hybrid method, time-stepping method
- Friction Modelling: maximum dissipation principle, LCP problem

## Case study: 

- X-Ray of Common Simulators: Mujoco, PhyX, Drake

</v-clicks>

---
layout: center
---

# Basics

## Coordinate, Discretization, Euler-Lagrange Equation


---

# Coordinate

## Minimum/Generalized Coordinate

Representation: in joint space $q, \dot{q}, \ddot{q}$

- ✅ Minimize the number of variables, used on most of the simulators
- ❌ Hard to compute constraints

Example: double-pendulum

## Maximum/Cartesian Coordinate

Representation: in task space $x, \dot{x}, \ddot{x}$

- ✅ Intuitive and easy to derive for simple system
- ❌ Leads to more constraints, extra computation

## Example: pendulum: 

$q = \theta, \dot{q} = \dot{\theta}, \ddot{q} = \ddot{\theta}$

$r = [x, y]^T, x^2 + y^2 = l^2$

- 

---

# Integration

$$
\dot{x} = f(x, u, t)
$$

## Explicit Integration

Euler: $x_{n+1} = x_n + \Delta t f(x_n, u_n, t_n)$
- 👍 Easy to implement, easy to calculate gradient 
- 👎 Diverge in energy

RK4: $x_{n+1} = x_n + \frac{\Delta t}{6} (k_1 + 2k_2 + 2k_3 + k_4)$
- 👍 More stable, more accurate
- 👎 Still diverge in energy, also $n\times$ evalutation time. 

## Implicit Integration
 
Solve $x_{n+1} - x_n - \Delta t f(\frac{x_{n+1}+x_n}{2}, u_n, t_n) = 0$

- 👍 Stable, conserve energy
- 👎 Hard to calculate gradient

---

# Optimization 

TODO

---
layout: center
---

# 2️⃣ Multi-rigid-body System

## How would an unconstrained system evolves over time?

---

# What is Euler-Lagrange Equation

## Newton's law for general system

Newton's law: $M\ddot{q} - \nabla V = 0$

Euler-Lagrangian Equation: $\frac{d}{dt} \frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0$

- $L = T - V$
- $T$: kinetic energy
- $V$: potential energy

---

# Why Euler-Lagrange Equation over Newton's Law

$$
\frac{d}{dt} \frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0
$$

<v-clicks>

- Optimization formulation: convert simulation problem into a calculating L
- Coordinate invariant
- Clear physics interpretation: minimum action principle $\min_q \int L(q, \dot{q}, t) dt$

</v-clicks>

---

# Physics Interpretation of Euler-Lagrange Equation

## Least Action Principle, Hamilton's Principle

$$
\min_q \int L(q, \dot{q}, t) dt, \quad L = T - V  \\
\text{s.t.} \quad q(t_0) = q_0, q(t_1) = q_1
$$

- Explain how the system evolves from one state to another
- Choose a kinematics path such that: the average of kinetic energy and potential energy is minimized.
- Least action principle describe global behavior of the system, while Euler-Lagrange equation describe local behavior. They should be consistent. (can be proved. )

<img src="./assets/least_action.png" style="width: 50%"/>

---

# Lagrangian Dynamics with External Force

"Lagrangian-D'Alembert" principle, aka "virtual work principle“: 

$$ {1|all}
\partial_q \int L' dt = \partial_q \int (L + F^T q) dt = 0 \\
 \frac{d}{dt} \frac{\partial L'}{\partial \dot{q}} - \frac{\partial L'}{\partial q} = F
$$

---

# Special Case: E-L Equation for multi-joint rigid body system

## General form for articulated rigid body:

$$
L = T - V = \frac{1}{2} \dot{q}^T M(q) \dot{q} - V(q)
$$

## Manipulator Dynamics

$$
M(q) \ddot{q} + C(q, \dot{q}) \dot{q} + G(q) = \tau
$$

- Used in most of the simulators, you can read those value from urdf parser. 
- $M(q)$: inertia matrix
- $C(q, \dot{q})$: Coriolis matrix
- $G(q)$: gravity term (sometimes integrated into $C$, called that "dynamic bias")
- $\tau$: joint torque
- Computationally expensive, $O(n^3)$ for $n$ joints, sometimes is sparse and can be reduced to $O(n)$ by exploiting the structure. 


---

# Example

## Cart-pole

$$
L = \frac{1}{2} m_1 \dot{x}^2 + \frac{1}{2} m_2 \dot{x}^2 + \frac{1}{2} I_1 \dot{\theta}^2 + \frac{1}{2} I_2 \dot{\theta}^2 + m_2 g x \cos \theta \\
\frac{d}{dt} \frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = M(q) \ddot{q} + C(q, \dot{q}) \dot{q} + G(q) = \tau \\
M(q) = \begin{bmatrix}
m_c + m_p & m_p l_p \cos \theta \\
m_p l_p \cos \theta &  m_p l_p^2
\end{bmatrix} \\
C(q, \dot{q}) = \begin{bmatrix}
- m_p l_p \dot{\theta}^2 \sin \theta \\
0
\end{bmatrix} \\
G(q) = \begin{bmatrix}
0 \\
- m_p g l_p \sin \theta
\end{bmatrix} 
$$

---

# Constraints in E-L Equation

## Equality Constraint

$$
C(q, \dot{q}) = 0
$$

with equality constraints, the E-L eqn. is:

$$
\partial_q \int L' dt = \partial_q \int (L + \lambda^T c(q)) dt = 0 \\
\frac{d}{dt} \frac{\partial L'}{\partial \dot{q}} - \frac{\partial L'}{\partial q} - \lambda^T \frac{dc(q)}{dq} = 0
$$

- $\lambda$ is the Lagrange multiplier (**constraint force**) 
- $J(q) = \frac{dc(q)}{dq}$ is the Jacobian matrix of the constraint. 

---

# Final Version of Manipulator Dynamics

## Manipulator Dynamics

$$
M(q) \ddot{q} + C(q, \dot{q}) \dot{q} + G(q)  = \tau + J^T \lambda
$$

## KKT Condition

$$
\begin{bmatrix}
M & -J^T \\
J & 0
\end{bmatrix}
\begin{bmatrix}
\ddot{q} \\
\lambda
\end{bmatrix}
= \begin{bmatrix}
\tau - C(q, \dot{q}) - G(q) \\
-\frac{d}{dq} (J \dot{q}) \dot{q}
\end{bmatrix}
$$

Notes: 
- $J(q)$ is introduced since the constraint is applied in the maximum space, need to transform to the joint space.
- For system with equality constraints, the final form of manipulator dynamics can be solved explicitly without iteration. 

--- 

# 2️⃣ Multi-rigid-body System

## Takeaway

$$
M(q) \ddot{q} + C(q, \dot{q}) \dot{q} + G(q)  = \tau + J^T \lambda
$$

- **First principle**: Minimum action principle, everything can be derived from it.
- **Forward Dynamics is easy**: multi-rigid-body system + equality constraints = explicit solution. 
- More interesting things: 
  - Dual form: Hamiltonian Dynamics
  - Discretization: Legendre Transform, Variational Integrator

---
layout: center
---

# 3️⃣ Contact

## The brightest jewel in the crown of simulation

---

# Why contact is such a headache?

## Impact = infinite acceleration/force

![alt text](./assets/contact_pic.png)

---

# Why contact is such a headache?

## Friction = non-continuous behavior

![alt text](./assets/friction.png)

---

# Common workarounds

<v-clicks>

| Method                    | Description                                                                 | Pros                                                                                                                                       | Cons                                                                                                                                                      | Demonstration Image |
|---------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| Smooth Contact Model      | Uses a smooth contact force, spring-damper model. Used in Mujoco*.          | :white_check_mark: Easy to implement <br> :white_check_mark: Differentiable <br> :white_check_mark: Easy to control                       | :x: Not accurate <br> :x: Interpenetration <br> :x: Energy dissipation not physical                                                                      | ![Demo](./assets/spring.png) |
| Hybrid/Event-Driven Method| Detects contact using a guard function with an extra jump map. Common in control. | :white_check_mark: Not stiff <br> :white_check_mark: Can use standard ODE solver                                                          | :x: Scales poorly with number of contacts <br> :x: Handles simultaneous contacts poorly <br> :x: Not differentiable <br> :x: Hard to design general simulators | ![Demo](./assets/hybrid.png) |
| Time-stepping Methods     | Computes contact force at each time step to satisfy contact constraints. Used in Gazebo, Dart, Bullet. | :white_check_mark: Scales well with number of contacts <br> :white_check_mark: Handles simultaneous contacts well <br> :white_check_mark: Gives correct physics | :x: Computationally expensive (solve optimization at each time step) <br> :x: Not differentiable                                                          | ![Demo](./assets/optimization.png) |

</v-clicks>

--- 

# Time-stepping Method

## Solving a Constrained Optimization Problem

$$
\min_q \int L(q, \dot{q}, t) dt \\
\text{s.t.} \quad q(t_0) = q_0, q(t_1) = q_1 \\
\quad c(q, \dot{q}, t) \geq 0
$$

- If $c$ only has impact and no friction, the problem is convex. 
- The problem could be non-convex due to the friction cone. 

# Handle Friction

## Maximum Dissipation Principle

The optimization problem solve for friction (dissipation means the energy loss in the system)

$$
\min_b \dot{T} \\
\text{s.t. } b \le \mu n \\
$$

## SOCP Formulation

$$
\min_b \dot{q}^T M b \\
\text{s.t. } b \le \mu n \\
$$

- When $\|b\| = 0$, the solver would fail. So we can use conic primal-dual interior point method to solve it. 
- A simple hack: use a smoothed 2-norm: $\|b\| = \sqrt{b^T b + \epsilon^2} - \epsilon$. 
- Impact, friction and q can be solved at the same time. 
- SOCP is NP-hard. 

<img src="./assets/undetermined_friction.png" style="width: 50%"/>

---

# Further Simplify the Contact Problem

## LCP Formulation

linear approximation of friction cone. 

$$
\|b\|_1 \le \mu n
$$

By introducing a new variable $d$, the problem can be written as a LCP problem:

$$
b = \begin{bmatrix}
1 & 0 & -1 & 0 \\
0 & 1 & 0 & -1
\end{bmatrix} d\\
[1 \ 1 \ 1 \ 1] d \le \mu n \\
$$

![alt text](./assets/lcp.png)

- Problem is still non-convex, NP-hard. 
- But there are better solver can handle it. 
- Used in most of the simulators. 

---

# 3️⃣ Contact

$$
\min_q \int L(q, \dot{q}, t) dt \\
\text{s.t.} \quad q(t_0) = q_0, q(t_1) = q_1 \\
\quad c(q, \dot{q}, t) \geq 0
$$

## Takeaway

- **Contact solver is a constrained optimization problem**: most of time it is non-convex. 
- **Contact always comes with approximation**: 
  - knowing what kind of approximation the solver is using can help you better understand the behavior of the system. 
  - But even the best solver, the contact is always approximated in the solving stage to make it tractable. 

> we haven't even mentioned the contact detection yet, which is another big topic. 

---
layout: center
---

# 4️⃣ Case Study

## X-Ray of Common Simulators: Mujoco, PhysX

---

# MuJoCo

- Contact solver: smoothing method + extra constraints. (i.e. multiple string model)

$$
\begin{aligned}
(\dot{v}, \dot{\omega}) & = \arg \min_{(x, y)} \left\| x - M^{-1}(\tau - c) \right\|_M^2 + \| y - a_{\text{ref}} \|_R^{-1} \cdot \text{Huber}(\eta) \\
\text{subject to} \quad & J_E x_E - y_E = 0, \\
& J_F x_F - y_F = 0, \\
& J_C x_C - y_C \in K^*.
\end{aligned}
$$


- 👑 Open source, clean JAX implementation, easy to modify and blinding fast
- ✅ Differentiable, contact force always available
- ✅ Lots of options to regularize the contact solver, could be very realistic (tons of contact modes `condim`)
- ✅ Problem is always convex, easy to solve
- ❌ Hard to control, hard to manually tune the contact parameters
- ❌ Ecosystem: GPU pipeline rendering, generative model support, etc. 

---

# PhysX

- Contact solver: time-stepping method, contact modelling with LCP
- Solver: Projected Gauss-Seidel (1st order, like coordinate descent)

- ✅ Open source (but issac series is not)
- ✅ Board learning community, first GPU-based simulator
- ✅ Comprehensive API and features
- ❌ Less modular
- ❌ The implementation is hard to read. Also, the wrapper is heavy. 
- ❌ Little bit hard to deploy, binding to distro and nvidia. 