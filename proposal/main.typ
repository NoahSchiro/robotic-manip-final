#show link: set text(fill: blue)

#let title = "MIT 6.4212 final project proposal"
#let authors = "Joshua Martinez, Mark Rifkin, Noah Schiro"

#set page(paper: "us-letter", margin: (x: 0.85in, top: 0.5in, bottom: 0.7in))
#set document(title: title, author: authors)

#align(center, text(size: 14pt, weight: "bold")[#title])
#align(center, authors)

= Introduction

Robots handling unfamiliar objects must account for unknown friction
and mass distribution. Interaction can reveal these properties, but
dedicated probing costs time that could be spent pursuing the task.
We will build a planar pushing testbed in Drake to evaluate different approaches to manipulating objects with unknown physical properties. By comparing a learned probing policy followed by MPC to both a no-probing adaptive MPC baseline and an end-to-end neural policy, we will investigate whether a separate probing phase improves performance enough to justify its cost. We will also compare learned and random probing with equal probe budgets and a shared estimator and MPC in order to evaluate the benefit of deliberate probe selection. 

= Problem setup
== Object and unknown parameters
We we will initially use a fixed, known object with uniform mass and a point mass displaced from the original COM by a 2-dimensional vector $b$ in the $x y$-plane. The unknown parameters are $eta = (mu, b) $, with $mu$ the friction coefficient between the object and the table. The total mass is fixed and known, though it may be added as a parameter after initial validation. The center of mass and rotational inertia are uniquely determined by $eta$, the point mass, the total mass, and the object geometry. The parameter set could be additionally extended to more complex friction modeling. 

== Observations and actions
The robot has access to the object and pusher pose and velocity, in addition to its contact force during each push.
Each action consists of pushing along an inward surface normal for a set speed and time $S_1$ and $T_1$ before the arm repositions for the next push (waiting and not pushing is also a valid action). The pushes are selected from a finite library of contact locations and commanded pusher speeds. Each push uses a fixed acceleration ramp to reach its commanded speed. 

= Dynamics modeling
A dynamics model is required for estimation and planning. We will initially use short-horizon Drake simulations for parameter estimation and MPC prediction. If runtime is prohibitive, we can adjust the hypothesis grid and planning horizon. 

= Parameter estimation
Tentatively, we will use a grid Bayes filter over $eta = (mu, b)$, starting at 9 values per parameter and $N=81$ hypotheses. Starting from uniform probabilities, we update after each push $t$: 
$ p_(t+1, i) = frac(p_(t,i) P (D_t | eta_i),sum_(j=1)^N p_(t,j) P (D_t | eta_j) ) $
Here, $D_t$ is the measured outcome of push $t$, and $P(D_t | eta_i)$ is its likelihood under hypothesis $i$. For each hypothesis, the dynamics model (Drake initially) predicts the object's motion under the executed push. Let $e_(t,i)$ be the measured minus predicted endpoint pose and velocity (with angles wrapped). We assume zero-mean Gaussian prediction errors to define: 
$ P(D_t | eta_i) prop exp(-frac(1,2) e_(t,i)^T R^(-1) e_(t,i)) $
with $R$ the covariance of prediction errors, calibrated on sample trajectories. 
As the confidence in the parameter estimation grows, we can adjust the grid search size to improve efficiency. 
We could evaluate other estimation options if the grid search proves too costly. 

= MPC controller
MPC optimizes over candidate push sequences using the estimator's maximum-probability hypothesis. Its pose cost is:
$V(q) = norm(r-r_g)^2 / ell^2 +  lambda_phi op("wrap")(phi-phi_g)^2, $
where $q=(r, phi)$, $r$ is position ($r_g$ goal position), $phi$ is orientation,  $ell$ is a representative object length, and $lambda_phi$ is a weight on orientation error. It minimizes predicted pose costs, in addition to action costs (the time that it takes to move the arm to the pushing pose, and the pushing time) over a short horizon (1-3 pushes/repositions), executes the first action, and replans. 

= Approaches 

- *Random probing*: select uniformly from the feasible push library for a fixed probing time $T_P$. After probing, push toward goal using MPC. Update the estimator throughout probing and placement
- *Learned adaptive probing*: Using PPO with randomized simulations, train a neural policy to output a probability distribution over the push library, given the current object/pusher pose/velocities, the estimator's belief probabilities, and the remaining probe budget. Reward based on the reduction in parameter uncertainty the probe generates. Using the Shannon entropy $H(p) = - sum_(i=1)^N p_i log(p_i)$, set reward $r_t = H(p_t) - H(p_(t+1))$. After probing time $T_P$, follow the same MPC controller as random probing. 
- *End-to-end neural policy:* Using PPO with random simulation, train a neural policy to output a probability distribution over push library, given a history of object/pusher pose and force. Reward based on reduction in distance from target pose, minimizing time, and bonus for successful placement.
- *(Baseline) No-probing MPC*: skip the probing step ($T_P=0$) and update the estimator as the MPC performs goal-directed pushing. 

= Evaluation
We will test the methods on a shared set of (held-out) parameters, initial poses, and target poses, with set maximum time. Define success as final pose within specified tolerances. We will report: success rate, final pose errors, total time (including probing), online computation time, and offline training cost. For estimator methods, we will also report parameter-estimation error. 

= Milestones
- *Milestone 1 - Feasibility:* Build simulation and push library.
  Benchmark Drake prediction runtime, check whether parameters are identifiable given push library, and compare true-parameter
  and nominal-parameter MPC. Verify PPO training of the end-to-end placement policy with fixed physical parameters. 
- *Milestone 2 - Integration:* Validate the estimator and complete random and no-probing MPC. Begin PPO training of the learned prober (based on estimator entropy reduction), and extend end-to-end training with randomized parameters. 
- *Milestone 3 - Comparison:* Compare all approaches on shared validation cases. Refine methods based on preliminary results. 
- *Final submission: * Complete evaluation on a held-out test set, analyze results, produce final report and video. 
= Division of work

- *Joshua Martinez:* Object parameterization, estimator,
  random-probing and no-probing baseline.
- *Mark Rifkin:* MPC, dynamics model evaluation,
  learned adaptive probing.
- *Noah Schiro:* Simulation environment, action library,
  end-to-end neural policy.

= Course topics covered
Contact and friction modeling, non-prehensile manipulation,
Bayesian estimation, MPC, and reinforcement learning.

= References

- #link("https://arxiv.org/abs/2404.12308")[ASID]:
  inspiration for learned probing.
- #link("https://arxiv.org/abs/2510.19974")[Push Anything]:
  related work on predictive control for planar pushing.
