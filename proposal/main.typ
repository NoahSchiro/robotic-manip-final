#show link: set text(fill: blue)

#let title = "MIT 6.4212 final project pre-proposal"
#let authors = "Joshua Martinez, Mark Rifkin, Noah Schiro"

#set page(paper: "us-letter", margin: (x: 0.85in, top: 0.5in, bottom: 0.7in))
#set document(title: title, author: authors)

#align(center, text(size: 14pt, weight: "bold")[#title])
#align(center, authors)

= Overview
We will build a simulated planar pushing testbed in Drake and use it to compare four strategies for probing a object before pushing it to a target pose. Each object has unknown physical parameters: its friction coefficient, its mass, and the offset of its center of mass. By the end of the project we will have:

- A pushing simulation in which each object is randomized with different friction, mass, and center-of-mass offset.
- (Stretch) A perception pipeline that estimates the object's pose from simulated camera images.
- A parameter estimator and an MPC controller shared by all methods.
- Four probing strategies built on top of that controller.
- A quantitative comparison of how the strategies perform.

= Motivation
When a robot handles an object it has never seen, it cannot assume it knows how slippery the object is or how its mass is distributed. These properties determine how the object moves when pushed, so they directly limit how accurately the robot can place it. The robot can learn them by interacting with the object first, but every probe costs time. This raises two questions: how much probing is worth doing, and which probes are most useful? This project studies those questions in a simple, controlled setting.

= Problem Setup
A rigid object rests on a planar surface. Its friction coefficient and mass distribution are unknown. The robot pushes the object's with a point or flat-faced pusher. The valid action space is a finite set of possible pushes, defined by location of contact and the motion profile (speed, contact time) of the pusher. The action space is used for both probing and goal-directed pushing.
The estimator receives the object and pusher pose and velocity states and contact force measurements, but not the physical parameters of the object. The goal is to push the object to a target pose with minimal error.

As a stretch goal, we could prepare the algorithms for real-world deployment by removing state information from the estimator and using a visual perception (point cloud) pipeline and force sensors on the motor joints.

Methods with dedicated probing have two phases, whereas the baseline method skips to the pushing phase:

1. Probing: contact the object to estimate its parameters.
2. Goal-directed pushing: contact the object to move it to a target pose.

= Approaches
All methods share the same parameter estimator and the same MPC controller (which receives the state information and estimated parameters), and continue updating the estimator during goal-directed pushing. Only the probe-selection policy differs.

- *No probing (baseline):* Skip the probing phase entirely. An adaptive MPC controller updates its parameter estimates while pushing toward the goal.
- *Fixed probing:* Run the same predetermined sequence of probes on every object.
- *Online adaptive probing:* At each step, evaluate a set of candidate pushes and pick the one expected to be most informative.
- *Learned probing:* Following ASID, train a neural network that chooses the next push based on the history of observations.

== Parameter Estimator
The shared estimator uses measured pusher forces and perceived object poses to estimate friction, mass, and center-of-mass offset, along with its uncertainty, which online adaptive probing needs. Candidates:

- *Unscented Kalman filter:* estimates the parameters jointly with the object's pose and velocity.
- *Particle filter:* represents the parameter distribution with samples; handles multimodal posteriors.
- *Grid Bayes filter:* discretizes the parameter space and updates the probability of each grid cell after every push.

== Course Topics Covered
- Contact and friction modeling in simulation
- Geometric perception and pose estimation
- Non-prehensile manipulation
- Optimization-based control (MPC)
- System identification
- Learning-based policies

== Evaluation
Each method is evaluated over the same set of randomized objects and target poses, using:

- *Placement accuracy:* final error in position and orientation.
- *Efficiency:* total elapsed time and number of pushes.
- *Estimation accuracy:* error in the estimated friction, mass, and center-of-mass offset.

We will also vary the probing budget (the number of dedicated probing pushes allowed) to characterize the trade-off between probing effort and task performance.

= Milestones
- *Progress update 1*: Working Drake pushing simulation with a randomized object. Push library defined. Compare MPC using true parameters against MPC using fixed nominal parameters to assess potential benefit of identification. Check whether the push library distinguishes the unknown parameters (and reduce the parameter set if necessary).
- *Progress update 2*: Parameter estimator implemented. No-probing adaptive MPC baseline running end-to-end.
- *Progress update 3*: Fixed and online adaptive probing implemented. Evaluation pipeline and metrics complete. Preliminary results for three of the four methods.
- *Project completion*: Learned probing trained and evaluated. Full comparison, budget-sweep analysis, final video, and report.

= Division of Work
Tasks:
- Simulation environment
- Perception pipeline
- Push action library
- Evaluation pipeline
- Parameter estimator
- MPC controller
- No-probing and fixed-probing baselines
- Online adaptive probing
- Learned probing

*Joshua Martinez:* ...

*Mark Rifkin:* ...

*Noah Schiro:* ...

= References
- #link("https://arxiv.org/abs/2404.12308")[ASID: Active Exploration for System Identification in Robotic Manipulation]: reference on offline learning of information-maximizing exploration policies
- #link("https://arxiv.org/pdf/2510.19974")[Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC]: reference on predictive control for planar pushing with unknown parameters. Implements CI-MPC which has a continuous rather than discrete action space (we use a discrete action space and focus on probing strategies).

