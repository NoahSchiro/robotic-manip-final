#show link: set text(fill: blue)

#let title = "MIT 6.4212 final project pre-proposal"
#let authors = "Joshua Martinez, Mark Rifkin, Noah Schiro"

#set page(paper: "us-letter", margin: (x: 0.85in, top: 0.5in, bottom: 0.7in))
#set document(title: title, author: authors)

#align(center, text(size: 14pt, weight: "bold")[#title])
#align(center, authors)

= Overview
We will build a simulated planar pushing testbed in Drake and use it to compare four strategies for probing a box before pushing it to a target pose. Each box has unknown physical parameters: its friction coefficient, its mass, and the offset of its center of mass. By the end of the project we will have:

- A pushing simulation in which each box is randomized with different friction, mass, and center-of-mass offset.
- A perception pipeline that estimates the box's pose from simulated camera images.
- A parameter estimator and an MPC controller shared by all methods.
- Four probing strategies built on top of that controller.
- A quantitative comparison of how the strategies perform.

= Motivation
When a robot handles an object it has never seen, it cannot assume it knows how slippery the object is or how its mass is distributed. These properties determine how the object moves when pushed, so they directly limit how accurately the robot can place it. The robot can learn them by interacting with the object first, but every probe costs time. This raises two questions: how much probing is worth doing, and which probes are most useful? This project studies those questions in a simple, controlled setting.

= Problem Setup
A rigid box rests on a flat surface. Its friction coefficient and mass distribution are unknown to the robot. The robot interacts with the box using a point or flat-faced pusher, and can apply a force at any point on the box's surface. The same set of actions is used both for probing and for pushing toward the goal.

The robot has two simulated sensors in the Drake environment. A force sensor on the pusher measures the contact force applied to the box, and an RGB-D camera observes the scene. Rather than reading the box's pose directly from the simulator, the robot estimates it from the rendered camera images with a perception pipeline: it segments the box from the depth point cloud and registers it against the box's known geometry to recover its position and orientation.

The task has two phases:

+ *Probe* the box to estimate its parameters.
+ *Push* the box to a target pose.

= Approaches
All four methods use the same parameter estimator and the same MPC controller. The only difference between them is how they choose probes.

- *No probing (baseline):* Skip the probing phase entirely. An adaptive MPC controller updates its parameter estimates while pushing toward the goal.
- *Fixed probing:* Run the same predetermined sequence of probes on every box.
- *Online adaptive probing:* At each step, evaluate a set of candidate pushes and pick the one expected to be most informative.
- *Learned probing:* Following ASID, train a neural network that chooses the next push based on the history of observations.

== Parameter Estimator
The shared estimator uses measured pusher forces and perceived box poses to estimate friction, mass, and center-of-mass offset, along with its uncertainty, which online adaptive probing needs. Candidates:

- *Unscented Kalman filter:* estimates the parameters jointly with the box's pose and velocity.
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
We will test each method on a set of randomized boxes and target poses, measuring:

- *Placement accuracy:* final error in position and orientation.
- *Efficiency:* total elapsed time and number of pushes.
- *Estimation accuracy:* error in the estimated friction, mass, and center-of-mass offset.

We will also vary the probing budget to see how the amount of probing trades off against task performance.

= Milestones
- *Progress update 1:* The Drake pushing simulation runs with randomized boxes. The parameter estimator is implemented, the push library is defined, and the no-probing baseline (adaptive MPC) runs end to end using ground-truth box pose.
- *Progress update 2:* The perception pipeline is integrated, so all methods use camera-estimated box pose. Fixed and online adaptive probing are implemented. The evaluation pipeline and metrics are complete, with preliminary results for three of the four methods.
- *Project completion:* Learned probing is trained and evaluated. We deliver the full comparison, the probing-budget analysis, a final video, and the report.

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
- #link("https://arxiv.org/abs/2404.12308")[ASID: Active Exploration for System Identification in Robotic Manipulation]
- #link("https://arxiv.org/pdf/2510.19974")[Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC]
