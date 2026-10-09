#show link: set text(fill: blue)

#let title = "MIT 6.4212 final project pre-proposal"
#let authors = "Joshua Martinez, Mark Rifkin, Noah Schiro"

#set page(paper: "us-letter", margin: (x: 0.85in, top: 0.5in, bottom: 0.7in))
#set document(title: title, author: authors)

#align(center, text(size: 14pt, weight: "bold")[#title])
#align(center, authors)

= Deliverable
We will build a simulated planar pushing testbed in Drake and compare four probing strategies for identifying unknown physical parameters of a box before pushing it to a target pose. The deliverables are:

- A pushing simulation with a randomized object with variable (unknown) friction coefficient, mass, and center-of-mass offset.
- A shared parameter estimator and MPC controller.
- Four probing strategies built on top of the controller.
- A quantitative comparison of the strategies.

= Motivation
A robot manipulating an unfamiliar object cannot assume it knows the object's friction or mass distribution. These properties govern how the object responds to contact and directly limit placement accuracy. The robot can interact with the object to learn them, but interaction costs time. How much probing is worthwhile and how the probes should be chosen? This project isolates that question in a simple, controlled setting.

= Problem Setup
A rigid object rests on a planar surface. Its friction coefficient and mass distribution are unknown. The robot pushes the box with a point or flat-faced pusher. The valid action space is a finite set of possible pushes, defined by location of contact and the motion profile (speed, contact time) of the pusher. The action space is used for both probing and goal-directed pushing.
The estimator receives the object and pusher pose and velocity states and contact force measurements, but not the physical parameters of the object. (In the real world, these would be estimated from force-torque sensors and vision.) The goal is to push the object to a target pose with minimal error.

Methods with dedicated probing have two phases, whereas the baseline method skips to the pushing phase:

1. Probing: contact the object to estimate its parameters.
2. Goal-directed pushing: contact the object to move it to a target pose.

= Approaches
All methods share the same parameter estimator and the same MPC controller (which receives the state information and estimated parameters), and continue updating the estimator during goal-directed pushing. Only the probe-selection policy differs.

- *Baseline*: no probing. use adaptive MPC and estimate parameters during goal-directed pushing phase.
- *Fixed probing*: predetermine probe sequence.
- *Online adaptive probing*: evaluate candidate probesand select the one that maximizes expected information gain.
- *Learned probing*: train a neural probing policy in randomized simulation using an information-based reward inspired by ASID.

== Course content covered
- Contact and friction modeling in simulation
- Non-prehensile manipulation
- Optimization-based control (MPC)
- System identification
- Learning-based policies

== Evaluation
Each method is evaluated over the same set of randomized boxes and target poses, using:

- Final placement error (position and orientation).
- Elapsed time and push count (including probing and placement, and reporting computation time).
- Parameter estimation error.

We will also vary the probing budget (the number of dedicated probing pushes allowed) to characterize the trade-off between probing effort and task performance.

= Milestones
- *Progress update 1*: Working Drake pushing simulation with a randomized box. Push library defined. Compare MPC using true parameters against MPC using fixed nominal parameters to assess potential benefit of identification. Check whether the push library distinguishes the unknown parameters (and reduce the parameter set if necessary).
- *Progress update 2*: Parameter estimator implemented. No-probing adaptive MPC baseline running end-to-end.
- *Progress update 3*: Fixed and online adaptive probing implemented. Evaluation pipeline and metrics complete. Preliminary results for three of the four methods.
- *Project completion*: Learned probing trained and evaluated. Full comparison, budget-sweep analysis, final video, and report.

= Division of Work

Tasks:
- simulation environment
- push action library
- and evaluation pipeline
- parameter estimator
- MPC controller
- no-probing and fixed-probing baselines
- online adaptive probing
- learned probing

Joshua Martinez: ...

Mark Rifkin: ...

Noah Schiro: ...

= References
- #link("https://arxiv.org/abs/2404.12308")[ASID: Active Exploration for System Identification in Robotic Manipulation]: reference on offline learning of information-maximizing exploration policies
- #link("https://arxiv.org/pdf/2510.19974")[Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC]: reference on predictive control for planar pushing with unknown parameters. Implements CI-MPC which has a continuous rather than discrete action space (we use a discrete action space and focus on probing strategies).

