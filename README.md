# Simple Elastic model for NASA InSight lander-regolith interactions

FEM of the regolith-lander interaction for the NASA InSight lander. This code is used for the results in Stott et al. (2021) "The Site Tilt and Lander Transfer Function from the Short-Period Seismometer of InSight on Mars" in BSSA.

The current model is used to approximate the displacement field of an elastic block when it is deformed by small surface loads imposed by circular feet.

This code produces displacement ratio map for a perturbation from the lander feet, indicating the transfer to the seismometer's feet. 

Uncomment the correct boundary conditions for displacement in vertical, north-south and east-west directions.

Uncomment the correct section to plot the respective vertical, north-south and east-west displacement ratio.

This code was adapted from that of Myhill et al. (2018) "Near-field seismic propagation and coupling through Mars’ regolith: Implications for the InSight mission" 
GitHub repo: https://github.com/bobmyhill/transfer_function

Modified from the FEniCS tutorial demo program: Linear elastic problem.
https://fenicsproject.org/pub/tutorial/html/._ftut1008.html
