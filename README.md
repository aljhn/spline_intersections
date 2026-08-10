# 3D Spline Trajectory Intersections

3D trajectories will never intersect exactly, so find the local minimum distances instead. If these are below some distance threshold consider it an intersection. Each trajectory is given as a list of cubic spline segments, and this implementation will do a global search to find all local minimums.

Implements and builds upon a high-performant polynomial root solver, that is more efficient for lower degree polynomials compared to a companion matrix.

See more detailed explanations here:

* [Part 1](https://aljhn.github.io/posts/1_splinetrajectories)
* [Part 2](https://aljhn.github.io/posts/2_splineintersections)
