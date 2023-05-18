# Solutions of tasks for "Special Methods of Programming" by R


# Task 0: hello MPI
**Task:** write "hello MPI" program.

**Solution:** [here](./task0_hello_mpi/src/main.rs).


# Task 1: series sum
**Task:** calculate series sum using different split types.

Chosen series to calculate:

$$ e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots .$$

Lets consider $size=3$. Then split types looks like:
1. Chunks:

$$
\begin{cases}
    \displaystyle \text{part}_0 = 1 + x + \frac{x^2}{2!} + \dots + \frac{x^{N/3}}{(N/3)!},       & \text{if } \text{rank} = 0,\\
    \displaystyle \text{part}_1 = \frac{x^{N/3+1}}{(N/3+1)!} + \dots + \frac{x^{2N/3}}{(2N/3)!}, & \text{if } \text{rank} = 1,\\
    \displaystyle \text{part}_2 = \frac{x^{2N/3+1}}{(2N/3+1)!} + \dots + \frac{x^N}{N!},         & \text{if } \text{rank} = 2.\\
\end{cases}
$$

2. Braid:

$$
\begin{cases}
    \displaystyle \text{part}_0 = 1 + \frac{x^3}{3!} + \frac{x^6}{6!} + \dots,              & \text{if } \text{rank} = 0,\\
    \displaystyle \text{part}_1 = x + \frac{x^4}{4!} + \frac{x^7}{7!} + \dots,              & \text{if } \text{rank} = 1,\\
    \displaystyle \text{part}_2 = \frac{x^2}{2!} + \frac{x^5}{5!} + \frac{x^8}{8!} + \dots, & \text{if } \text{rank} = 2.\\
\end{cases}
$$

**Solution:** [here](./task1_calc_series/src/main.rs).

**Answers:**
```
exp(3) =
    expected        : 20.085536923187668
    by braid  split : 20.085536923187668
    by chunks split : 20.085537612432425
```


# Task 2: optimized scheduling
**Task:** write optimized and non-optimized scheduling algorithms
for evaluating function on a set of points.

Non-optimized scheduling: give equal number of inputs to every process.

Optimized scheduling: give new input only to processes that finished their evaluation.

**Solution:** [here](./task2_optimized_scheduling/src/main.rs).

**Conclusion:** such algorithm shows major advantage only
in case of non-homogeneous (heterogeneous) system
(when evaluation time depends of process' number (rank)),
rather than in case of dependency of function evaluation time of input parameters.

**Numerical results:**
Here "OS" stands for Optimized Scheduling, "NOS" - Non-Optimized Scheduling,
$N=100$, time in seconds, APG - Average Performance Gain.

| Delay distribution         |        NOS times         |         OS times         |          APG          |
| -------------------------- | ------------------------ | ------------------------ | --------------------- |
| linear                     | `6.234, 6.957, 6.093`    | `5.848, 5.377, 5.177`    | `1.18x`               |
| linear to power            | `8.455, 10.933, 9.436`   | `7.789, 5.445, 6.478`    | `1.46x`               |
| linear multiplied          | `15.604, 13.384, 8.875`  | `8.579, 7.717, 10.497`   | `1.41x`               |
| Poisson (N=1000)           | `10.426, 10.309, 10.440` | `10.258, 10.291, 10.272` | `1.01x`               |
| `rank==1 ? 1000 : 100`     | `10.241, 10.228, 10.221` | `2.228, 2.226, 2.226`    | `4.59x`               |
| `rank<=3 ? 1000 : 100`     | `10.228, 10.233, 10.219` | `2.249, 2.229, 2.186`    | `4.60x`               |
| `rank%3==0 ? 1000 : 100`   | `10.237, 10.244, 10.232` | `2.245, 2.181, 2.189`    | `4.64x`               |
| `rank<size/2 ? 1000 : 100` | `10.241, 10.234, 10.228` | `2.232, 2.183, 2.180`    | `4.66x`               |
| N = 300:                   |                          |                          |                       |
| `rank==1 ? 1000 : 100`     | `30.178, 30.232, 30.243` | `4.215, 4.172, 4.186`    | `7.21x`               |
| `rank<=3 ? 1000 : 100`     | `30.230, 30.217, 30.246` | `5.233, 5.172, 5.173`    | `5.82x`               |
| `rank%3==0 ? 1000 : 100`   | `30.255, 30.244, 30.260` | `5.215, 5.182, 5.196`    | `5.82x`               |
| `rank<size/2 ? 1000 : 100` | `30.226, 30.243, 30.227` | `5.219, 5.177, 5.166`    | `5.83x`               |


# Task 2.5: fast matrix multiplication by Fox Algorithm
**Task:** write fast matrix multiplication using Fox Algorithm.

Fox Algorithm consists of calculating each element (or block) of the resulting NxN matrix on each process,
therefore requiring $N^2+1$ processes in simplest scenario.

**Solution:** [here](./task2.5_matrix_multiplication_by_fox_algorithm/src/main.rs).


# Task 3: series sum using barrier, scatter, gather functions.
**Task:** write program that calculates series sum
using MPI functions barrier, scatter, gather.

**Solution:** [here](./task3_series_sum_using_barrier_scatter_gather/src/main.rs).


# Task 5.2: Jacobi iteration method
**Task:** write program that solves SOLE (System Of Linear Equations)
using [Jacobi iteration method](https://en.wikipedia.org/wiki/Jacobi_method).

**Solution:** [here](./task5.2_jacobi_iteration_method/src/main.rs).

