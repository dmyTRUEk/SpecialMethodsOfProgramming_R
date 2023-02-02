# Solutions of tasks for "Special Methods of Programming" by R


## Task 0: hello MPI
**Task:** write "hello MPI" program.

**Solution:** [here](./task0_hello_mpi/src/main.rs).


## Task 1: series sum
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

