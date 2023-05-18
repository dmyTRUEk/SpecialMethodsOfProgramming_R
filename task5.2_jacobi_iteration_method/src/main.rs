//! Task 5.2: impl algorithm for solving SOLE
//! (System Of Linear Equations) using Jacobi iteration method.

use mpi::traits::*;


#[allow(non_camel_case_types)]
type float = f64;


const SOLE_SIZE: usize = 3;

const SOLE_COEFS_A: [[float; SOLE_SIZE]; SOLE_SIZE] = [
    [20., 3., 1.],
    [4., 10., 2.],
    [1., 3., 30.],
];

const SOLE_COEFS_B: [float; SOLE_SIZE] = [
    10.,
    8.,
    6.,
];

const TARGET_PRECISION: float = 1e-9;
const MAX_ITERS: usize = 1_000;


fn calc_new_x<const SOLE_SIZE: usize>(
    sole_coefs_a: [[float; SOLE_SIZE]; SOLE_SIZE],
    sole_coefs_b: [float; SOLE_SIZE],
    xs: [float; SOLE_SIZE],
    k: usize, // index of x in `xs`
) -> float {
    (
        sole_coefs_b[k]
        - (0..SOLE_SIZE).into_iter()
            .filter(|&i| i != k)
            .map(|i| sole_coefs_a[k][i] * xs[i])
            .sum::<float>()
    ) / sole_coefs_a[k][k]
}


fn calc_solution_precision<const SOLE_SIZE: usize>(
    sole_coefs_a: [[float; SOLE_SIZE]; SOLE_SIZE],
    sole_coefs_b: [float; SOLE_SIZE],
    xs: [float; SOLE_SIZE],
) -> float {
    (0..SOLE_SIZE).into_iter()
        .map(|k|
            (0..SOLE_SIZE).into_iter()
                .map(|i| sole_coefs_a[k][i] * xs[i])
                .sum::<float>()
            - sole_coefs_b[k]
        )
        .map(|delta| delta.powi(2))
        // .map(|delta| delta.abs())
        .sum::<float>()
        .sqrt()
}


fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    assert_eq!(SOLE_SIZE, size as usize, "need exactly {SOLE_SIZE} processes, but {size} was given.");
    let rank = world.rank();

    let mut xs = [0.; SOLE_SIZE];
    let mut iter: usize = 0;
    loop {
        iter += 1;
        if iter > MAX_ITERS { break }
        world.barrier();
        if rank == 0 { println!("iter: {iter}") }

        println!("rank {rank}: xs = {xs:?}");
        let mut is_precision_enough: i8 = if rank == 0 {
            // calculate precision
            let precision = calc_solution_precision(SOLE_COEFS_A, SOLE_COEFS_B, xs);
            println!("precision = {precision}");
            if precision < TARGET_PRECISION {
                println!("precision is enough, stoping...");
                1 // true
            } else {
                println!("precision is not enough, continuing...");
                0 // false
            }
        } else {
            -1 // none
        };
        world.process_at_rank(0).broadcast_into(&mut is_precision_enough);
        if is_precision_enough == 1 { break }

        // calculate new x
        let new_x = calc_new_x(SOLE_COEFS_A, SOLE_COEFS_B, xs, rank as usize);
        xs[rank as usize] = new_x;

        // broadcast new xs
        for k in 0..SOLE_SIZE {
            world.process_at_rank(k as i32).broadcast_into(&mut xs[k])
        }

        world.barrier();
        if rank == 0 { println!() }
    }
}

