//! Task 3: impl algorithm to calculate series sum
//! using these mpi funcions: `MPI_Barrier`, `MPI_Scatter`, `MPI_Gather`.

use mpi::traits::*;
use rand::{Rng, thread_rng};


const POINTS_PER_PROCESS: usize = 2;


fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    assert!(size > 1, "need more than 1 processes, but {size} was given.");
    let rank = world.rank();
    let process_root = world.process_at_rank(0);

    if rank == 0 { println!("world.size = {size}") }

    let mut series_part = [0.; POINTS_PER_PROCESS];
    if rank == 0 {
        let mut rng = thread_rng();
        let series_full: Vec<f64> = (0..POINTS_PER_PROCESS*(size as usize)).into_iter()
            .map(|_| rng.gen_range(0. ..= 1.))
            .collect();
        assert_eq!(POINTS_PER_PROCESS*size as usize, series_full.len());
        println!("generated array to calc sum to: {series_full:?}");
        process_root.scatter_into_root(&series_full, &mut series_part);
    } else {
        process_root.scatter_into(&mut series_part);
    }
    println!("rank {rank}: received `series_part`: {series_part:?}");

    world.barrier();

    let series_part_sum: f64 = series_part.into_iter().sum();
    println!("rank {rank}: sum = {series_part_sum}");

    world.barrier();

    if rank == 0 {
        let mut series_sums = vec![0.; size as usize];
        process_root.gather_into_root(&series_part_sum, &mut series_sums);
        println!("series_sums = {series_sums:?}");
        let series_sum: f64 = series_sums.into_iter().sum();
        println!("series_sum = {series_sum}");
    } else {
        process_root.gather_into(&series_part_sum);
    }
}

