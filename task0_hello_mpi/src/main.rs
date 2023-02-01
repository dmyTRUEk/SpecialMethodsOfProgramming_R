//! MPI Hello World.
//!
//! Run the program:
//! ```
//! cargo b -r && mpirun --hostfile ../myhostfile -n 12 ./target/release/task0_hello_mpi
//! ```

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    println!(
        "Hello parallel world from process {} of {}!",
        world.rank(),
        world.size()
    );
}

