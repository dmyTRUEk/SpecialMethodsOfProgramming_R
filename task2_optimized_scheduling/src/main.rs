//! Task 2: impl algorithm for calculating a function
//! with variable evaluation time on different input.

use std::{
    time::Duration,
    thread::sleep,
};

use mpi::{traits::*, Rank};
use rand::{thread_rng, Rng};


const N: usize = 100;

// const USE_OPTIMIZED_SCHEDULING: bool = false;
const USE_OPTIMIZED_SCHEDULING: bool = true;


#[allow(unused_doc_comments)]
fn f(_x: f64, rank: Rank, _size: Rank) -> f64 {
    /// Generating random delay from any of these distributions
    /// doesn't show major advantage of optimized scheduling over
    /// non-optimized one.
    // let delay = RNG::gen_from_linear_distribution(0., 1000.);
    // let delay = RNG::gen_from_linear_powered_distribution(0., 5., 5);
    // let delay = RNG::gen_from_linear_powered_distribution(0., 1.2, 40);
    // let delay = RNG::gen_from_linear_powered_distribution(0., 1.1, 100);
    // let delay = RNG::gen_from_linear_multiplied_distribution(0., 10., 4);
    // let delay = RNG::gen_from_poisson_distribution(100.);

    /// Only in case of these dependencies of the delay of input parameters
    /// optimized scheduling shows major advantage. In the real world it means
    /// that some computing unit(s) are much slower than other.
    let delay = if rank == 1 { 1000 } else { 100 };
    // let delay = if rank <= 3 { 1000 } else { 100 };
    // let delay = if rank % 3 == 0 { 1000 } else { 100 };
    // let delay = if rank < size/2 { 1000 } else { 100 };

    /// Sleep for `delay` milliseconds.
    sleep(Duration::from_millis(delay as u64));
    // sleep(Duration::from_secs(1));

    // return 0
    0.
}


fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    assert!(size > 1, "need more than one process.");
    let rank = world.rank();

    if rank == 0 {
        let mut points = range(0., (N-1) as f64, 1.0);
        let mut results = Vec::<f64>::with_capacity(points.len());
        if USE_OPTIMIZED_SCHEDULING {
            for i in 1..size {
                let point = points.pop();
                if point.is_none() { break }
                let point = point.unwrap();
                world.process_at_rank(i).send(&point);
            }
            print("results.len() =");
            while !points.is_empty() {
                // find free process:
                let mut free_process = None;
                for i in 1..size {
                    let status = world.process_at_rank(i).immediate_probe();
                    if status.is_some() {
                        free_process = Some(i);
                        break;
                    }
                }
                if free_process.is_none() { continue }
                let free_process = free_process.unwrap();

                let (msg, _status) = world.process_at_rank(free_process).receive::<f64>();
                results.push(msg);
                print(format!(" {}", results.len()));

                let point = points.pop().unwrap();
                world.process_at_rank(free_process).send(&point);
            }
            for i in 1..size {
                let (msg, _status) = world.any_process().receive::<f64>();
                results.push(msg);
                print(format!(" {}", results.len()));
                world.process_at_rank(i).send(&f64::NAN);
            }
        }
        else {
            let mut i = 1;
            for point in points.iter() {
                world.process_at_rank(i).send(point);
                i += 1;
                if i >= size {
                    i = 1;
                }
            }
            for i in 1..size {
                world.process_at_rank(i).send(&f64::NAN);
            }
            print("results.len() =");
            for _ in 0..points.len() {
                let (msg, _status) = world.any_process().receive::<f64>();
                results.push(msg);
                print(format!(" {}", results.len()));
            }
        }
        println!();
        println!("results:");
        println!("{:?}", results);
    }
    else {
        loop {
            let (point, _status) = world.process_at_rank(0).receive::<f64>();
            if point.is_nan() { break }
            let res = f(point, rank, size);
            world.process_at_rank(0).send(&res);
        }
    }
}


fn range(start: f64, end: f64, step_by: f64) -> Vec<f64> {
    let range = end - start;
    let range = range - (range % step_by);
    if range < 0. { return vec![] }
    if range == 0. { return vec![start] }
    let vec_len: usize = (range / step_by) as usize;
    let mut res = Vec::<f64>::with_capacity(vec_len);
    // for i in 0..vec_len {
    //     let t = (i as f64) / (vec_len as f64);
    //     let x = start + range * t;
    //     res.push(x);
    // }
    // alternative implementation:
    let mut x = start;
    let mut i = 0;
    while x <= end {
        let t = (i as f64) / (vec_len as f64);
        x = start + range * t;
        res.push(x);
        i += 1;
        x += step_by;
    }
    res
}


struct RNG {}
#[allow(unused)]
impl RNG {
    pub fn gen_from_linear_distribution(min: f64, max: f64) -> f64 {
        // create `ThreadRNG`: Random Number Generator from `rand` library.
        let mut rng = thread_rng();
        rng.gen_range(min ..= max)
    }

    pub fn gen_from_linear_powered_distribution(min: f64, max: f64, pow: i32) -> f64 {
        let x = RNG::gen_from_linear_distribution(min, max);
        x.powi(pow)
    }

    pub fn gen_from_linear_multiplied_distribution(min: f64, max: f64, n: i32) -> f64 {
        let mut res: f64 = 1.;
        for _ in 0..n {
            res *= RNG::gen_from_linear_distribution(min, max);
        }
        res
    }

    pub fn gen_from_poisson_distribution(lambda: f64) -> u8 {
        const CUT_OFF: usize = 1000;
        let x = RNG::gen_from_linear_distribution(0., 1.);
        let mut term: f64 = exp(-lambda); // k = 0
        if term == 0. { unreachable!() }
        let mut sum: f64 = term;
        for k in 1..CUT_OFF {
            term *= lambda / (k as f64);
            sum += term;
            if sum >= x {
                return k as u8;
            }
        }
        unreachable!()
    }
}


pub const fn fact(n: u8) -> u64 {
    assert!(n <= 20);
    let mut r = 1;
    let mut i = 2;
    while i <= n {
        r *= i as u64;
        i += 1;
    }
    r
}


fn exp(x: f64) -> f64 {
    x.exp()
}


fn print(msg: impl ToString) {
    use std::io::{stdout, Write};
    print!("{}", msg.to_string());
    stdout().flush().unwrap();
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_() {
        assert_eq!(Vec::<f64>::new(), range(1., 0., 1.));
        assert_eq!(vec![0.], range(0., 0., 1.));
        assert_eq!(vec![0.], range(0., 0.1, 1.));
        assert_eq!(vec![0., 1.], range(0., 1., 1.));
        assert_eq!(vec![0., 1.], range(0., 1.1, 1.));
        assert_eq!(vec![0., 1., 2.], range(0., 2., 1.));
        assert_eq!(vec![0., 1., 2.], range(0., 2.1, 1.));
        assert_eq!(vec![0., 1., 2., 3.], range(0., 3., 1.));
        assert_eq!(vec![0., 1., 2., 3.], range(0., 3.1, 1.));
        assert_eq!(vec![0., 2., 4.], range(0., 5., 2.));
        assert_eq!(vec![0., 2., 4.], range(0., 5.3, 2.));
    }
}

