//! Calc series sum using MPI.
//!
//! Run test and program:
//! ```
//! cargo t && cargo b -r && mpirun --hostfile ../myhostfile -n 12 ./target/release/task1_calc_series
//! ```

use mpi::{traits::*, Rank};


/// Terms in taylor series.
const N: i32 = 30;


fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    assert!(size > 1, "need more than one processes.");
    let rank = world.rank();

    let x = 3.;

    if rank != 0 {
        let series_part_by_chunks_split = calc_series_part_by_chunks_split(x, rank, size);
        let series_part_by_braid_split  = calc_series_part_by_braid_split(x, rank, size);
        let msg: [f64; 2] = [series_part_by_chunks_split, series_part_by_braid_split];
        world.process_at_rank(0).send(&msg);
        return;
    }
    // only main continues:

    let mut series_parts_by_chunks_split: Vec<f64> = vec![];
    let mut series_parts_by_braid_split : Vec<f64> = vec![];
    for _ in 1..size {
        let (msg, _status) = world.any_process().receive_vec::<f64>();
        assert_eq!(2, msg.len());
        series_parts_by_chunks_split.push(msg[0]);
        series_parts_by_braid_split .push(msg[1]);
    }
    assert_eq!(size-1, series_parts_by_chunks_split.len() as i32);
    assert_eq!(size-1, series_parts_by_braid_split .len() as i32);

    // series_parts_by_chunks_split.sort_by(|a, b| b.total_cmp(a));
    // series_parts_by_braid_split .sort_by(|a, b| b.total_cmp(a));
    // println!("series_parts_by_chunks_split: {series_parts_by_chunks_split:#?}");
    // println!("series_parts_by_braid_split : {series_parts_by_braid_split :#?}");

    let series_sum_by_chunks_split: f64 = series_parts_by_chunks_split.iter().sum();
    let series_sum_by_braid_split : f64 = series_parts_by_braid_split .iter().sum();

    println!("results: exp({x}) = ");
    println!("    expected        : {}", x.exp());
    println!("    by braid  split : {series_sum_by_braid_split}");
    println!("    by chunks split : {series_sum_by_chunks_split}");
    // results: exp(3) =
    //     expected        : 20.085536923187668
    //     by braid  split : 20.085536923187668
    //     by chunks split : 20.085537612432425
}


/// Calculate series' part by "chunks" split.
///
/// exp(x) = 1 + x + x^2/2! + x^3/3! + x^4/4! + … x^N/N!
///
/// if size = 2:
///     rank = 0: 1 + x + x^2/2! + … + x^(N/2)/(N/2)!
///     rank = 1: x^(N/2+1)/(N/2+1)! + … + x^N/N!
///
/// if size = 3:
///     rank = 0: 1 + x + x^2/2! + … + x^(N/3)/(N/3)!
///     rank = 1: x^(N/3+1)/(N/3+1)! + … + x^(N/3*2)/(N/3*2)!
///     rank = 2: x^(N/3*2+1)/(N/3*2+1)! + … + x^N/N!
///
#[allow(unused)]
fn calc_series_part_by_chunks_split(x: f64, rank: Rank, size: Rank) -> f64 {
    let rank: i32 = rank - 1; // bc first(0) is just collecting results
    let size: i32 = size - 1; // bc first(0) is just collecting results

    let i_start = N*rank/size;
    let i_end   = if rank == size-1 { N } else { N*(rank+1)/size-1 };

    let mut index_iter = i_start..=i_end;
    let i0 = index_iter.next().unwrap();

    let mut term = x.powi(i0) / fact(i0) as f64;
    let mut res = term;
    for i in index_iter {
        term *= x / i as f64;
        res += term;
    }
    res
}


/// Calculate series' part by "braid" split.
///
/// exp(x) = 1 + x + x^2/2! + x^3/3! + x^4/4! + … x^N/N!
///
/// if size = 2:
///     rank = 0: 1 + x^2/2! + x^4/4! + …
///     rank = 1: x + x^3/3! + x^5/5! + …
///
/// if size = 3:
///     rank = 0: 1      + x^3/3! + x^6/6! + …
///     rank = 1: x      + x^4/4! + x^7/7! + …
///     rank = 2: x^2/2! + x^5/5! + x^8/8! + …
///
#[allow(unused)]
fn calc_series_part_by_braid_split(x: f64, rank: Rank, size: Rank) -> f64 {
    let rank: i32 = rank - 1; // bc first(0) is just collecting results
    let size: i32 = size - 1; // bc first(0) is just collecting results

    let i_start = rank;
    let i_end   = N;

    let mut index_iter = (i_start..=i_end).step_by(size as usize);
    let i0 = index_iter.next().unwrap();

    let mut term: f64 = x.powi(i0) / fact(i0) as f64;
    let mut res = term;
    for i in index_iter {
        term *= x.powi(size) / fact_part(i-size+1, i) as f64;
        res += term;
    }
    res
}


/// Calculate series' part by `split_series_type`.
#[allow(unused)]
fn calc_series_part(
    x: f64,
    rank: Rank,
    size: Rank,
    split_series_type: SplitSeriesType
) -> f64 {
    let rank: i32 = rank - 1; // bc first(0) is just collecting results
    let size: i32 = size - 1; // bc first(0) is just collecting results

    let i_start = match split_series_type {
        SplitSeriesType::Chunks => N*rank/size,
        SplitSeriesType::Braid  => rank,
    };
    let i_end = match split_series_type {
        SplitSeriesType::Chunks => if rank == size-1 { N } else { N*(rank+1)/size-1 },
        SplitSeriesType::Braid  => N,
    };

    let mut index_iter = match split_series_type {
        SplitSeriesType::Chunks => (i_start..=i_end)                       .collect::<Vec<_>>().into_iter(),
        SplitSeriesType::Braid  => (i_start..=i_end).step_by(size as usize).collect::<Vec<_>>().into_iter(),
    };

    let i0 = index_iter.next().unwrap();

    let mut term = x.powi(i0) / fact(i0) as f64;
    let mut res = term;
    for i in index_iter {
        term *= match split_series_type {
            SplitSeriesType::Chunks => x / i as f64,
            SplitSeriesType::Braid  => x.powi(size) / fact_part(i-size+1, i) as f64,
        };
        res += term;
    }
    res
}


#[allow(unused)]
enum SplitSeriesType {
    Chunks,
    Braid,
}





/// Factorial on `n` = `n!`.
fn fact(n: i32) -> u64 {
    assert!(n >= 0);
    (2..=n as u64).product()
}

/// Factorial partial = `start * (start+1) * … * (end-1) * end` = `end! / start!`.
fn fact_part(start: i32, end: i32) -> u64 {
    (start.max(2)..=end).map(|n| n as u64).product()
}





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_() {
        assert_eq!(1, fact(0));
        assert_eq!(1, fact(1));
        assert_eq!(2, fact(2));
        assert_eq!(6, fact(3));
        assert_eq!(24, fact(4));
        assert_eq!(120, fact(5));
        assert_eq!(720, fact(6));
        assert_eq!(3628800, fact(10));
    }

    #[test]
    fn fact_part_() {
        assert_eq!(1, fact_part(5, -1));
        assert_eq!(1, fact_part(5, 0));
        assert_eq!(1, fact_part(5, 1));
        assert_eq!(1, fact_part(5, 2));
        assert_eq!(1, fact_part(5, 3));
        assert_eq!(1, fact_part(5, 4));
        assert_eq!(5, fact_part(5, 5));
        assert_eq!(5*6, fact_part(5, 6));
        assert_eq!(5*6*7, fact_part(5, 7));
        assert_eq!(5*6*7*8, fact_part(5, 8));
        assert_eq!(100*101*102*103*104*105, fact_part(100, 105));
    }
}

