//! Task 2.5: impl algorithm for fast matrix multiplication by Fox Algorithm.

#![feature(generic_const_exprs)]

use std::{
    env::args as cli_args,
    ops::{Index, IndexMut},
};

use mpi::traits::*;


#[allow(non_camel_case_types)]
type float = f64;


const N: usize = 2;


fn main() {
    // Set up universe and world:
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    assert_eq!(size, (N as i32).pow(2)+1, "Required N^2+1 processes.");
    let rank = world.rank();

    if rank == 0 {
        // Get matrices from CLI args:
        let cli_args = cli_args().collect::<Vec<_>>();

        // Parse strings into matrices.
        let matrix_a = Matrix::<N>::from_rows_str(&cli_args[1]);
        let matrix_b = Matrix::<N>::from_rows_str(&cli_args[2]);

        // Send rows and columns to corresponding processes:
        for r in 0..N {
            for c in 0..N {
                let row    = matrix_a.get_row(r);
                let column = matrix_b.get_column(c);
                let data = WrappedArrays::new(row, column);
                world.process_at_rank((r*N+c+1) as i32).send(&data);
            }
        }

        // Receive results:
        let mut matrix_c = Matrix::<N>::zeros();
        for r in 0..N {
            for c in 0..N {
                let (c_ij, _status) = world.process_at_rank((r*N+c+1) as i32).receive::<float>();
                matrix_c[(r, c)] = c_ij;
            }
        }

        // Print results:
        println!("Result:");
        for r in 0..N {
            for c in 0..N {
                print!("{}\t", matrix_c[(r, c)]);
            }
            println!();
        }
    }
    else {
        // Get row and column to multiply:
        let (data, _status) = world.process_at_rank(0).receive::<WrappedArrays>();
        let (row, column) = data.unwrap();
        let mut res: float = 0.;

        // Multiply row by column:
        for i in 0..N {
            res += row[i] * column[i];
        }

        // Send result back:
        world.process_at_rank(0).send(&res);
    }
}


struct Matrix<const N: usize> {
    elements: [[float; N]; N],
}
impl<const N: usize> Matrix<N> {
    fn from_elements(elements: [[float; N]; N]) -> Self {
        Self { elements }
    }
    fn from_element(element: float) -> Self {
        Self::from_elements([[element; N]; N])
    }
    fn zeros() -> Self {
        Self::from_element(0.)
    }
    fn from_rows(array: [float; N*N]) -> Self {
        let mut self_ = Self { elements: [[0.; N]; N] };
        for r in 0..N {
            for c in 0..N {
                self_.elements[r][c] = array[r*N+c];
            }
        }
        self_
    }
    fn from_rows_str(str: &str) -> Self
    where [(); N*N]:
    {
        let elements: Vec<&str> = str.split(' ').collect();
        assert_eq!(N.pow(2), elements.len());
        let elements: Vec<float> = elements.into_iter()
            .map(|el| el.parse::<float>().unwrap())
            .collect();
        let elements: [float; N*N] = elements.try_into().unwrap();
        Self::from_rows(elements)
    }
    fn get_row(&self, index: usize) -> [float; N] {
        self.elements[index]
    }
    fn get_column(&self, index: usize) -> [float; N] {
        let mut column: [float; N] = [0.; N];
        for r in 0..N {
            column[r] = self.elements[r][index];
        }
        column
    }
}
impl<const N: usize> Index<(usize, usize)> for Matrix<N> {
    type Output = float;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.elements[i][j]
    }
}
impl<const N: usize> IndexMut<(usize, usize)> for Matrix<N> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.elements[i][j]
    }
}


#[derive(Equivalence)]
struct WrappedArrays {
    row   : [float; N],
    column: [float; N]
}
impl WrappedArrays {
    fn new(array_1: [float; N], array_2: [float; N]) -> Self {
        Self { row: array_1, column: array_2 }
    }
    fn unwrap(self) -> ([float; N], [float; N]) {
        (self.row, self.column)
    }
}



#[cfg(test)]
mod matrix {
    mod index {
        use super::super::Matrix;
        #[test]
        fn _2x2() {
            let m: Matrix<2> = Matrix {
                elements: [
                    [1., 2.],
                    [3., 4.],
                ]
            };
            assert_eq!(1., m[(0, 0)]);
            assert_eq!(2., m[(0, 1)]);
            assert_eq!(3., m[(1, 0)]);
            assert_eq!(4., m[(1, 1)]);
        }
    }
    mod index_mut {
        mod _2x2 {
            use super::super::super::Matrix;
            #[test]
            fn _0_0() {
                let mut m: Matrix<2> = Matrix {
                    elements: [
                        [1., 2.],
                        [3., 4.],
                    ]
                };
                m[(0, 0)] = 42.;
                assert_eq!(42., m[(0, 0)]);
                assert_eq!(2., m[(0, 1)]);
                assert_eq!(3., m[(1, 0)]);
                assert_eq!(4., m[(1, 1)]);
            }
            #[test]
            fn _0_1() {
                let mut m: Matrix<2> = Matrix {
                    elements: [
                        [1., 2.],
                        [3., 4.],
                    ]
                };
                m[(0, 1)] = 42.;
                assert_eq!(1., m[(0, 0)]);
                assert_eq!(42., m[(0, 1)]);
                assert_eq!(3., m[(1, 0)]);
                assert_eq!(4., m[(1, 1)]);
            }
            #[test]
            fn _1_0() {
                let mut m: Matrix<2> = Matrix {
                    elements: [
                        [1., 2.],
                        [3., 4.],
                    ]
                };
                m[(1, 0)] = 42.;
                assert_eq!(1., m[(0, 0)]);
                assert_eq!(2., m[(0, 1)]);
                assert_eq!(42., m[(1, 0)]);
                assert_eq!(4., m[(1, 1)]);
            }
            #[test]
            fn _1_1() {
                let mut m: Matrix<2> = Matrix {
                    elements: [
                        [1., 2.],
                        [3., 4.],
                    ]
                };
                m[(1, 1)] = 42.;
                assert_eq!(1., m[(0, 0)]);
                assert_eq!(2., m[(0, 1)]);
                assert_eq!(3., m[(1, 0)]);
                assert_eq!(42., m[(1, 1)]);
            }
        }
    }
    mod from_elements {
        mod _2x2 {
            use super::super::super::Matrix;
            #[test]
            fn zeros() {
                let m: Matrix<2> = Matrix {
                    elements: [
                        [0., 0.],
                        [0., 0.],
                    ]
                };
                assert_eq!(0., m[(0, 0)]);
                assert_eq!(0., m[(0, 1)]);
                assert_eq!(0., m[(1, 0)]);
                assert_eq!(0., m[(1, 1)]);
            }
            #[test]
            fn consts() {
                let c = 42.137;
                let m: Matrix<2> = Matrix {
                    elements: [
                        [c, c],
                        [c, c],
                    ]
                };
                assert_eq!(c, m[(0, 0)]);
                assert_eq!(c, m[(0, 1)]);
                assert_eq!(c, m[(1, 0)]);
                assert_eq!(c, m[(1, 1)]);
            }
            #[test]
            fn identity() {
                let m: Matrix<2> = Matrix {
                    elements: [
                        [1., 0.],
                        [0., 1.],
                    ]
                };
                assert_eq!(1., m[(0, 0)]);
                assert_eq!(0., m[(0, 1)]);
                assert_eq!(0., m[(1, 0)]);
                assert_eq!(1., m[(1, 1)]);
            }
            #[test]
            fn custom() {
                let m: Matrix<2> = Matrix {
                    elements: [
                        [1., 2.],
                        [3., 4.],
                    ]
                };
                assert_eq!(1., m[(0, 0)]);
                assert_eq!(2., m[(0, 1)]);
                assert_eq!(3., m[(1, 0)]);
                assert_eq!(4., m[(1, 1)]);
            }
        }
    }
    mod from_rows {
        mod _2x2 {
            use super::super::super::Matrix;
            #[test]
            fn zeros() {
                let m: Matrix<2> = Matrix::from_rows([0., 0., 0., 0.]);
                assert_eq!(0., m[(0, 0)]);
                assert_eq!(0., m[(0, 1)]);
                assert_eq!(0., m[(1, 0)]);
                assert_eq!(0., m[(1, 1)]);
            }
            #[test]
            fn consts() {
                let c = 42.137;
                let m: Matrix<2> = Matrix::from_rows([c, c, c, c]);
                assert_eq!(c, m[(0, 0)]);
                assert_eq!(c, m[(0, 1)]);
                assert_eq!(c, m[(1, 0)]);
                assert_eq!(c, m[(1, 1)]);
            }
            #[test]
            fn identity() {
                let m: Matrix<2> = Matrix::from_rows([1., 0., 0., 1.]);
                assert_eq!(1., m[(0, 0)]);
                assert_eq!(0., m[(0, 1)]);
                assert_eq!(0., m[(1, 0)]);
                assert_eq!(1., m[(1, 1)]);
            }
            #[test]
            fn custom() {
                let m: Matrix<2> = Matrix::from_rows([1., 2., 3., 4.]);
                assert_eq!(1., m[(0, 0)]);
                assert_eq!(2., m[(0, 1)]);
                assert_eq!(3., m[(1, 0)]);
                assert_eq!(4., m[(1, 1)]);
            }
        }
    }
    mod from_rows_str {
        mod _2x2 {
            use super::super::super::Matrix;
            #[test]
            fn zeros() {
                let m: Matrix<2> = Matrix::from_rows_str("0. 0. 0. 0.");
                assert_eq!(0., m[(0, 0)]);
                assert_eq!(0., m[(0, 1)]);
                assert_eq!(0., m[(1, 0)]);
                assert_eq!(0., m[(1, 1)]);
            }
            #[test]
            fn consts() {
                let c = 42.137;
                let m: Matrix<2> = Matrix::from_rows_str(&format!("{c} {c} {c} {c}"));
                assert_eq!(c, m[(0, 0)]);
                assert_eq!(c, m[(0, 1)]);
                assert_eq!(c, m[(1, 0)]);
                assert_eq!(c, m[(1, 1)]);
            }
            #[test]
            fn identity() {
                let m: Matrix<2> = Matrix::from_rows_str("1. 0. 0. 1.");
                assert_eq!(1., m[(0, 0)]);
                assert_eq!(0., m[(0, 1)]);
                assert_eq!(0., m[(1, 0)]);
                assert_eq!(1., m[(1, 1)]);
            }
            #[test]
            fn custom() {
                let m: Matrix<2> = Matrix::from_rows_str("1. 2. 3. 4.");
                assert_eq!(1., m[(0, 0)]);
                assert_eq!(2., m[(0, 1)]);
                assert_eq!(3., m[(1, 0)]);
                assert_eq!(4., m[(1, 1)]);
            }
        }
    }
    mod get_row {
        mod _2x2 {
            use super::super::super::Matrix;
            #[test]
            fn _0() {
                let m: Matrix<2> = Matrix {
                    elements: [
                        [1., 2.],
                        [3., 4.],
                    ]
                };
                assert_eq!([1., 2.], m.get_row(0));
            }
            #[test]
            fn _1() {
                let m: Matrix<2> = Matrix {
                    elements: [
                        [1., 2.],
                        [3., 4.],
                    ]
                };
                assert_eq!([3., 4.], m.get_row(1));
            }
        }
    }
    mod get_column {
        mod _2x2 {
            use super::super::super::Matrix;
            #[test]
            fn _0() {
                let m: Matrix<2> = Matrix {
                    elements: [
                        [1., 2.],
                        [3., 4.],
                    ]
                };
                assert_eq!([1., 3.], m.get_column(0));
            }
            #[test]
            fn _1() {
                let m: Matrix<2> = Matrix {
                    elements: [
                        [1., 2.],
                        [3., 4.],
                    ]
                };
                assert_eq!([2., 4.], m.get_column(1));
            }
        }
    }
}

