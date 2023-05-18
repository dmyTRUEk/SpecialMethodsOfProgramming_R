//! Task 4: make complex datatypes exchange,
//! for example using matrices parts.

use mpi::traits::*;


const MATRIX_SIZE: usize = 5;

const MATRIX: [[u8; MATRIX_SIZE]; MATRIX_SIZE] = [
    [11, 12, 13, 14, 15],
    [21, 22, 23, 24, 25],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45],
    [51, 52, 53, 54, 55],
];

const SEND: Send = Send::Triangles;


#[allow(dead_code)]
enum Send {
    /// Main diagonal.
    Diagonal,

    /// Main diagonal (0), one above it (1), below main diagonal (-1), and so on.
    Diagonals,

    /// Triangle part of matrix: upper triangle (1), lower triangle (-1), and so on.
    Triangles,
}


fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    // assert!(size > 1, "need more than 1 processes, but {size} was given.");
    let rank = world.rank();

    match SEND {
        Send::Diagonal => {
            let mut diagonal_main = [0; MATRIX_SIZE];
            if rank == 0 {
                for i in 0..MATRIX_SIZE {
                    diagonal_main[i] = MATRIX[i][i];
                }
            }
            world.process_at_rank(0).broadcast_into(&mut diagonal_main);
            println!("rank {rank}: diagonal_main = {diagonal_main:?}");
        }
        Send::Diagonals => {
            if rank == 0 {
                for i in 0..(size as usize) {
                    let diagonal_index: isize = diagonal_index(i);
                    let diagonal: Vec<u8> = diagonal(diagonal_index);
                    println!("for rank {i}: diagonal_index = {diagonal_index} => diagonal = {diagonal:?}");
                    world.process_at_rank(i as i32).send(&diagonal);
                }
            }
            world.barrier();
            let diagonal = world.process_at_rank(0).receive_vec::<u8>().0;
            println!("rank {rank}: diagonal = {diagonal:?}");
        }
        Send::Triangles => {
            if rank == 0 {
                for i in 0..(size as usize) {
                    let diagonal_index: isize = diagonal_index(i+1);
                    let is_upper = diagonal_index > 0;
                    let triangle_packed: Vec<u8> = triangle_packer::pack(MATRIX, diagonal_index);
                    world.process_at_rank(i as i32).send::<Vec<u8>>(&triangle_packed);
                    world.process_at_rank(i as i32).send::<bool>(&is_upper);
                }
            }
            world.barrier();

            let triangle_packed = world.process_at_rank(0).receive_vec::<u8>().0;
            let is_upper = world.process_at_rank(0).receive::<bool>().0;

            let triangle_unpacked = triangle_packer::unpack::<MATRIX_SIZE>(triangle_packed, is_upper);
            println!("rank {rank}: triangle_unpacked = {}", matrix_to_string(triangle_unpacked, Some("--")));
        }
    }
}


mod triangle_packer {
    pub fn pack<const MATRIX_SIZE: usize>(
        matrix: [[u8; MATRIX_SIZE]; MATRIX_SIZE],
        diagonal_index: isize,
    ) -> Vec<u8> {
        assert!(diagonal_index.abs() as usize <= MATRIX_SIZE);
        assert_ne!(diagonal_index, 0);
        let small_matrix_size: usize = MATRIX_SIZE - diagonal_index.abs() as usize + 1;
        let packed_len: usize = small_matrix_size * (small_matrix_size - 1) / 2;
        let is_upper = diagonal_index > 0;
        let mut packed = Vec::<u8>::with_capacity(packed_len);
        for r in 0..MATRIX_SIZE {
            for c in 0..MATRIX_SIZE {
                if is_upper {
                    if c as isize > r as isize + diagonal_index.abs() - 1 {
                        packed.push(matrix[r][c]);
                    }
                } else {
                    if r as isize > c as isize + diagonal_index.abs() - 1 {
                        packed.push(matrix[r][c]);
                    }
                }
            }
        }
        assert_eq!(packed_len, packed.len());
        packed
    }

    pub fn unpack<const MATRIX_SIZE: usize>(packed: Vec<u8>, is_upper: bool) -> [[u8; MATRIX_SIZE]; MATRIX_SIZE] {
        // let small_matrix_size_2: usize = match packed.len() {
        //     1 => 2,
        //     3 => 3,
        //     6 => 4,
        //     10 => 5,
        //     15 => 6,
        //     21 => 7,
        //     28 => 8,
        //     36 => 9,
        //     45 => 10,
        //     55 => 11,
        //     _ => unimplemented!()
        // };
        let small_matrix_size: usize = (1 + ((1+8*packed.len()) as f64).sqrt().round() as usize) / 2;
        // assert_eq!(small_matrix_size_2, small_matrix_size);
        let packed_len_expected: usize = small_matrix_size * (small_matrix_size - 1) / 2;
        assert_eq!(packed_len_expected, packed.len());
        let mut diagonal_index: isize = (MATRIX_SIZE - small_matrix_size) as isize + 1;
        if !is_upper {
            diagonal_index = -diagonal_index;
        }
        // println!("packed.len() = {}, packed = {packed:?}, diagonal_index = {diagonal_index}", packed.len());
        let mut matrix = [[0_u8; MATRIX_SIZE]; MATRIX_SIZE];
        let mut i: usize = 0;
        for r in 0..MATRIX_SIZE {
            for c in 0..MATRIX_SIZE {
                if is_upper {
                    if c as isize > r as isize + diagonal_index.abs() - 1 {
                        matrix[r][c] = packed[i];
                        i += 1;
                    }
                } else {
                    if r as isize > c as isize + diagonal_index.abs() - 1 {
                        matrix[r][c] = packed[i];
                        i += 1;
                    }
                }
            }
        }
        matrix
    }
}


// // #[derive(Equivalence)]
// struct DiagonalsWrapper//<const I: isize>
// // where [(); diagonal_len(I)]:
// {
//     // diagonal: [u8; diagonal_len(I)]
//     diagonal: Vec<u8>
// }
fn diagonal_index(i: usize) -> isize {
    let sign = if i % 2 == 0 { -1 } else { 1 };
    let i = i as isize;
    sign * ((i + 1) / 2)
}

const fn diagonal_len(index: isize) -> usize {
    if index < MATRIX_SIZE as isize {
        MATRIX_SIZE - index.abs() as usize
    } else {
        panic!("`index` must be less than `MATRIX_SIZE`");
    }
}

fn diagonal(index: isize) -> Vec<u8> {
    let diagonal_len = diagonal_len(index);
    let mut diagonal: Vec<u8> = vec![0; diagonal_len];
    for j in 0..diagonal_len {
        //  0 -> 00 11 22 33 …
        //  1 -> 01 12 23 34 …
        // -1 -> 10 21 32 43 …
        //  2 -> 02 13 24 35 …
        // -2 -> 20 31 42 53 …
        diagonal[j] = MATRIX[(-index).max(0) as usize + j][index.max(0) as usize + j];
    }
    diagonal
}


fn matrix_to_string<const MATRIX_SIZE: usize>(
    matrix: [[u8; MATRIX_SIZE]; MATRIX_SIZE],
    replace_zeros_by: Option<&str>,
) -> String {
    if let Some(replace_zeros_by) = replace_zeros_by {
        assert!(replace_zeros_by.len() <= 2);
    }
    let mut s = "[\n".to_string();
    for row in matrix {
        s += &" ".repeat(4);
        for element in row {
            s += &match replace_zeros_by {
                Some(replace_zeros_by) if element == 0 => format!("{replace_zeros_by:<2}"),
                _ => format!("{element:<2}"),
            };
            s += " ";
        }
        s += "\n";
    }
    s += "]";
    s
}

