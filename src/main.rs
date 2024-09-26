use std::{
    env::args,
    mem,
    path::PathBuf,
    process::{Command, Stdio},
    time::Duration,
};

use image::{GrayImage, Luma};
use ndarray::Array2;
use rand::Rng;
use rayon::iter::{ParallelBridge, ParallelIterator};
use tokio::time::sleep;

const MAX_ITER: usize = 35;
const PROB: f64 = 0.15;
const DELAY: u64 = 1000;
const SIZE: usize = 10;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = args();
    if args.len() < 4 {
        return Err(anyhow::anyhow!(
            "Usage: {} <path> <width> <height> [command]",
            args.next().unwrap()
        ));
    }
    let mut args = args.skip(1);
    let path: PathBuf = args.next().unwrap().into();
    let width: usize = args.next().unwrap().parse()?;
    let height: usize = args.next().unwrap().parse()?;
    let command = args.next();
    let mut cur: &mut Array2<bool> = &mut Array2::from_elem((width, height), false);
    let mut next: &mut Array2<bool> = &mut Array2::from_elem((width, height), false);
    let mut iter = MAX_ITER;
    loop {
        if iter == MAX_ITER {
            cur.iter_mut().par_bridge().for_each(|val| {
                let mut rng = rand::thread_rng();
                *val = rng.gen_bool(PROB);
            });
            iter = 0;
        }
        array2_to_image(&cur).save(&path)?;
        if let Some(ref command) = command {
            Command::new(&command).stdout(Stdio::null()).spawn()?;
        }
        next_generation(cur, next);
        mem::swap(&mut cur, &mut next);
        iter += 1;
        sleep(Duration::from_millis(DELAY)).await;
    }
}

fn next_state(grid: &Array2<bool>, x: usize, y: usize) -> bool {
    let mut live_neighbors = 0;
    for i in -1..=1 {
        for j in -1..=1 {
            if i == 0 && j == 0 {
                continue;
            }
            let nx = (x as isize + i).rem_euclid(grid.ncols() as isize) as usize;
            let ny = (y as isize + j).rem_euclid(grid.ncols() as isize) as usize;
            if grid[(ny, nx)] {
                live_neighbors += 1;
            }
        }
    }
    match (grid[(y, x)], live_neighbors) {
        (true, 2) | (_, 3) => true,
        _ => false,
    }
}

fn next_generation(cur: &Array2<bool>, next: &mut Array2<bool>) {
    next.indexed_iter_mut()
        .par_bridge()
        .for_each(|((y, x), next_val)| {
            *next_val = next_state(cur, x, y);
        });
}

fn array2_to_image(grid: &Array2<bool>) -> GrayImage {
    let height = grid.ncols() * SIZE;
    let width = grid.nrows() * SIZE;

    let mut img = GrayImage::new(width as u32, height as u32);

    for ((x, y), &value) in grid.indexed_iter() {
        let pixel_value = Luma([if value { 64 } else { 0 }]);
        for i in 0..SIZE {
            for j in 0..SIZE {
                img.put_pixel((x * SIZE + j) as u32, (y * SIZE + i) as u32, pixel_value);
            }
        }
    }

    img
}
