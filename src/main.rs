use std::{
    mem,
    process::{Command, Stdio},
    time::Duration,
};

use clap::Parser;
use image::{GrayImage, Luma};
use ndarray::Array2;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::iter::{ParallelBridge, ParallelIterator};
use tokio::time::sleep;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path of the image that will be generated
    #[arg(short, long, default_value_t = String::from("output.png"))]
    output: String,
    /// Width of the game of life matrix
    #[arg(short, long)]
    width: usize,
    /// Height of the game of life matrix
    #[arg(short, long)]
    height: usize,
    /// Cell size in pixels
    #[arg(short, long, value_parser=clap::value_parser!(u32).range(1..), default_value_t = 10)]
    size: u32,
    /// Command which will be executed after generating each image
    #[arg(short, long)]
    command: Option<String>,
    /// Number of generations before resetting the grid to a random state
    #[arg(short, long, default_value_t = 35)]
    max_iter: usize,
    /// Probability of a cell being alive when randomly filling the grid
    #[arg(short, long, default_value_t = 0.15)]
    fill: f64,
    /// Delay in ms between each image generation
    #[arg(short, long, value_parser=clap::value_parser!(u64).range(1..), default_value_t = 1000)]
    delay: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut cur: &mut Array2<bool> = &mut Array2::from_elem((args.width, args.height), false);
    let mut next: &mut Array2<bool> = &mut Array2::from_elem((args.width, args.height), false);
    let mut iter = args.max_iter;
    loop {
        if iter == args.max_iter {
            cur.iter_mut().par_bridge().for_each_init(
                || Xoshiro256PlusPlus::from_entropy(),
                |rng, val| {
                    *val = rng.gen_bool(args.fill);
                },
            );
            iter = 0;
        }
        array2_to_image(&cur, args.size).save(&args.output)?;
        if let Some(ref command) = args.command {
            Command::new(&command).stdout(Stdio::null()).spawn()?;
        }
        next_generation(cur, next);
        mem::swap(&mut cur, &mut next);
        iter += 1;
        sleep(Duration::from_millis(args.delay)).await;
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

fn array2_to_image(grid: &Array2<bool>, size: u32) -> GrayImage {
    let height: u32 = grid.ncols() as u32 * size;
    let width: u32 = grid.nrows() as u32 * size;

    let mut img = GrayImage::new(width, height);

    for ((x, y), &value) in grid.indexed_iter() {
        let pixel_value = Luma([if value { 64 } else { 0 }]);
        for i in 0..size {
            for j in 0..size {
                img.put_pixel(x as u32 * size + j, y as u32 * size + i, pixel_value);
            }
        }
    }

    img
}
