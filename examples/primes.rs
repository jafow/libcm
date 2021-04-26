use std::time::{Duration, Instant};
use libcm;

fn main() {
    let mut primes: Vec<u64> = Vec::new();
    let start = Instant::now();
    for x in 3..100_000 {
        if let Some(x) = libcm::miller_rabin(x as u64) {
            primes.push(x);
        }
    }
    println!("we got count: {:?} primes", primes.len());
    let end = start.elapsed().as_secs();
    println!("runtime {:?}", &end);
}
