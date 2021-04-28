use libcm;
use std::time::Instant;
use rayon::prelude::*;

fn main() {
    let start = Instant::now();
    let prime_count = count_primes(100_000);
    let end = start.elapsed().as_secs();
    println!("we got count: {:?} primes", prime_count);
    println!("runtime {:?}", &end);
}

fn count_primes (limit: u64) -> u64 {
    (3..limit).into_par_iter()
    .filter(|i| match libcm::miller_rabin(*i) {
        Some(_) => true,
        None => false
    })
    .count() as u64
}
