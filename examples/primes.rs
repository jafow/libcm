use libcm;
use std::time::Instant;
use rayon::prelude::*;

fn main() {
    let start = Instant::now();
    let prime_count = count_primes(100_000);
    let end = start.elapsed().as_secs_f64();
    println!("we got count: {:?} primes", prime_count);
    println!("runtime {:?}", &end);
}

fn count_primes (limit: u64) -> u64 {
    (3..limit).into_par_iter()
    .filter(|i| libcm::miller_rabin(*i).is_some())
    .count() as u64
}
