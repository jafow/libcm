use std::time::{Duration, Instant};
use is_prime;

fn main() {
    let mut primes: Vec<u64> = Vec::new();
    let start = Instant::now();
    for x in 3..100_000 {
        let s = format!("{}", x);
        if is_prime::is_prime(s.as_str()) {
            primes.push(x);
        }
    }
    println!("we got count: {:?} primes", primes.len());
    let end = start.elapsed().as_secs();
    println!("runtime {:?}", &end);
}

