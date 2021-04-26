use libcm;
use std::env;

pub fn main() {
    let args: Vec<String> = env::args().collect();
    let argv = &args[1].parse::<u64>();

    match argv {
        Ok(n) => {
            println!("Is this num {:?} prime?", n);
            match libcm::miller_rabin(*n) {
                Some(_) => println!("yes it is prime"),
                None => println!("no it is not"),
            }
        }
        _ => println!("Error"),
    }
}
