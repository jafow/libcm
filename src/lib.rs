use std::collections::HashSet;
use std::convert::TryInto;
use num_bigint::BigUint;
use num_traits::{One, CheckedDiv, ToPrimitive};
use std::ops::{Div, MulAssign, Rem};

/// CrtPredicate type
pub struct CrtPredicate {
    residue: u32,
    dividend: u32,
}

/// FastExp type returned from binary exponentiation
#[derive(Debug, PartialEq)]
struct FastExp<'a> {
    exp: &'a BigUint,
    pub total: BigUint,
    residue: BigUint,
}

#[derive(Debug, PartialEq)]
struct Factor2n<'a> {
    e: u64,
    k: &'a BigUint,
}

impl <'a>Factor2n<'a> {
    fn xp(&self) -> BigUint {
        // 2^e * k
        let b = BigUint::from(2u32);
        b.pow((self.e).try_into().unwrap()) * self.k
    }
}

// the max number of times we will check that a number is a strong liar or a strong witness
// to a number's primality
const MAX_WITNESS_CHECK: u64 = 4;

// sieve
pub fn sieve(target: u32) -> Vec<u32> {
    if target < 2 {
        return vec![1];
    }
    let mut res: Vec<u32> = (0..target).collect();
    let mut i = 2_usize;

    while i < res.len() {
        let mut j = i.pow(2);

        while j < res.len() {
            if res[j] != 0_u32 {
                res[j] = 0_u32;
            }
            j += i
        }
        i += 1;
    }
    // cut out 0 and 1
    res.into_iter()
        .filter(|x| *x >= 1_u32)
        .collect::<Vec<u32>>()
}

// determine the prime factorization of p
pub fn prime_factor(p: u64) -> Vec<u64> {
    let mut pb = p;
    let mut factor = 2_u64;
    let mut res: Vec<u64> = Vec::new();

    while pb != 1_u64 {
        if pb % factor == 0 {
            res.push(factor);
            pb /= factor;
        } else {
            factor += 1;
        }
    }
    res
}

fn coprimes(n: u32) -> Vec<u32> {
    // return a list of ints coprime to N
    let mut res: Vec<u32> = Vec::new();

    for x in 0..n {
        if gcd(n, x as u32) == 1 {
            // n and x are co prime
            res.push(x as u32);
        }
    }
    res
}

pub fn is_congruent(a: u32, b: u32, m: u32) -> bool {
    // are A and B congruent mod m?
    let is_congruent: bool = a % m == b % m;
    let lo: u32 = std::cmp::min(a, b);
    let hi: u32 = std::cmp::max(a, b);
    let n: bool = ((hi - lo) % m) == 0;
    is_congruent && n
}

pub fn totient(n: u32) -> u32 {
    coprimes(n).len() as u32
}

pub fn is_multiplicative_inverse_mod(modulus: i64, a: i64, b: i64) -> bool {
    // returns bool for whether b is a m
    ((a * b) - 1) % modulus == 0
}

pub fn gcd(a: u32, b: u32) -> u32 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    if a == b {
        return a;
    }
    let m = std::cmp::max(a, b);
    let min = std::cmp::min(a, b);
    gcd(m % min, min)
}

// helper printer formatter
pub fn printp(problem: &str) {
    println!(" =================== {:?} ============ ", problem);
}

pub fn square_and_multiply(base: i32, exp: i32) -> i32 {
    let mut res = base;
    let mut e = exp;

    while e > 0 {
        if e % 2 == 1 {
            e -= 1;
            res *= base;
            dbg!(&res);
        }
        e /= 2;
        res *= base.pow(2);
    }
    res * base
}

pub fn mod_exponent(base: i64, exp: i64, m: i64) -> i64 {
    let mut _exp: i64 = exp;
    let mut res: i64 = 1_i64;
    let mut _b: i64 = base % m;

    while _exp > 0 {
        if _exp % 2 == 1 {
            // odd
            res = (res * _b) % m;
        }
        _exp >>= 1;
        _b = (_b * _b) % m;
    }
    res
}

fn factor_2n(n: &BigUint, e: u32) -> u32 {
    if n.rem(&2u32) == One::one() {
        return e;
    }
    let reduced: BigUint = n.div(&2u8);
    factor_2n(&reduced, e + 1u32)
}

#[test]
fn factor2n_test() {
    let b = BigUint::new(vec![83u32]);
    assert_eq!(factor_2n(&b, 0u32), 0u32);

    let b = BigUint::new(vec![82u32]);
    assert_eq!(factor_2n(&b, 0u32), 1u32);

    let b = BigUint::new(vec![4u32]);
    assert_eq!(factor_2n(&b, 0u32), 2u32);

    let b = BigUint::new(vec![176u32]);
    assert_eq!(factor_2n(&b, 0u32), 4u32);

    let b = BigUint::new(vec![176u32]);
    let exp = factor_2n(&b, 0u32);
    let x = b.div(2u32.pow(exp));
    assert_eq!(x, BigUint::from(11u32));

    let x: [u32; 10] = [
        8_u32, 4_u32, 6_u32, 3_u32, 8_u32, 4_u32, 7_u32, 4_u32, 1_u32, 2_u32,
    ];
    let b = BigUint::from_slice(&x);
    assert_eq!(factor_2n(&b, 0u32), 3u32);
}

// calculate a's congruence mod m
fn witness(a: BigUint, exp: Factor2n, modulus: BigUint, count: u64) -> Option<u64> {
    let cong = fast_exponentiation(a, exp.xp(), &modulus).unwrap();

    if count == MAX_WITNESS_CHECK {
        // we have made it as far as we can go
        return cong.to_u64();
    }
    if cong == One::one() {
        // this is a probable prime to modulus base a, lets run it again to be sure with a new base a
        let a_ceil = modulus.to_u64().unwrap();
        let new_a = BigUint::from(fastrand::u64(1..a_ceil));
        return witness(new_a, exp, modulus, count + 1);
    }
    cong.to_u64()
}

#[test]
fn witness_test() {
    assert_eq!(Some(64u64), witness(BigUint::from(11u32), Factor2n { e: 2, k: &BigUint::from(3u32)}, BigUint::from(123u32), 0u64));
    assert_eq!(Some(0u64), witness(BigUint::from(11u32), Factor2n { e: 2, k: &BigUint::from(3u32)}, BigUint::from(121u64), 0u64));
    assert_eq!(Some(1u64), witness(BigUint::from(9u32), Factor2n { e: 2, k: &BigUint::from(3u32)}, BigUint::from(13u32), 0u64));
}

/// Run a Miller-Rabin primality check on n, returning None if n is composite.
pub fn miller_rabin(n: u64) -> Option<u64> {
    // exit early if n is even
    if n % 2 == 0 {
        return None;
    }
    // let big_n = BigUint::from_bytes_le(&n.to_le_bytes());
    let big_n = BigUint::from(n);
    let big_n_minus_one = BigUint::from(&n - 1);
    let exp: u32 = factor_2n(&big_n_minus_one, 0u32);
    let k = n / 2u64.pow(exp);
    let f: Factor2n = Factor2n {
        e: exp as u64,
        k: &BigUint::from(k)
    };

    // choose a random base A between 1 ->
    fastrand::seed(99);
    let a: BigUint = BigUint::from(fastrand::u64(1..n));
    let witness = witness(a, f, big_n, 0u64).unwrap();
    match witness {
        1u64 => Some(1u64),
        _ => None,
    }
}

#[test]
fn miller_test() {
    assert_eq!(Some(1), miller_rabin(17u64));
    // assert_eq!(None, miller_rabin(244u64));
    // assert_eq!(Some(1), miller_rabin(71u64));
    // assert_eq!(Some(1), miller_rabin(151u64));
    // assert_eq!(Some(1), miller_rabin(191u64));
}

// #[test]
// fn fact2_test() {
//     assert_eq!(factor_2n(36, None), Factor2n { e: 2, k: 9 });
//     assert_eq!(factor_2n(18, None), Factor2n { e: 1, k: 9 });
//     debug_assert_ne!(factor_2n(18, None), Factor2n { e: 9, k: 9818 });
// }

#[test]
fn fact2xp_test() {
    let f = Factor2n { e: 3, k: &BigUint::from(31u32) };
    assert_eq!(BigUint::from(248u32), f.xp());
}
// fn into_bin(mut e: BigUint) -> Vec<u32> {
//     let mut exps: Vec<u32> = Vec::new();
//     let mut k: u32 = 0;
//     let ex = e;
//     let zz = Zero::zero();
//     // ex.clone_from(&e);
//     while ex > &mut zz {
//         let one: BigUint = One::one();
//         if ex.bitand(one) == One::one() {
//             exps.push(k);
//         }
//         k += 1;
//         ex.shr_assign(1);
//     }
//     exps
// }

fn into_bin(e: BigUint) -> Vec<u32> {
    let bfmt = format!("{:b}", e);
    bfmt
        .chars()
        .rev()
        .enumerate()
        .filter(|c| c.1 == '1')
        .map(|x| x.0 as u32)
        .collect::<Vec<u32>>()
}

#[test]
fn into_bin_test() {
    assert_eq!(into_bin(BigUint::from(23u32)), vec![0u32, 1u32, 2u32, 4u32]);
    assert_eq!(into_bin(BigUint::from(23u32)), vec![0u32, 1u32, 2u32, 4u32]);
}

fn _pows(exps: Vec<u32>) -> Vec<BigUint> {
    exps.iter()
        .map(|exp| BigUint::new(vec![2_u32]).pow(*exp as u32))
        .collect::<Vec<BigUint>>()
}

#[test]
fn pows_test() {
    assert_eq!(_pows(vec![0u32, 1u32, 2u32]), vec![BigUint::from(1u32), BigUint::from(2u32), BigUint::from(4u32)]);
    assert_eq!(_pows(vec![]), vec![]);
    assert_eq!(_pows(vec![0u32]), vec![BigUint::from(1u32)]);
    assert_eq!(_pows(into_bin(BigUint::from(23u32))), vec![BigUint::from(1u32), BigUint::from(2u32), BigUint::from(4u32), BigUint::from(16u32)]);
}

/// determine the residue for a^exp mod m using binary
/// modular exponentiation
pub fn fast_exponentiation(a: BigUint, exp: BigUint, m: &BigUint) -> Option<BigUint> {
    let bins = into_bin(exp);
    let pows = _pows(bins);
    let _exp = &pows[0];
    let r = a.modpow(_exp, m);
    let mut last = FastExp {
        exp: &_exp,
        total: One::one(),
        residue: r,
    };

    for p in pows[1..].iter() {
        let tmp = p.checked_div(last.exp);
        if let Some(t) = tmp {
            last.total.mul_assign(&last.residue);
            last.residue = last.residue.modpow(&t, m);
            last.exp = p;
        }
    }
    last.total.mul_assign(last.residue);
    Some(last.total.rem(m))
}

#[test]
fn fastexponent_test() {
    assert_eq!(fast_exponentiation(BigUint::from(17_u32), BigUint::from(23_u32), &BigUint::from(121_u32)), Some(BigUint::from(51_u32)));
    assert_eq!(fast_exponentiation(BigUint::from(5_u64), BigUint::from(117_u64), &BigUint::from(19u64)), Some(BigUint::from(1_u32)));
    assert_eq!(fast_exponentiation(BigUint::from(15_u64), BigUint::from(1181_u64), &BigUint::from(41u64)), Some(BigUint::from(26_u32)));
    assert_eq!(fast_exponentiation(BigUint::from(91_u64), BigUint::from(19721_u64), &BigUint::from(87u64)), Some(BigUint::from(13_u32)));
    assert_eq!(fast_exponentiation(BigUint::from(192_u64), BigUint::from(39288849120094815_u64), &BigUint::from(71u64)), Some(BigUint::from(30_u32)));
}

pub fn pairs_set(set: HashSet<&u32>) -> HashSet<(&u32, &u32)> {
    // build a set of every pair of numbers in set
    let mut pairs: HashSet<(&u32, &u32)> = HashSet::new();

    for x in &set {
        for y in &set {
            if x != y && !pairs.contains(&(x, y)) && !pairs.contains(&(y, x)) {
                pairs.insert((x, y));
            }
        }
    }
    pairs
}

pub fn pairwise_coprime(nums: Vec<&u32>) -> bool {
    let set: HashSet<_> = nums.iter().cloned().collect();
    let pairs: HashSet<(&u32, &u32)> = pairs_set(set);

    for pair in pairs.iter() {
        dbg!(&pair.1);
        if gcd(*pair.0, *pair.1) != 1 {
            return false;
        }
    }
    true
}

/// brute force (exponential) check on a CRT
/// runs through a set of predicates to resolve a member of CRT residue class
pub fn crt_brute_force(predicates: Vec<CrtPredicate>, max: u32) -> u32 {
    let mut possible: HashSet<u32> = HashSet::new();

    for num in 0..max {
        if predicates.iter().all(|p| num % p.dividend == p.residue) {
            possible.insert(num as u32);
        }
    }
    if possible.is_empty() {
        1_u32
    } else {
        possible.iter().cloned().collect::<Vec<u32>>()[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_congruent() {
        assert_eq!(is_congruent(24_u32, 10_u32, 7_u32), true);
        assert_eq!(is_congruent(12_u32, 32_u32, 10_u32), true);
    }

    #[test]
    fn gcd_test() {
        assert_eq!(gcd(36_u32, 24_u32), 12_u32);
        assert_eq!(gcd(15_u32, 7_u32), 1_u32);
        assert_eq!(gcd(2017, 1024), 1_u32);
        assert_eq!(gcd(930, 992), 62_u32);
        assert_eq!(gcd(527, 612), 17_u32);
        assert_eq!(gcd(8, 15), 1u32);
    }

    #[test]
    fn sieve_test() {
        assert_eq!(sieve(12_u32), vec![1, 2, 3, 5, 7, 11]);
        assert_eq!(sieve(30_u32), vec![1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn prime_factor_test() {
        assert_eq!(prime_factor(54), vec![2, 3, 3, 3]);
        assert_eq!(prime_factor(12), vec![2, 2, 3]);
        assert_eq!(prime_factor(60), vec![2, 2, 3, 5]);
        assert_eq!(prime_factor(6125), vec![5, 5, 5, 7, 7]);
    }

    #[test]
    fn square_and_multiply_test() {
        assert_eq!(8192_i32, square_and_multiply(2_i32, 13_i32));
    }

    #[test]
    fn mod_exponent_test() {
        assert_eq!(445_i64, mod_exponent(4_i64, 13_i64, 497_i64));
        assert_eq!(526_i64, mod_exponent(526_i64, 959_i64, 527_i64));

        assert_eq!(125_i64, mod_exponent(47_i64, 69_i64, 143_i64));
        assert_eq!(1_i64, mod_exponent(15_i64, 15_i64, 14_i64));
    }

    #[test]
    fn totient_test() {
        assert_eq!(12_u32, totient(36_u32));
        assert_eq!(12_u32, totient(21_u32));
        assert_eq!(1_u32, totient(2_u32));
        assert_eq!(6_u32, totient(7_u32));
        assert_eq!(1_u32, totient(1_u32));
        assert_eq!(40_u32, totient(100_u32));
        assert_eq!(54_u32, totient(81_u32));
        assert_eq!(totient(326_095_u32), 203_280_u32);
        assert_eq!(totient(2717_u32), 2160_u32);
        assert_eq!(totient(81_u32), 54_u32);
    }

    #[test]
    fn coprimes_test() {
        assert_eq!(vec![1, 3, 7, 9], coprimes(10_u32));
        assert_eq!(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], coprimes(11_u32));
    }

    #[test]
    fn pairwise_test() {
        assert!(pairwise_coprime(vec![&5_u32, &12_u32, &19_u32]), true);
        assert_eq!(
            pairwise_coprime(vec![&5_u32, &12_u32, &19_u32, &25_u32]),
            false
        );
        assert_eq!(
            pairwise_coprime(vec![&237803_u32, &240199_u32, &242653_u32, &274327_u32]),
            false
        );
        assert_eq!(
            pairwise_coprime(vec![&237803_u32, &240199_u32, &242653_u32, &274327_u32]),
            false
        );
    }

    #[test]
    fn crt_brute_force_test() {
        assert_eq!(
            crt_brute_force(
                vec![
                    CrtPredicate {
                        residue: 19,
                        dividend: 25
                    },
                    CrtPredicate {
                        residue: 7,
                        dividend: 9
                    },
                    CrtPredicate {
                        residue: 2,
                        dividend: 4
                    },
                ],
                900_u32
            ),
            394_u32
        );
    }
}
