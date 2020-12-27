use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use streamhist::StreamHist;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(2.0, 3.0).unwrap();
    let mut hist = StreamHist::new(64);

    let maxn = 1000;
    let vals: Vec<f64> = (0..maxn).map(|_| normal.sample(&mut rng)).collect();

    for v in vals.iter() {
        hist.insert(*v, 1);
    }

    let n: f64 = vals.len() as f64;
    let exact_sum: f64 = vals.iter().sum();
    let exact_mean: f64 = exact_sum / n;
    let exact_var: f64 = vals.iter().map(|v| (v - exact_mean).powf(2.0) / n).sum();

    println!("------------------------------------------------");
    println!("StreamHist Mean {:?}", hist.mean());
    println!("Mean            {:?}", exact_mean);
    println!("Abs Difference  {:?}", exact_mean - hist.mean());
    println!("------------------------------------------------");
    println!("StreamHist Var  {:?}", hist.var());
    println!("Var             {:?}", exact_var);
    println!("ABS Difference  {:?}", exact_var - hist.var());
    println!("------------------------------------------------");
}
