use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_isaac::Isaac64Rng;
use streamhist::StreamHist;

fn main() {
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let dist = Normal::new(2.0, 3.0).unwrap();
    let mut hist = StreamHist::new(32);

    let maxn = 10000;
    let mut vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();

    for v in vals.iter() {
        hist.insert(*v, 1);
    }

    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n: f64 = vals.len() as f64;
    let exact_sum: f64 = vals.iter().sum();
    let exact_mean: f64 = exact_sum / n;
    let exact_var: f64 = vals.iter().map(|v| (v - exact_mean).powf(2.0) / n).sum();
    let exact_median: f64 = vals[vals.len() / 2];
    let exact_sum: u64 = vals.iter().map(|v| if *v <= 2.0 { 1 } else { 0 }).sum();

    println!("------------------------------------------------");
    println!("Est Mean               {:?}", hist.mean().unwrap());
    println!("Est Var                {:?}", hist.var().unwrap());
    println!("Est Median             {:?}", hist.median().unwrap());
    println!("Est Count vals <= 2.0  {:?}", hist.count_less_then_eq(2.0));
    println!("Est quantile           {:?}", hist.quantile(0.75).unwrap());
    println!("Min                    {:?}", hist.min().unwrap());
    println!("Max                    {:?}", hist.max().unwrap());
    println!("Count                  {:?}", hist.count());
    println!("------------------------------------------------");
    println!("Mean                   {:?}", exact_mean);
    println!("Var                    {:?}", exact_var);
    println!("Median                 {:?}", exact_median);
    println!("Count vals <= 2.0      {:?}", exact_sum);
    println!("------------------------------------------------");
}
