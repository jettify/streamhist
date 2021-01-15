# StreamHist
[![ci-badge](https://github.com/jettify/streamhist/workflows/CI/badge.svg)](https://github.com/jettify/streamhist/actions?query=workflow%3ACI)

A rust implementation of a streaming centroid histogram algorithm found in
[Streaming Parallel Decision Trees](http://jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf)
 paper by Ben-Haim/Tom-Tov.

 # Example

 ```rust

 use rand::SeedableRng;
 use rand_distr::{Distribution, Normal};
 use rand_isaac::Isaac64Rng;
 use streamhist::StreamingHistogram;

 fn main() {
     let mut rng = Isaac64Rng::seed_from_u64(42);
     let dist = Normal::new(2.0, 3.0).unwrap();
     let mut hist = StreamingHistogram::new(32);

     let maxn = 10000;
     let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();

     for v in vals.iter() {
         hist.insert_one(*v);
     }

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

     assert_eq!(hist.count(), maxn);
 }
```

 # Lincese
  Licensed under the Apache License, Version 2.0
