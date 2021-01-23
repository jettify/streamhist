//! A rust implementation of a streaming centroid histogram algorithm found in
//!  [Streaming Parallel Decision Trees](http://jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf)
//!  paper by Ben-Haim/Tom-Tov.
//!
//! # Example
//!
//! ```rust
//! use rand::SeedableRng;
//! use rand_distr::{Distribution, Normal};
//! use rand_isaac::Isaac64Rng;
//! use streamhist::StreamingHistogram;
//!
//! fn main() {
//!     let mut rng = Isaac64Rng::seed_from_u64(42);
//!     let dist = Normal::new(2.0, 3.0).unwrap();
//!     let mut hist = StreamingHistogram::new(32);
//!
//!     let maxn = 10000;
//!     let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();
//!
//!     for v in vals.iter() {
//!         hist.insert_one(*v);
//!     }
//!
//!     println!("------------------------------------------------");
//!     println!("Est Mean               {:?}", hist.mean().unwrap());
//!     println!("Est Var                {:?}", hist.var().unwrap());
//!     println!("Est Median             {:?}", hist.median().unwrap());
//!     println!("Est Count vals <= 2.0  {:?}", hist.count_less_then_eq(2.0));
//!     println!("Est quantile           {:?}", hist.quantile(0.75).unwrap());
//!     println!("Min                    {:?}", hist.min().unwrap());
//!     println!("Max                    {:?}", hist.max().unwrap());
//!     println!("Count                  {:?}", hist.count());
//!     println!("------------------------------------------------");
//!
//!     assert_eq!(hist.count(), maxn);
//! }
//!```

#![warn(clippy::all)]

use std::convert::TryFrom;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// Centroid is represented by value (p in paper) and its count (m in paper).
/// Half of the `count` located to lest if centorid other to right.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
struct Centroid {
    value: f64, // p
    count: u64, // m
}

impl Centroid {
    // Merges two centroids into one, by simple summing counts and weighted sum
    // of values
    #[inline]
    fn merge(&mut self, other: &Centroid) {
        let sum = self.count + other.count;
        let val =
            (self.value * (self.count as f64) + other.value * (other.count as f64)) / (sum as f64);
        self.value = val;
        self.count = sum;
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct StreamingHistogram {
    min: f64,
    max: f64,
    max_centroids: u16,
    count: u64,
    centroids: Vec<Centroid>,
}

impl Default for StreamingHistogram {
    /// Creates new StreamingHistogram with default `max_centroids=64`
    fn default() -> StreamingHistogram {
        StreamingHistogram {
            min: f64::MAX,
            max: f64::MIN,
            max_centroids: 64,
            count: 0,
            centroids: Vec::with_capacity(64 + 1),
        }
    }
}

impl StreamingHistogram {
    /// Creates new StreamingHistogram
    ///
    /// * `max_centroids` indicates how many centroids should be used to estimate
    /// statistics, more centroids more accurate estimation. Each centroids
    /// stroes `f64` and `u64` and uses about 16 bytes of space.
    ///
    /// # Example
    /// ```
    /// use streamhist::StreamingHistogram;
    /// let mut hist = StreamingHistogram::new(32);
    /// assert_eq!(hist.count(), 0)
    /// ```
    /// # Panics
    ///
    /// The function panics if the `max_centroids` is zero.
    ///
    /// ```rust,should_panic
    /// use streamhist::StreamingHistogram;
    /// // panics on empty histogram
    /// let mut hist = StreamingHistogram::new(0);
    /// ```
    pub fn new(max_centroids: u16) -> StreamingHistogram {
        if max_centroids == 0 {
            panic!("Max number of centroids can not be zero.");
        };
        let size = usize::try_from(max_centroids).unwrap() + 1;
        StreamingHistogram {
            min: f64::MAX,
            max: f64::MIN,
            max_centroids: max_centroids,
            count: 0,
            centroids: Vec::with_capacity(size),
        }
    }

    /// Clearas all state like centroids and restests all counts.
    ///
    /// # Example
    /// ```
    /// use streamhist::StreamingHistogram;
    /// let mut hist = StreamingHistogram::new(32);
    /// hist.insert_one(10.0);
    /// assert_eq!(hist.count(), 1);
    /// hist.clear();
    /// assert_eq!(hist.count(), 0)
    /// ```
    pub fn clear(&mut self) {
        self.min = f64::MAX;
        self.max = f64::MIN;
        self.count = 0;
        self.centroids.clear();
    }

    /// Returns true if this histogram has no recorded values.
    ///
    /// # Example
    /// ```
    /// use streamhist::StreamingHistogram;
    /// let mut hist = StreamingHistogram::new(32);
    /// assert!(hist.is_empty(), 1);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns number of items observed by histogram.
    ///
    /// # Example
    /// ```
    /// use streamhist::StreamingHistogram;
    /// let mut hist = StreamingHistogram::new(32);
    /// assert_eq!(hist.count(), 0);
    /// hist.insert_one(10.0);
    /// assert_eq!(hist.count(), 1)
    /// ```
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns max value ever observed by histogram.
    ///
    /// # Example
    /// ```
    /// use streamhist::StreamingHistogram;
    /// let mut hist = StreamingHistogram::new(32);
    /// assert_eq!(hist.max(), None);
    /// hist.insert_one(10.0);
    /// assert!(hist.max().unwrap() > 9.0);
    /// ```
    pub fn max(&self) -> Option<f64> {
        if self.count() == 0 {
            None
        } else {
            Some(self.max)
        }
    }

    /// Returns min value ever observed by histogram.
    ///
    /// # Example
    /// ```
    /// use streamhist::StreamingHistogram;
    /// let mut hist = StreamingHistogram::new(32);
    /// assert_eq!(hist.min(), None);
    /// hist.insert_one(10.0);
    /// assert!(hist.min().unwrap() > 9.0);
    /// ```
    pub fn min(&self) -> Option<f64> {
        if self.count() == 0 {
            None
        } else {
            Some(self.min)
        }
    }

    pub fn insert_one(&mut self, value: f64) {
        self.insert(value, 1)
    }

    pub fn insert(&mut self, value: f64, count: u64) {
        self.min = f64::min(self.min, value);
        self.max = f64::max(self.max, value);
        self.count += count;

        let c = Centroid { value, count };

        let k = self.centroids.iter().position(|c| c.value >= value);

        if let Some(idx) = k {
            if self.centroids[idx].value == value {
                self.centroids[idx].count += count;
                return;
            }
        } else {
            self.centroids.push(c);
            return;
        }
        self.centroids.insert(k.unwrap(), c);
        if self.centroids.len() <= usize::try_from(self.max_centroids).unwrap() {
            return;
        }

        let l = self.closest_centroids();
        let cj = self.centroids[l + 1];
        let ci = &mut self.centroids[l];
        ci.merge(&cj);
        self.centroids.remove(l + 1);
    }

    /// Returns estimate of mean value using centroids.
    ///
    /// Note, it is possible to compute exact mean value over data stream
    /// using itertive algorithm. Estimate with centroids is resonably accurate
    /// try with your distribution.
    ///
    /// # Example
    /// ```
    /// use streamhist::StreamingHistogram;
    /// let mut hist = StreamingHistogram::new(32);
    /// hist.insert(10.0, 1);
    /// assert!(hist.mean().unwrap() > 9.0);
    /// ```
    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let n: f64 = self.count as f64;
        let m = self
            .centroids
            .iter()
            .map(|c| c.value * (c.count as f64) / n)
            .sum();
        Some(m)
    }

    /// Returns estimate of variance value using centroids.
    ///
    /// Note, it is possible to compute exact mean value over data stream
    /// using itertive algorithm. Estimate with centroids is resonably accurate
    /// try with your distribution.
    ///
    /// # Example
    /// ```
    /// use streamhist::StreamingHistogram;
    /// let mut hist = StreamingHistogram::new(32);
    /// hist.insert(10.0, 1);
    /// hist.insert(15.0, 1);
    /// assert!(hist.var().unwrap() < 7.0);
    /// ```
    pub fn var(&self) -> Option<f64> {
        match self.mean() {
            Some(m) => {
                let n: f64 = self.count as f64;
                let v = self
                    .centroids
                    .iter()
                    .map(|c| (c.value - m).powf(2.0) * (c.count as f64) / n)
                    .sum();
                Some(v)
            }
            _ => None,
        }
    }

    pub fn sum(&self, value: f64) -> u64 {
        if value >= self.max {
            return self.count;
        } else if value < self.min {
            return 0;
        }
        let k = self.centroids.iter().position(|c| c.value > value).unwrap();
        let s: u64 = self.centroids.iter().take(k - 1).map(|c| c.count).sum();

        let (ci, cj) = self.border_centroids(k);
        if ci.count > 0 && (cj.count > 0 && cj.value == value) {
            return s + cj.count;
        }
        let ci_count = ci.count as f64;
        let cj_count = cj.count as f64;
        let r: f64 = (value - ci.value) / (cj.value - ci.value);
        let mb: f64 = ci_count + (cj_count - ci_count) * r;
        s + (0.5 * ci_count + 0.5 * (ci_count + mb) * r).round() as u64
    }

    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        if q < 0.0 {
            return Some(self.min);
        }
        if q > 1.0 {
            return Some(self.max);
        }

        let t: f64 = q * (self.count as f64);
        let mut prev_half_count = 0.0;
        let mut s: f64 = 0.0;
        let mut idx: Option<usize> = None;

        for (i, c) in self.centroids.iter().enumerate() {
            let half_count = (c.count as f64) / 2.0;

            if s + half_count + prev_half_count > t {
                idx = Some(i);
                break;
            }

            s += half_count + prev_half_count;
            prev_half_count = half_count;
        }
        // TODO check this logic
        if idx.is_none() {
            return Some(self.max);
        }
        let (ci, cj) = self.border_centroids(idx.unwrap());

        let ci_count = ci.count as f64;
        let cj_count = cj.count as f64;
        let d = t - s;
        let a = (cj_count - ci_count) as f64;
        if a == 0.0 {
            return Some(ci.value + (cj.value - ci.value) * (d / ci_count));
        }
        let b = 2.0 * ci_count;
        let c = -2.0 * d;
        let z = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);
        Some(ci.value + (cj.value - ci.value) * z)
    }

    /// Returns estimate of median.
    ///
    /// # Example
    /// ```
    /// use streamhist::StreamingHistogram;
    /// let mut hist = StreamingHistogram::new(32);
    /// hist.insert(5.0, 1);
    /// hist.insert(10.0, 1);
    /// hist.insert(15.0, 1);
    /// assert!((hist.median().unwrap() - 10.0).abs() < 0.00001);
    /// ```
    pub fn median(&self) -> Option<f64> {
        self.quantile(0.5)
    }

    /// Merge second histogram into this one.
    ///
    /// # Example
    /// ```
    /// use streamhist::StreamingHistogram;
    /// let mut one = StreamingHistogram::new(32);
    /// let mut two = StreamingHistogram::new(32);
    /// one.insert(5.0, 1);
    /// one.insert(10.0, 1);
    /// one.insert(15.0, 1);
    /// two.insert(20.0, 1);
    /// two.insert(25.0, 1);
    /// one.merge(&two);
    /// assert_eq!(one.count(), 5);
    /// ```
    pub fn merge(&mut self, other: &Self) {
        for c in other.centroids.iter() {
            self.insert(c.value, c.count)
        }
    }

    pub fn count_less_then_eq(&self, value: f64) -> u64 {
        self.sum(value)
    }

    #[inline]
    fn border_centroids(&self, k: usize) -> (Centroid, Centroid) {
        if k == 0 {
            return (
                Centroid {
                    value: self.min,
                    count: 0,
                },
                self.centroids[0],
            );
        } else if k == self.centroids.len() {
            return (
                self.centroids[k],
                Centroid {
                    value: self.max,
                    count: 0,
                },
            );
        }
        (self.centroids[k - 1], self.centroids[k])
    }

    #[inline]
    fn closest_centroids(&self) -> usize {
        let mut min_dis = f64::MAX;
        let mut idx = 0;
        // TODO: use window function to do iterations
        for i in 0..self.centroids.len() - 1 {
            let d = (self.centroids[i + 1].value - self.centroids[i].value).abs();
            if d < min_dis {
                min_dis = d;
                idx = i;
            }
        }
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingHistogram;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Exp, LogNormal, Normal, Uniform};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn test_basic_ctor() {
        let mut hist = StreamingHistogram::new(3);
        assert!(hist.is_empty());

        hist.insert(25.0, 1);
        hist.insert(21.0, 1);
        hist.insert(20.0, 1);

        assert!(!hist.is_empty());

        assert_eq!(hist.count(), 3);
        assert_eq!(hist.count_less_then_eq(-100.0), 0);
        assert_eq!(hist.count_less_then_eq(100.0), 3);
        assert_relative_eq!(hist.max().unwrap(), 25.0);
        assert_relative_eq!(hist.min().unwrap(), 20.0);
        assert_relative_eq!(hist.quantile(0.0).unwrap(), 20.0);
        assert_relative_eq!(hist.quantile(1.0).unwrap(), 25.0);
    }

    #[test]
    fn test_default_ctor() {
        let mut hist = StreamingHistogram::default();
        hist.insert(25.0, 1);
        assert!(!hist.is_empty());
        assert_eq!(hist.count(), 1);
    }

    #[test]
    fn test_empty_struct() {
        let hist = StreamingHistogram::new(3);
        assert!(hist.is_empty());

        assert_eq!(hist.count(), 0);
        assert_eq!(hist.count_less_then_eq(-100.0), 0);
        assert_eq!(hist.count_less_then_eq(100.0), 0);
        assert_eq!(hist.max(), None);
        assert_eq!(hist.mean(), None);
        assert_eq!(hist.median(), None);
        assert_eq!(hist.min(), None);
        assert_eq!(hist.quantile(0.0), None);
        assert_eq!(hist.quantile(1.0), None);
        assert_eq!(hist.var(), None);
    }

    #[test]
    #[should_panic]
    fn test_zero_centroids() {
        StreamingHistogram::new(0);
    }

    #[test]
    fn test_same_value() {
        let mut hist = StreamingHistogram::new(3);
        hist.insert(25.0, 1);
        hist.insert(25.0, 1);
        hist.insert(25.0, 1);
        hist.insert(25.0, 1);
        hist.insert(25.0, 1);
        assert_eq!(hist.centroids.len(), 1);

        assert_relative_eq!(hist.max().unwrap(), 25.0);
        assert_relative_eq!(hist.min().unwrap(), 25.0);
        assert_relative_eq!(hist.quantile(0.0).unwrap(), 25.0);
        assert_relative_eq!(hist.quantile(1.0).unwrap(), 25.0);
        assert_relative_eq!(hist.median().unwrap(), 25.0);
        assert_relative_eq!(hist.var().unwrap(), 0.0);
        assert_eq!(hist.count_less_then_eq(-100.0), 0);
        assert_eq!(hist.count_less_then_eq(100.0), 5);
        assert_eq!(hist.count(), 5);
    }

    #[test]
    fn test_closest_centroids() {
        let mut hist = StreamingHistogram::new(3);
        hist.insert(1.0, 1);
        hist.insert(5.0, 1);
        hist.insert(11.0, 1);
        assert_eq!(hist.closest_centroids(), 0);

        hist.clear();
        hist.insert(1.0, 1);
        hist.insert(5.0, 1);
        hist.insert(7.0, 1);
        assert_eq!(hist.closest_centroids(), 1);
    }
    #[test]
    fn test_merge() {
        let mut hist = StreamingHistogram::new(3);
        let mut other = StreamingHistogram::new(3);
        hist.insert(1.0, 1);
        hist.insert(5.0, 1);

        other.insert(10.0, 10);
        other.insert(20.0, 10);
        hist.merge(&other);
        assert_eq!(hist.count(), 22);
    }

    fn assert_distribution(mut vals: Vec<f64>, mut hist: StreamingHistogram, tol: f64) {
        assert!(hist.is_empty());
        for v in vals.iter() {
            hist.insert(*v, 1);
        }
        assert!(!hist.is_empty());
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n: f64 = vals.len() as f64;
        let s: f64 = vals.iter().sum();
        let exact_mean: f64 = s / n;
        let exact_var: f64 = vals.iter().map(|v| (v - exact_mean).powf(2.0) / n).sum();
        let exact_median: f64 = vals[vals.len() / 2];
        let exact_sum: u64 = vals.iter().map(|v| if *v <= 2.0 { 1 } else { 0 }).sum();

        assert_eq!(hist.count(), vals.len() as u64);
        assert_relative_eq!(hist.min().unwrap(), vals[0]);
        assert_relative_eq!(hist.max().unwrap(), vals[vals.len() - 1]);
        assert_relative_eq!(hist.mean().unwrap(), exact_mean, max_relative = tol);
        assert_relative_eq!(hist.var().unwrap(), exact_var, max_relative = tol);
        assert_relative_eq!(hist.median().unwrap(), exact_median, max_relative = tol);
        assert_relative_eq!(
            hist.quantile(0.5).unwrap(),
            exact_median,
            max_relative = tol
        );
        assert_relative_eq!(
            hist.count_less_then_eq(2.0) as f64,
            exact_sum as f64,
            max_relative = tol
        );

        assert_eq!(hist.count_less_then_eq(vals[0] - 1.0), 0);
        assert_eq!(
            hist.count_less_then_eq(vals[vals.len() - 1] + 1.0),
            vals.len() as u64
        );
    }

    #[test]
    fn test_normal_distribution() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let dist = Normal::new(2.0, 3.0).unwrap();
        let hist = StreamingHistogram::new(64);

        let tol = 0.005;
        let maxn = 10000;
        let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();
        assert_distribution(vals, hist, tol)
    }

    #[test]
    fn test_log_normal_distribution() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let dist = LogNormal::new(1.0, 1.0).unwrap();
        let hist = StreamingHistogram::new(64);

        let tol = 0.02;
        let maxn = 10000;
        let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();
        assert_distribution(vals, hist, tol)
    }

    #[test]
    fn test_exp_distribution() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let dist = Exp::new(1.5).unwrap();
        let hist = StreamingHistogram::new(64);

        let tol = 0.05;
        let maxn = 10000;
        let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();
        assert_distribution(vals, hist, tol)
    }

    #[test]
    fn test_uniform_distribution() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let dist = Uniform::new(0.0, 1.0);
        let hist = StreamingHistogram::new(16);

        let tol = 0.01;
        let maxn = 10000;
        let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();
        assert_distribution(vals, hist, tol)
    }

    #[test]
    fn test_all_values_the_same() {
        let hist = StreamingHistogram::new(2);

        let tol = 0.0001;
        let maxn = 10000;
        let vals: Vec<f64> = vec![17.0; maxn];
        assert_distribution(vals, hist, tol)
    }

    #[test]
    fn test_normal_distribution_one_centroid() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let dist = Normal::new(2.0, 3.0).unwrap();
        let hist = StreamingHistogram::new(1);

        let tol = 0.5;
        let maxn = 10000;
        let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();
        assert_distribution(vals, hist, tol)
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde() {
        let mut hist = StreamingHistogram::new(3);

        hist.insert(1.0, 1);
        hist.insert(5.0, 1);
        hist.insert(11.0, 1);

        let serialized_h = serde_json::to_string(&hist).unwrap();
        let other_hist: StreamingHistogram = serde_json::from_str(&serialized_h).unwrap();
        assert_eq!(hist.count(), other_hist.count());
    }
}
