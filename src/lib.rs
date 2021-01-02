// #![feature(core_intrinsics)]
// use std::intrinsics::breakpoint;
use std::convert::TryFrom;

#[derive(Copy, Clone, Debug)]
pub struct Centroid {
    value: f64,
    count: u64,
}

impl Centroid {
    fn merge(&mut self, other: &Centroid) {
        let sum = self.count + other.count;
        let val =
            (self.value * (self.count as f64) + other.value * (other.count as f64)) / (sum as f64);
        self.value = val;
        self.count = sum;
    }
}

#[derive(Clone, Debug)]
pub struct StreamHist {
    min: f64,
    max: f64,
    max_centroids: u16,
    count: u64,
    centroids: Vec<Centroid>,
}

impl Default for StreamHist {
    fn default() -> StreamHist {
        StreamHist {
            min: f64::MAX,
            max: f64::MIN,
            max_centroids: 64,
            count: 0,
            centroids: Vec::with_capacity(64 + 1),
        }
    }
}

impl StreamHist {
    pub fn new(max_centroids: u16) -> StreamHist {
        let size = usize::try_from(max_centroids).unwrap() + 1;
        StreamHist {
            min: f64::MAX,
            max: f64::MIN,
            max_centroids: max_centroids,
            count: 0,
            centroids: Vec::with_capacity(size),
        }
    }

    pub fn clear(&mut self) {
        self.min = f64::MAX;
        self.max = f64::MIN;
        self.count = 0;
        self.centroids.clear();
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn min(&self) -> f64 {
        self.min
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
        // TODO: use copy
        let cj = self.centroids[l + 1].clone();
        let ci = &mut self.centroids[l];
        ci.merge(&cj);
        self.centroids.remove(l + 1);
    }
    pub fn mean(&self) -> f64 {
        let n: f64 = self.count as f64;
        self.centroids
            .iter()
            .map(|c| c.value * (c.count as f64) / n)
            .sum()
    }

    pub fn var(&self) -> f64 {
        let m: f64 = self.mean();
        let n: f64 = self.count as f64;
        self.centroids
            .iter()
            .map(|c| (c.value - m).powf(2.0) * (c.count as f64) / n)
            .sum()
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

    pub fn quantile(&self, q: f64) -> f64 {
        if q < 0.0 {
            return self.min;
        }
        if q > 1.0 {
            return self.max;
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
        if idx.is_none() {
            return self.max;
        }
        let (ci, cj) = self.border_centroids(idx.unwrap());

        let ci_count = ci.count as f64;
        let cj_count = cj.count as f64;
        let d = t - s;
        let a = (cj_count - ci_count) as f64;
        if a == 0.0 {
            return ci.value + (cj.value - ci.value) * (d / ci_count);
        }
        let b = 2.0 * ci_count;
        let c = -2.0 * d;
        let z = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);
        ci.value + (cj.value - ci.value) * z
    }

    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }

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

    pub fn count_less_then_eq(&self, value: f64) -> u64 {
        self.sum(value)
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

    pub fn merge(&mut self, other: &Self) {
        for c in other.centroids.iter() {
            self.insert(c.value, c.count)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StreamHist;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Exp, LogNormal, Normal, Uniform};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn test_basic_ctor() {
        let mut hist = StreamHist::new(3);
        hist.insert(25.0, 1);
        hist.insert(21.0, 1);
        hist.insert(20.0, 1);

        assert_relative_eq!(hist.max(), 25.0);
        assert_relative_eq!(hist.min(), 20.0);
        assert_eq!(hist.count(), 3);
        assert_relative_eq!(hist.quantile(0.0), 20.0);
        assert_relative_eq!(hist.quantile(1.0), 25.0);
        assert_eq!(hist.count_less_then_eq(-100.0), 0);
        assert_eq!(hist.count_less_then_eq(100.0), 3);
    }

    #[test]
    fn test_closest_centroids() {
        let mut hist = StreamHist::new(3);
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
        let mut hist = StreamHist::new(3);
        let mut other = StreamHist::new(3);
        hist.insert(1.0, 1);
        hist.insert(5.0, 1);

        other.insert(10.0, 10);
        other.insert(20.0, 10);
        hist.merge(&other);
        assert_eq!(hist.count(), 22);
    }

    fn assert_distribution(mut vals: Vec<f64>, mut hist: StreamHist, tol: f64) {
        for v in vals.iter() {
            hist.insert(*v, 1);
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n: f64 = vals.len() as f64;
        let s: f64 = vals.iter().sum();
        let exact_mean: f64 = s / n;
        let exact_var: f64 = vals.iter().map(|v| (v - exact_mean).powf(2.0) / n).sum();
        let exact_median: f64 = vals[vals.len() / 2];
        let exact_sum: u64 = vals.iter().map(|v| if *v <= 2.0 { 1 } else { 0 }).sum();
        assert_relative_eq!(hist.min(), vals[0]);
        assert_relative_eq!(hist.max(), vals[vals.len() - 1]);
        assert_eq!(hist.count(), vals.len() as u64);

        assert_relative_eq!(hist.mean(), exact_mean, max_relative = tol);
        assert_relative_eq!(hist.var(), exact_var, max_relative = tol);
        assert_relative_eq!(hist.median(), exact_median, max_relative = tol);
        assert_relative_eq!(hist.quantile(0.5), exact_median, max_relative = tol);
        assert_relative_eq!(
            hist.count_less_then_eq(2.0) as f64,
            exact_sum as f64,
            max_relative = tol
        );
    }

    #[test]
    fn test_normal_distribution() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let dist = Normal::new(2.0, 3.0).unwrap();
        let hist = StreamHist::new(64);

        let tol = 0.005;
        let maxn = 10000;
        let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();
        assert_distribution(vals, hist, tol)
    }

    #[test]
    fn test_log_normal_distribution() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let dist = LogNormal::new(1.0, 1.0).unwrap();
        let hist = StreamHist::new(64);

        let tol = 0.02;
        let maxn = 10000;
        let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();
        assert_distribution(vals, hist, tol)
    }

    #[test]
    fn test_exp_distribution() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let dist = Exp::new(1.5).unwrap();
        let hist = StreamHist::new(64);

        let tol = 0.05;
        let maxn = 10000;
        let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();
        assert_distribution(vals, hist, tol)
    }

    //#[test]
    fn test_uniform_distribution() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let dist = Uniform::new(-1000.0, 1000.0);
        let hist = StreamHist::new(64);

        let tol = 0.01;
        let maxn = 10000;
        let vals: Vec<f64> = (0..maxn).map(|_| dist.sample(&mut rng)).collect();
        assert_distribution(vals, hist, tol)
    }
}
