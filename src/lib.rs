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

        // unsafe { breakpoint() };
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
        let mut pv = 0.0;
        let mut s: f64 = 0.0;
        let mut idx: usize = 0;

        for (i, c) in self.centroids.iter().enumerate() {
            idx = i;
            let v = (c.count as f64) / 2.0;
            if s + v + pv > t {
                break;
            }
            s += v + pv;
            pv = v;
        }
        let (ci, cj) = self.border_centroids(idx);
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
        (self.centroids[k], self.centroids[k + 1])
    }

    pub fn count_lte(&self, value: f64) -> u64 {
        self.sum(value)
    }

    fn closest_centroids(&self) -> usize {
        let mut min_dis = f64::MAX;
        let mut idx = 0;
        let last = self.centroids.len() - 1;
        // TODO: use window function to do iterations
        for (i, c) in self.centroids[..last - 1].iter().enumerate() {
            if (self.centroids[i + 1].value - self.centroids[i].value).abs() < min_dis {
                min_dis = (self.centroids[i + 1].value - self.centroids[i].value).abs();
                idx = i
            }
        }
        return idx;
    }
}

#[cfg(test)]
mod tests {
    extern crate approx;

    use super::StreamHist;
    use approx::relative_eq;

    #[test]
    fn test_basic_ctor() {
        let mut hist = StreamHist::new(3);
        hist.insert(25.0, 1);
        hist.insert(20.0, 1);
        hist.insert(10.0, 1);
        hist.insert(1.0, 1);
        hist.insert(19.0, 1);
        assert_eq!(hist.count, 5);

        relative_eq!(hist.max(), 1.0);
        relative_eq!(hist.min(), 25.0);
        assert_eq!(hist.count(), 5);
        println!("{:?}", hist)
    }
}
