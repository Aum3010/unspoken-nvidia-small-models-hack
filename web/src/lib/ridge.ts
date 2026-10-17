// Small multi-output ridge regression for gaze calibration (features -> screen x, y).
// Features are z-scored with the calibration statistics; the bias is not penalised.

export interface RidgeModel { mean: number[]; std: number[]; w: number[][] } // w: one weight vector (with bias) per output

export function fitRidge(X: number[][], Y: number[][], lambda = 1): RidgeModel {
  const n = X.length, d = X[0].length, k = Y[0].length
  const mean = Array.from({ length: d }, (_, j) => X.reduce((a, r) => a + r[j], 0) / n)
  const std = Array.from({ length: d }, (_, j) => Math.sqrt(X.reduce((a, r) => a + (r[j] - mean[j]) ** 2, 0) / n) || 1)
  const Z = X.map((r) => [1, ...r.map((v, j) => (v - mean[j]) / std[j])])
  const D = d + 1
  const A = Array.from({ length: D }, (_, i) => Array.from({ length: D }, (_, j) => Z.reduce((a, r) => a + r[i] * r[j], 0) + (i === j && i > 0 ? lambda : 0)))
  const w = Array.from({ length: k }, (_, o) => solve(A, Array.from({ length: D }, (_, i) => Z.reduce((a, r, t) => a + r[i] * Y[t][o], 0))))
  return { mean, std, w }
}

export function predictRidge(m: RidgeModel, x: number[]): number[] {
  const z = [1, ...x.map((v, j) => (v - m.mean[j]) / m.std[j])]
  return m.w.map((w) => w.reduce((a, wi, i) => a + wi * z[i], 0))
}

/** Gaussian elimination with partial pivoting (A is copied). */
function solve(A0: number[][], b0: number[]): number[] {
  const n = b0.length
  const A = A0.map((r, i) => [...r, b0[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r
    ;[A[c], A[p]] = [A[p], A[c]]
    const piv = A[c][c] || 1e-12
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / piv
      for (let j = c; j <= n; j++) A[r][j] -= f * A[c][j]
    }
  }
  const x = new Array(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let s = A[r][n]
    for (let j = r + 1; j < n; j++) s -= A[r][j] * x[j]
    x[r] = s / (A[r][r] || 1e-12)
  }
  return x
}
