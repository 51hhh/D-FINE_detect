#pragma once
/// @file mat6.h
/// Header-only 固定大小矩阵/向量（N=3,6），用于 EKF 运算。无外部依赖。

#include <array>
#include <cmath>
#include <cstring>

namespace zed_ds {

// ── 固定大小向量 ──
template <int N>
struct VecN {
  std::array<float, N> d{};

  float &operator[](int i) { return d[i]; }
  float operator[](int i) const { return d[i]; }

  VecN operator+(const VecN &o) const {
    VecN r;
    for (int i = 0; i < N; ++i)
      r.d[i] = d[i] + o.d[i];
    return r;
  }
  VecN operator-(const VecN &o) const {
    VecN r;
    for (int i = 0; i < N; ++i)
      r.d[i] = d[i] - o.d[i];
    return r;
  }
  VecN operator*(float s) const {
    VecN r;
    for (int i = 0; i < N; ++i)
      r.d[i] = d[i] * s;
    return r;
  }
  float Dot(const VecN &o) const {
    float s = 0.0f;
    for (int i = 0; i < N; ++i)
      s += d[i] * o.d[i];
    return s;
  }
  float Norm() const { return std::sqrt(Dot(*this)); }
};

// ── 固定大小 N×N 矩阵（行优先） ──
template <int N>
struct MatN {
  std::array<float, N * N> d{};

  float &operator()(int r, int c) { return d[r * N + c]; }
  float operator()(int r, int c) const { return d[r * N + c]; }

  static MatN Identity() {
    MatN m;
    for (int i = 0; i < N; ++i)
      m(i, i) = 1.0f;
    return m;
  }

  static MatN Zero() { return MatN{}; }

  MatN operator+(const MatN &o) const {
    MatN r;
    for (int i = 0; i < N * N; ++i)
      r.d[i] = d[i] + o.d[i];
    return r;
  }

  MatN operator-(const MatN &o) const {
    MatN r;
    for (int i = 0; i < N * N; ++i)
      r.d[i] = d[i] - o.d[i];
    return r;
  }

  MatN operator*(const MatN &o) const {
    MatN r;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        float s = 0.0f;
        for (int k = 0; k < N; ++k)
          s += (*this)(i, k) * o(k, j);
        r(i, j) = s;
      }
    return r;
  }

  MatN operator*(float s) const {
    MatN r;
    for (int i = 0; i < N * N; ++i)
      r.d[i] = d[i] * s;
    return r;
  }

  MatN Transpose() const {
    MatN r;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        r(j, i) = (*this)(i, j);
    return r;
  }

  float Trace() const {
    float s = 0.0f;
    for (int i = 0; i < N; ++i)
      s += (*this)(i, i);
    return s;
  }

  /// 高斯消元法求逆。返回 false 表示奇异。
  bool Inverse(MatN *out) const {
    if (!out)
      return false;
    // 增广矩阵 [A | I]
    float aug[N][2 * N]{};
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j)
        aug[i][j] = (*this)(i, j);
      aug[i][N + i] = 1.0f;
    }
    for (int col = 0; col < N; ++col) {
      // 列主元选取
      int pivot = col;
      float max_val = std::abs(aug[col][col]);
      for (int row = col + 1; row < N; ++row) {
        float v = std::abs(aug[row][col]);
        if (v > max_val) {
          max_val = v;
          pivot = row;
        }
      }
      if (max_val < 1e-12f)
        return false; // 奇异
      if (pivot != col) {
        for (int j = 0; j < 2 * N; ++j)
          std::swap(aug[col][j], aug[pivot][j]);
      }
      const float diag = aug[col][col];
      for (int j = 0; j < 2 * N; ++j)
        aug[col][j] /= diag;
      for (int row = 0; row < N; ++row) {
        if (row == col)
          continue;
        const float factor = aug[row][col];
        for (int j = 0; j < 2 * N; ++j)
          aug[row][j] -= factor * aug[col][j];
      }
    }
    *out = MatN{};
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        (*out)(i, j) = aug[i][N + j];
    return true;
  }
};

// ── 矩阵 × 向量 ──
template <int N>
VecN<N> operator*(const MatN<N> &m, const VecN<N> &v) {
  VecN<N> r;
  for (int i = 0; i < N; ++i) {
    float s = 0.0f;
    for (int j = 0; j < N; ++j)
      s += m(i, j) * v[j];
    r[i] = s;
  }
  return r;
}

// ── 向量外积：v * w^T → N×N 矩阵 ──
template <int N>
MatN<N> OuterProduct(const VecN<N> &v, const VecN<N> &w) {
  MatN<N> r;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      r(i, j) = v[i] * w[j];
  return r;
}

// ── 常用别名 ──
using Mat6 = MatN<6>;
using Mat3 = MatN<3>;
using Vec6 = VecN<6>;
using Vec3 = VecN<3>;

// ── 辅助：从 Mat6 提取 3×3 子块 ──
inline Mat3 SubBlock33(const Mat6 &m, int r0, int c0) {
  Mat3 s;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      s(i, j) = m(r0 + i, c0 + j);
  return s;
}

} // namespace zed_ds
