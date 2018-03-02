
//#define EIGEN_USE_BLAS
//#define EIGEN_ENABLE_AVX512
//#define EIGEN_VECTORIZE_AVX512

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <benchmark/benchmark.h>

typedef std::complex<double> value_t;

extern size_t o, n;
extern int O, N;
extern double s;

/*

// This is an incorrect expression, it is not a gemm operation!

static void G2_MM_ph_w0_last_term_Eigen_gemm(benchmark::State &state) {

  Eigen::Tensor<value_t, 4> M1(o, o, n, n);
  Eigen::Tensor<value_t, 4> M2(o, o, n, n);
  Eigen::Tensor<value_t, 6> g2(o, o, o, o, n, n);

  typedef Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
  typedef Eigen::Map<matrix_t> matrix_view_t;

  for (auto _ : state) {

    for (int i = 0; i < o; i++)
      for (int j = 0; j < o; j++)
        for (int k = 0; k < o; k++)
          for (int l = 0; l < o; l++) {

            auto em1 = matrix_view_t(&M1(i, l, 0, 0), n, n);
            auto em2 = matrix_view_t(&M2(k, j, 0, 0), n, n);
            auto eg2 = matrix_view_t(&g2(i, j, k, l, 0, 0), n, n);

            eg2.noalias() -= s * em1 * em2.transpose();
          }
  }
}
*/

static void G2_MM_ph_w0_last_term_Eigen_tensor_map(benchmark::State &state) {

  Eigen::Tensor<value_t, 4> M1(o, o, n, n);
  Eigen::Tensor<value_t, 4> M2(o, o, n, n);
  Eigen::Tensor<value_t, 6> g2(o, o, o, o, n, n);

  typedef Eigen::TensorMap<value_t, 2> tensor_map_t;

  for (auto _ : state) {

    for (int i = 0; i < o; i++)
      for (int j = 0; j < o; j++)
        for (int k = 0; k < o; k++)
          for (int l = 0; l < o; l++) {

	    /*
            tensor_map_t em1(M1(i, l, 0, 0), n, n);
            tensor_map_t em2(M2(k, j, 0, 0), n, n);
            tensor_map_t eg2(g2(i, j, k, l, 0, 0), n, n);
	    */

	    auto const em1 = M1.chip(i, 0).chip(l, 0);
	    auto const em2 = M2.chip(k, 0).chip(j, 0);
	    auto eg2 = g2.chip(i, 0).chip(j, 0).chip(k, 0).chip(l, 0);

            eg2 -= s * em1 * em2;

	    //Eigen::Tensor<value_t, 2> em2 = M2.chip(k, 0).chip(j, 0);
	    //Eigen::Tensor<value_t, 2> em2_s = em2.shuffle({1, 0});
            //eg2 -= s * em1 * em2_s;
          }
  }
}

static void G2_MM_ph_w0_last_term_Eigen_tensor_rowmajor(benchmark::State &state) {

  Eigen::Tensor<value_t, 4, Eigen::RowMajor> M1(o, o, n, n);
  Eigen::Tensor<value_t, 4, Eigen::RowMajor> M2(o, o, n, n);
  Eigen::Tensor<value_t, 6, Eigen::RowMajor> g2(o, o, o, o, n, n);

  for (auto _ : state) {

    for (int i = 0; i < o; i++)
      for (int j = 0; j < o; j++)
        for (int k = 0; k < o; k++)
          for (int l = 0; l < o; l++)
            for (int n1 = 0; n1 < n; n1++)
              for (int n2 = 0; n2 < n; n2++)
                g2(i, j, k, l, n1, n2) -=
                    s * M1(i, l, n1, n2) * M2(k, j, n2, n1);
  }
}

static void G2_MM_ph_w0_last_term_Eigen_tensor_colmajor(benchmark::State &state) {

  Eigen::Tensor<value_t, 4> M1(n, n, o, o);
  Eigen::Tensor<value_t, 4> M2(n, n, o, o);
  Eigen::Tensor<value_t, 6> g2(n, n, o, o, o, o);

  for (auto _ : state) {

    for (int i = 0; i < o; i++)
      for (int j = 0; j < o; j++)
        for (int k = 0; k < o; k++)
          for (int l = 0; l < o; l++)
            for (int n1 = 0; n1 < n; n1++)
              for (int n2 = 0; n2 < n; n2++)
                g2(n2, n1, l, k, j, i) -=
		  s * M1(n2, n1, l, i) * M2(n1, n2, j, k);
  }
}

//BENCHMARK(G2_MM_ph_w0_last_term_Eigen_gemm);
BENCHMARK(G2_MM_ph_w0_last_term_Eigen_tensor_map);
BENCHMARK(G2_MM_ph_w0_last_term_Eigen_tensor_rowmajor);
BENCHMARK(G2_MM_ph_w0_last_term_Eigen_tensor_colmajor);
