
#include <benchmark/benchmark.h>

#include <triqs/arrays.hpp>
#include <triqs/clef.hpp>
#include <triqs/cthyb/types.hpp>
#include <triqs/gfs.hpp>

using namespace triqs;
using namespace triqs::arrays;
using namespace triqs::gfs;

using imfreq_cube_mesh_t = cartesian_product<imfreq, imfreq, imfreq>;
using g2_iw_t = gf<imfreq_cube_mesh_t, tensor_valued<4>>;

using imfreq_square_mesh_t = cartesian_product<imfreq, imfreq>;
using M_iw_t = gf<imfreq_square_mesh_t, matrix_valued>;

namespace {
// Index placeholders
clef::placeholder<0> i;
clef::placeholder<1> j;
clef::placeholder<2> k;
clef::placeholder<3> l;

// Frequency placeholders
clef::placeholder<4> w;
clef::placeholder<5> n1;
clef::placeholder<6> n2;
clef::placeholder<7> n3;
}; // namespace

typedef std::complex<double> value_t;

size_t o = 4;
size_t n = 100;

int O = static_cast<int>(o);
int N = static_cast<int>(n);

double s = 1.0;
double beta = 1.0;

int n_bosonic = 1;
int n_fermionic = N;

gf_mesh<imfreq> mesh_f{beta, Fermion, n_fermionic};
gf_mesh<imfreq> mesh_b{beta, Boson, n_bosonic};

g2_iw_t::mesh_t mesh_bff{mesh_b, mesh_f, mesh_f};
M_iw_t::mesh_t mesh_ff{mesh_f, mesh_f};

static void G2_MM_ph_full_clef_recomb_gf(benchmark::State &state) {

  M_iw_t M1{mesh_ff, {O, O}};
  M_iw_t M2{mesh_ff, {O, O}};
  g2_iw_t g2{mesh_bff, {O, O, O, O}};

  for (auto _ : state) {

    g2(w, n1, n2)(i, j, k, l)
        << g2(w, n1, n2)(i, j, k, l) +
               s * M1(n1, n1 + w)(i, j) * M2(n2 + w, n2)(k, l);

    g2(w, n1, n2)(i, j, k, l)
        << g2(w, n1, n2)(i, j, k, l) -
               s * M1(n1, n2)(i, l) * M2(n2 + w, n1 + w)(k, j);
  }
}

static void G2_MM_ph_full_loop_recomb_gf(benchmark::State &state) {

  M_iw_t M1{mesh_ff, {O, O}};
  M_iw_t M2{mesh_ff, {O, O}};
  g2_iw_t g2{mesh_bff, {O, O, O, O}};

  for (auto _ : state) {

    for (auto const &w : mesh_b)
      for (auto const &n1 : mesh_f)
        for (auto const &n2 : mesh_f)
          for (auto const i : range(g2.target_shape()[0]))
            for (auto const j : range(g2.target_shape()[1]))
              for (auto const k : range(g2.target_shape()[2]))
                for (auto const l : range(g2.target_shape()[3]))
                  g2[w, n1, n2](i, j, k, l) +=
                      s * M1[n1, n1 + w](i, j) * M2[n2 + w, n2](k, l);

    for (auto const &w : mesh_b)
      for (auto const &n1 : mesh_f)
        for (auto const &n2 : mesh_f)
          for (auto const i : range(g2.target_shape()[0]))
            for (auto const j : range(g2.target_shape()[1]))
              for (auto const k : range(g2.target_shape()[2]))
                for (auto const l : range(g2.target_shape()[3]))
                  g2[w, n1, n2](i, j, k, l) -=
                      s * M1[n1, n2](i, l) * M2[n2 + w, n1 + w](k, j);
  }
}

static void G2_MM_ph_w0_loop_recomb_gf(benchmark::State &state) {

  M_iw_t M1{mesh_ff, {O, O}};
  M_iw_t M2{mesh_ff, {O, O}};
  g2_iw_t g2{mesh_bff, {O, O, O, O}};

  for (auto _ : state) {

    for (auto const &n1 : mesh_f)
      for (auto const &n2 : mesh_f)
        for (auto const i : range(g2.target_shape()[0]))
          for (auto const j : range(g2.target_shape()[1]))
            for (auto const k : range(g2.target_shape()[2]))
              for (auto const l : range(g2.target_shape()[3]))
                g2[0, n1, n2](i, j, k, l) +=
                    s * M1[n1, n1](i, j) * M2[n2, n2](k, l);

      for (auto const &n1 : mesh_f)
        for (auto const &n2 : mesh_f)
          for (auto const i : range(g2.target_shape()[0]))
            for (auto const j : range(g2.target_shape()[1]))
              for (auto const k : range(g2.target_shape()[2]))
                for (auto const l : range(g2.target_shape()[3]))
                  g2[0, n1, n2](i, j, k, l) -=
                      s * M1[n1, n2](i, l) * M2[n2, n1](k, j);
  }
}

static void G2_MM_ph_w0_loop_recomb_gf_rev_loop_order(benchmark::State &state) {

  M_iw_t M1{mesh_ff, {O, O}};
  M_iw_t M2{mesh_ff, {O, O}};
  g2_iw_t g2{mesh_bff, {O, O, O, O}};

  for (auto _ : state) {

    for (auto const i : range(g2.target_shape()[0]))
      for (auto const j : range(g2.target_shape()[1]))
        for (auto const k : range(g2.target_shape()[2]))
          for (auto const l : range(g2.target_shape()[3]))
            for (auto const &n2 : mesh_f)
              for (auto const &n1 : mesh_f)
                g2[0, n1, n2](i, j, k, l) +=
                    s * M1[n1, n1](i, j) * M2[n2, n2](k, l);

      for (auto const i : range(g2.target_shape()[0]))
        for (auto const j : range(g2.target_shape()[1]))
          for (auto const k : range(g2.target_shape()[2]))
            for (auto const l : range(g2.target_shape()[3]))
              for (auto const &n2 : mesh_f)
                for (auto const &n1 : mesh_f)
                  g2[0, n1, n2](i, j, k, l) -=
                      s * M1[n1, n2](i, l) * M2[n2, n1](k, j);
  }
}

static void G2_MM_ph_w0_loop_recomb_gf_memlayout(benchmark::State &state) {

  M_iw_t M1{mesh_ff, {O, O}, make_memory_layout(3, 2, 1, 0)};
  M_iw_t M2{mesh_ff, {O, O}, make_memory_layout(3, 2, 1, 0)};
  g2_iw_t g2{mesh_bff, {O, O, O, O}, make_memory_layout(6, 5, 4, 3, 0, 2, 1)};

  for (auto _ : state) {

      for (auto const k : range(g2.target_shape()[2]))
        for (auto const j : range(g2.target_shape()[1]))
          for (auto const i : range(g2.target_shape()[0]))
            for (auto const l : range(g2.target_shape()[3]))
              for (auto const &n2 : mesh_f)
                for (auto const &n1 : mesh_f)
                  g2[0, n1, n2](i, j, k, l) +=
                      s * M1[n1, n1](i, j) * M2[n2, n2](k, l);

    for (auto const l : range(g2.target_shape()[3]))
      for (auto const k : range(g2.target_shape()[2]))
        for (auto const j : range(g2.target_shape()[1]))
          for (auto const i : range(g2.target_shape()[0]))
            for (auto const &n2 : mesh_f)
              for (auto const &n1 : mesh_f)
                g2[0, n1, n2](i, j, k, l) -=
                    s * M1[n1, n2](i, l) * M2[n2, n1](k, j);
  }
}

static void G2_MM_ph_w0_loop_rearrange_recomb_arr(benchmark::State &state) {

  array<value_t, 4> M1{o, o, n, n};
  array<value_t, 4> M2{o, o, n, n};
  array<value_t, 6> g2{o, o, o, o, n, n};

  for (auto _ : state) {

    for (auto const i : range(g2.shape()[0]))
      for (auto const j : range(g2.shape()[1]))
        for (auto const k : range(g2.shape()[2]))
          for (auto const l : range(g2.shape()[3]))
            for (auto const n1 : range(g2.shape()[4]))
              for (auto const n2 : range(g2.shape()[5]))
                g2(i, j, k, l, n1, n2) +=
                    s * M1(i, j, n1, n1) * M2(k, l, n2, n2);

    for (auto const i : range(g2.shape()[0]))
      for (auto const j : range(g2.shape()[1]))
        for (auto const k : range(g2.shape()[2]))
          for (auto const l : range(g2.shape()[3]))
            for (auto const n1 : range(g2.shape()[4]))
              for (auto const n2 : range(g2.shape()[5]))
                g2(i, j, k, l, n1, n2) -=
                    s * M1(i, l, n1, n2) * M2(k, j, n2, n1);
  }
}

static void G2_MM_ph_w0_loop_rearrange_v2_recomb_arr(benchmark::State &state) {

  array<value_t, 4> M1{o, o, n, n};
  array<value_t, 4> M2{o, o, n, n};
  array<value_t, 6> g2{o, o, o, o, n, n};

  for (auto _ : state) {

    for (auto const i : range(g2.shape()[0]))
      for (auto const j : range(g2.shape()[1]))
        for (auto const k : range(g2.shape()[2]))
          for (auto const l : range(g2.shape()[3])) {

            for (auto const n1 : range(g2.shape()[4]))
              for (auto const n2 : range(g2.shape()[5]))
                g2(i, j, k, l, n1, n2) +=
                    s * M1(i, j, n1, n1) * M2(k, l, n2, n2);

            for (auto const n1 : range(g2.shape()[4]))
              for (auto const n2 : range(g2.shape()[5]))
                g2(i, j, k, l, n1, n2) -=
                    s * M1(i, l, n1, n2) * M2(k, j, n2, n1);
          }
  }
}

static void G2_MM_ph_w0_loop_rearrange_v3_recomb_arr(benchmark::State &state) {

  typedef std::complex<double> value_t;

  array<value_t, 4> M1{o, o, n, n};
  array<value_t, 4> M2{o, o, n, n};
  array<value_t, 6> g2{o, o, o, o, n, n};

  for (auto _ : state) {

    for (auto const i : range(g2.shape()[0]))
      for (auto const j : range(g2.shape()[1]))
        for (auto const k : range(g2.shape()[2]))
          for (auto const l : range(g2.shape()[3]))
            for (auto const n1 : range(g2.shape()[4]))
              for (auto const n2 : range(g2.shape()[5])) {
                g2(i, j, k, l, n1, n2) +=
                    s * M1(i, j, n1, n1) * M2(k, l, n2, n2);
                g2(i, j, k, l, n1, n2) -=
                    s * M1(i, l, n1, n2) * M2(k, j, n2, n1);
              }
  }
}

static void G2_MM_ph_w0_last_term(benchmark::State &state) {

  array<value_t, 4> M1{o, o, n, n};
  array<value_t, 4> M2{o, o, n, n};
  array<value_t, 6> g2{o, o, o, o, n, n};

  for (auto _ : state) {

    for (auto const i : range(g2.shape()[0]))
      for (auto const j : range(g2.shape()[1]))
        for (auto const k : range(g2.shape()[2]))
          for (auto const l : range(g2.shape()[3]))
            for (auto const n1 : range(g2.shape()[4]))
              for (auto const n2 : range(g2.shape()[5])) {
                g2(i, j, k, l, n1, n2) -=
                    s * M1(i, l, n1, n2) * M2(k, j, n2, n1);
              }
  }
}

/*
static void G2_MM_ph_w0_last_term_gemm(benchmark::State &state) {

  array<value_t, 4> M1{o, o, n, n};
  array<value_t, 4> M2{o, o, n, n};
  array<value_t, 6> g2{o, o, o, o, n, n};

  array<value_t, 1> m1{n};
  array<value_t, 1> m2{n};

  for (auto _ : state) {

    for (auto const i : range(g2.shape()[0]))
      for (auto const j : range(g2.shape()[1]))
        for (auto const k : range(g2.shape()[2]))
          for (auto const l : range(g2.shape()[3])) {

            // This is a matmul(M1, M2.T) (BLAS L3 gemm)
            //g2(i, j, k, l, n1, n2) -= s * M1(i, l, n1, n2) * M2(k, j, n2, n1);

            auto _ = range{};

            matrix_view<value_t> m1 = M1(i, j, _, _);
            matrix_view<value_t> m2 = M2(k, l, _, _);
            matrix_view<value_t> g2_m = g2(i, j, k, l, _, _);

            g2_m -= s * m1 * m2.transpose();

          }
  }
}
*/

BENCHMARK(G2_MM_ph_full_clef_recomb_gf);
BENCHMARK(G2_MM_ph_full_loop_recomb_gf);

BENCHMARK(G2_MM_ph_w0_loop_recomb_gf);
BENCHMARK(G2_MM_ph_w0_loop_recomb_gf_rev_loop_order);
BENCHMARK(G2_MM_ph_w0_loop_recomb_gf_memlayout);
BENCHMARK(G2_MM_ph_w0_loop_rearrange_recomb_arr);
BENCHMARK(G2_MM_ph_w0_loop_rearrange_v2_recomb_arr);
BENCHMARK(G2_MM_ph_w0_loop_rearrange_v3_recomb_arr);

BENCHMARK(G2_MM_ph_w0_last_term);
// BENCHMARK(G2_MM_ph_w0_last_term_gemm);
