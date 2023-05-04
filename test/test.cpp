#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "ctgtt.hpp"

#include <iostream>
#include <complex>

using namespace std;

void print_matrix(complex<float> *M, size_t length, size_t cstride, size_t rstride)
{
  for (size_t j = 0; j < rstride; j++)
  {
    for (size_t i = j; i < length; i += cstride)
    {
      cout << M[i] << " ";
    }
    cout << endl;
  }
}

TEST_CASE("1x1 . 1x1 => 1x1")
{
  int d1 = 1, d2 = 1, d3 = 1;
  complex<float> *A = nullptr, *B = nullptr, *C = nullptr;
  ctgtt::utils::alloc_aligned(&A, d1 * d2);
  ctgtt::utils::alloc_aligned(&B, d2 * d3);
  ctgtt::utils::alloc_aligned(&C, d1 * d3);

  A[0] = complex<float>(2., 1.);
  B[0] = complex<float>(1.337, 0.5);

  auto tensorA = ctgtt::CTensor<float>({d1, d2}, A, ctgtt::COLUMN_MAJOR);
  auto tensorB = ctgtt::CTensor<float>({d2, d3}, B, ctgtt::COLUMN_MAJOR);
  auto tensorC = ctgtt::CTensor<float>({d1, d3}, C, ctgtt::COLUMN_MAJOR);

  SUBCASE("") {
    const complex<float> alpha = 1.0;
    const complex<float> beta = 0.0;

    ctgtt::contract(alpha, tensorA, "a", tensorB, "a", beta, tensorC, "a");

    REQUIRE(C[0].real() == doctest::Approx(2.174).epsilon(0.001));
    REQUIRE(C[0].imag() == doctest::Approx(2.337).epsilon(0.001));
  }

  SUBCASE("x 2.0") {
    const complex<float> alpha = 2.0;
    const complex<float> beta = 0.0;
    ctgtt::contract(alpha, tensorA, "a", tensorB, "a", beta, tensorC, "a");

    REQUIRE(C[0].real() == doctest::Approx(4.348).epsilon(0.001));
    REQUIRE(C[0].imag() == doctest::Approx(4.674).epsilon(0.001));
  }
}

TEST_CASE("2x2 . 1x2 => 1x2")
{
  const complex<float> alpha = 1.0;
  const complex<float> beta = 0.0;

  int d1 = 2, d2 = 2, d3 = 1;
  complex<float> *A = nullptr, *B = nullptr, *C = nullptr;
  ctgtt::utils::alloc_aligned(&A, d1 * d2);
  ctgtt::utils::alloc_aligned(&B, d2 * d3);
  ctgtt::utils::alloc_aligned(&C, d1 * d3);

  /*
    A_0 = (
      3+2i 0+i
       0-i 1
       )
  */

  A[0] = complex<float>(3., 2.);
  A[2] = complex<float>(0., 1.);
  A[1] = complex<float>(0, -1.);
  A[3] = complex<float>(1.);

  /*
    B = (
      0 + 2i
      3
     )
  */

  B[0] = complex<float>(0., 2.);
  B[1] = complex<float>(3., 0.);

  auto tensorA = ctgtt::CTensor<float>({d1, d2}, A, ctgtt::COLUMN_MAJOR);
  auto tensorB = ctgtt::CTensor<float>({d2, d3}, B, ctgtt::COLUMN_MAJOR);
  auto tensorC = ctgtt::CTensor<float>({d1, d3}, C, ctgtt::COLUMN_MAJOR);

  SUBCASE("")
  {
    ctgtt::contract(alpha, tensorA, "ab", tensorB, "b", beta, tensorC, "a");

    REQUIRE(C[0].real() == doctest::Approx(-4.).epsilon(0.001));
    REQUIRE(C[0].imag() == doctest::Approx(9.).epsilon(0.001));
    REQUIRE(C[1].real() == doctest::Approx(5.).epsilon(0.001));
    REQUIRE(C[1].imag() == doctest::Approx(0.).epsilon(0.001));
  }

  SUBCASE("")
  {
    ctgtt::contract(alpha, tensorA, "ab", tensorB, "a", beta, tensorC, "b");

    REQUIRE(C[0].real() == doctest::Approx(-4.).epsilon(0.001));
    REQUIRE(C[0].imag() == doctest::Approx(3.).epsilon(0.001));
    REQUIRE(C[1].real() == doctest::Approx(1.).epsilon(0.001));
    REQUIRE(C[1].imag() == doctest::Approx(0.).epsilon(0.001));
  }
}

TEST_CASE("2x2 . 2x2 => 2x2")
{
  const complex<float> alpha = 1.0;
  const complex<float> beta = 0.0;

  int d1 = 2, d2 = 2, d3 = 2;
  complex<float> *A = nullptr, *B = nullptr, *C = nullptr;
  ctgtt::utils::alloc_aligned(&A, d1 * d2);
  ctgtt::utils::alloc_aligned(&B, d2 * d3);
  ctgtt::utils::alloc_aligned(&C, d1 * d3);

  /*
    A_0 = (
      3+2i 0+i
       0-i 1
       )
  */

  A[0] = complex<float>(3., 2.); A[2] = complex<float>(0., 1.);
  A[1] = complex<float>(0, -1.); A[3] = complex<float>(1.);

   /*
    B_0 = (
      4          0+7i
     -0.5+0.5i 3.3
     )
  */

  B[0] = complex<float>(4.); B[2] = complex<float>(0., 7.);
  B[1] = complex<float>(-0.5,0.5); B[3] = complex<float>(3.3);

  auto tensorA = ctgtt::CTensor<float>({d1, d2}, A, ctgtt::COLUMN_MAJOR);
  auto tensorB = ctgtt::CTensor<float>({d2, d3}, B, ctgtt::COLUMN_MAJOR);
  auto tensorC = ctgtt::CTensor<float>({d1, d3}, C, ctgtt::COLUMN_MAJOR);

  SUBCASE(""){
    ctgtt::contract(alpha, tensorA, "ab", tensorB, "bc", beta, tensorC, "ac");

    REQUIRE(C[0].real() == doctest::Approx(11.5).epsilon(0.001));
    REQUIRE(C[0].imag() == doctest::Approx(7.5).epsilon(0.001));
    REQUIRE(C[1].real() == doctest::Approx(-0.5).epsilon(0.001));
    REQUIRE(C[1].imag() == doctest::Approx(-3.5).epsilon(0.001));
    REQUIRE(C[2].real() == doctest::Approx(-14).epsilon(0.001));
    REQUIRE(C[2].imag() == doctest::Approx(24.3).epsilon(0.001));
    REQUIRE(C[3].real() == doctest::Approx(10.3).epsilon(0.001));
    REQUIRE(C[3].imag() == doctest::Approx(0.).epsilon(0.001));
  }

  SUBCASE("") {
    ctgtt::contract(alpha, tensorA, "ab", tensorB, "cb", beta, tensorC, "ac");
    REQUIRE(C[0].real() == doctest::Approx(5.).epsilon(0.001));
    REQUIRE(C[0].imag() == doctest::Approx(8).epsilon(0.001));
    REQUIRE(C[1].real() == doctest::Approx(0.).epsilon(0.001));
    REQUIRE(C[1].imag() == doctest::Approx(3.).epsilon(0.001));
    REQUIRE(C[2].real() == doctest::Approx(-2.5).epsilon(0.001));
    REQUIRE(C[2].imag() == doctest::Approx(3.8).epsilon(0.001));
    REQUIRE(C[3].real() == doctest::Approx(3.8).epsilon(0.001));
    REQUIRE(C[3].imag() == doctest::Approx(0.5).epsilon(0.001));
  }

  SUBCASE("") {
    ctgtt::contract(alpha, tensorA, "ba", tensorB, "cb", beta, tensorC, "ac");
    REQUIRE(C[0].real() == doctest::Approx(19.).epsilon(0.001));
    REQUIRE(C[0].imag() == doctest::Approx(8).epsilon(0.001));
    REQUIRE(C[1].real() == doctest::Approx(0.).epsilon(0.001));
    REQUIRE(C[1].imag() == doctest::Approx(11.).epsilon(0.001));
    REQUIRE(C[2].real() == doctest::Approx(-2.5).epsilon(0.001));
    REQUIRE(C[2].imag() == doctest::Approx(-2.8).epsilon(0.001));
    REQUIRE(C[3].real() == doctest::Approx(2.8).epsilon(0.001));
    REQUIRE(C[3].imag() == doctest::Approx(-0.5).epsilon(0.001));
  }

  SUBCASE("") {
    ctgtt::contract(alpha, tensorA, "ba", tensorB, "bc", beta, tensorC, "ac");
    REQUIRE(C[0].real() == doctest::Approx(12.5).epsilon(0.001));
    REQUIRE(C[0].imag() == doctest::Approx(8.5).epsilon(0.001));
    REQUIRE(C[1].real() == doctest::Approx(-0.5).epsilon(0.001));
    REQUIRE(C[1].imag() == doctest::Approx(4.5).epsilon(0.001));
    REQUIRE(C[2].real() == doctest::Approx(-14).epsilon(0.001));
    REQUIRE(C[2].imag() == doctest::Approx(17.7).epsilon(0.001));
    REQUIRE(C[3].real() == doctest::Approx(-3.7).epsilon(0.001));
    REQUIRE(C[3].imag() == doctest::Approx(0.).epsilon(0.001));
  }
}

TEST_CASE("2x2x2 . 2x2 => 1x2")
{
  const complex<float> alpha = 1.0;
  const complex<float> beta = 0.0;

  complex<float> *A = nullptr, *B = nullptr, *C = nullptr;
  ctgtt::utils::alloc_aligned(&A, 2 * 2 * 2);
  ctgtt::utils::alloc_aligned(&B, 2 * 2);
  ctgtt::utils::alloc_aligned(&C, 2);

  /*
    A_0 = (
      i 0 
      0 i
    )

    A_1 = (
      -i 0 
      0 -i
    )
  */

  A[0] = complex<float>(0., 1.); A[2] = complex<float>(0.);
  A[1] = complex<float>(0.);     A[3] = complex<float>(0., 1.);
  
  A[4] = complex<float>(0., -1.); A[6] = complex<float>(0);
  A[5] = complex<float>(0.);      A[7] = complex<float>(0., -1.);

  /*
    B_0 = (
      4         7i
     -0.5+0.5i 3.3
     )
  */

  B[0] = complex<float>(4.); B[2] = complex<float>(0., 7.);
  B[1] = complex<float>(-0.5,0.5); B[3] = complex<float>(3.3);

  auto tensorA = ctgtt::CTensor<float>({2, 2, 2}, A, ctgtt::COLUMN_MAJOR);
  auto tensorB = ctgtt::CTensor<float>({2, 2}, B, ctgtt::COLUMN_MAJOR);
  auto tensorC = ctgtt::CTensor<float>({2}, C, ctgtt::COLUMN_MAJOR);

  SUBCASE(""){
    ctgtt::contract(alpha, tensorA, "abc", tensorB, "bc", beta, tensorC, "a");
    REQUIRE(C[0].real() == doctest::Approx(7.).epsilon(0.001));
    REQUIRE(C[0].imag() == doctest::Approx(4.).epsilon(0.001));
    REQUIRE(C[1].real() == doctest::Approx(-0.5).epsilon(0.001));
    REQUIRE(C[1].imag() == doctest::Approx(-3.8).epsilon(0.001));
  }
}