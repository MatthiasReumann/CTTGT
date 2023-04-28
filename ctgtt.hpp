#include <numeric>
#include <complex>
#include <unordered_set>
#include <stdlib.h>
#include <string>
#include "utils.hpp"
#include "hptt.h"
#include "blis.h"
#include "marray.hpp"

namespace ctgtt
{
  template <class T>
  using CTensor = MArray::marray_view<std::complex<T>>;

  constexpr MArray::layout COLUMN_MAJOR = MArray::COLUMN_MAJOR;

  template <class T>
  void contract(std::complex<T> alpha,
                const CTensor<T> A, std::string labelsA,
                const CTensor<T> B, std::string labelsB,
                std::complex<T> beta,
                CTensor<T> C, std::string labelsC)
  {
    utils::IndexBundle I, pA, pB, J;
    std::vector<int> permA, permB, permC(labelsC.length());
    std::complex<T> *A_ = nullptr, *B_ = nullptr, *C_ = nullptr;

    // Step 1: Find Index Bundles I, J, P
    utils::find_index_bundles(labelsA, A.lengths(), labelsB, B.lengths(), &I, &pA, &pB, &J);

    // Step 2: Prepare transposition
    permA.reserve(I.size());
    permB.reserve(J.size());

    permA.insert(permA.end(), I.indices.begin(), I.indices.end());
    permA.insert(permA.end(), pA.indices.begin(), pA.indices.end());

    permB.insert(permB.end(), pB.indices.begin(), pB.indices.end());
    permB.insert(permB.end(), J.indices.begin(), J.indices.end());

    utils::find_c_permutation(labelsC, &permC, &I, &J);

    utils::alloc_aligned(&A_, A.size(A.lengths()));
    utils::alloc_aligned(&B_, B.size(B.lengths()));
    utils::alloc_aligned(&C_, C.size(C.lengths()));

    // Step 3: Transpose A, B
    utils::transpose(A.data(), A_, permA, A.lengths());
    utils::transpose(B.data(), B_, permB, B.lengths());

    // Step 4: GEMM and transpose C
    utils::gemm(alpha, beta, A_, B_, C_, I.size(), J.size(), pA.size());
    utils::transpose(C_, C.data(), permC, C.lengths());

    free(A_);
    free(B_);
    free(C_);
  }
}