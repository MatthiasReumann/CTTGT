# CTTGT
Complex Tensor Contractions via TTGT. Transposed with the help of HPTT. Complex matrix multiplication is achieved with BLIS.

## Dependencies 

* [BLIS](https://github.com/flame/blis)
* [HPTT](https://github.com/springer13/hptt)
* [MArray](https://github.com/devinamatthews/marray)

## Usage

```cpp
#include <complex>
#include "ctgtt.hpp"

const complex<double> ALPHA = 1.0;
const complex<double> BETA = 0.0;

int main() {
    int d1 = 200, d2 = 200, d3 = 200;
    complex<double> *A = nullptr, *B = nullptr, *C = nullptr;
    ctgtt::utils::alloc_aligned(&A, d1 * d2);
    ctgtt::utils::alloc_aligned(&B, d2 * d3);
    ctgtt::utils::alloc_aligned(&C, d1 * d3);

    /* 
        set data ...
    */

    auto tensorA = ctgtt::CTensor<double>({d1, d2}, A, ctgtt::COLUMN_MAJOR);
    auto tensorB = ctgtt::CTensor<double>({d2, d3}, B, ctgtt::COLUMN_MAJOR);
    auto tensorC = ctgtt::CTensor<double>({d1, d3}, C, ctgtt::COLUMN_MAJOR);

    ctgtt::contract(ALPHA, tensorA, "ab", tensorB, "cb", BETA, tensorC, "ac");
    return 0;
}

