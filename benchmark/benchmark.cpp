#include <iostream>
#include <chrono>
#include <complex>
#include "ctgtt.hpp"

using namespace std;

const int N = 1000;

int main()
{

  for(int i = 0; i < 11; i++) {
    int d1 = pow(2, i), d2 = pow(2, i), d3 = pow(2, i);
    complex<double> *A = nullptr, *B = nullptr, *C = nullptr;
    ctgtt::utils::alloc_aligned(&A, d1 * d2);
    ctgtt::utils::alloc_aligned(&B, d2 * d3);
    ctgtt::utils::alloc_aligned(&C, d1 * d3);

    auto tensorA = ctgtt::CTensor<double>({d1, d2}, A, ctgtt::COLUMN_MAJOR);
    auto tensorB = ctgtt::CTensor<double>({d2, d3}, B, ctgtt::COLUMN_MAJOR);
    auto tensorC = ctgtt::CTensor<double>({d1, d3}, C, ctgtt::COLUMN_MAJOR);
    
    const complex<double> alpha = 1.0;
    const complex<double> beta = 0.0;

    vector<float> time(N);
    for (uint i = 0; i < N; i++)
    {
      auto t0 = chrono::high_resolution_clock::now();
      ctgtt::contract(alpha, tensorA, "ab",
                      tensorB, "cb",
                      beta, tensorC, "ac");
      auto t1 = chrono::high_resolution_clock::now();
      time[i] = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    }

    cout << "TEST: d1="<<d1 << " d2="<<d2 << " d3=" << d3 << endl;
    cout << "Minimum: " << *min_element(time.begin(), time.end()) << " ms" << endl;
    cout << "Average: " << accumulate(time.begin(), time.end(), 0.0) * 1.0 / N << " ms" << endl;
  }
}