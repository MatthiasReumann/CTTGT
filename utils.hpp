#include <numeric>
#include <complex>
#include <unordered_set>
#include <stdlib.h>
#include <string>
#include "hptt.h"
#include "blis.h"
#include "marray.hpp"

namespace ctgtt
{
  namespace internal
  {
    template <typename T, num_t BLIS_TYPE>
    void gemm(
        std::complex<T> alpha, std::complex<T> beta,
        std::complex<T> *A, std::complex<T> *B, std::complex<T> *C,
        int nI, int nJ, int nP)
    {
      obj_t blis_A, blis_B, blis_C, blis_alpha, blis_beta;

      bli_obj_create_1x1(BLIS_TYPE, &blis_alpha);
      bli_setsc(alpha.real(), alpha.imag(), &blis_alpha);

      bli_obj_create_1x1(BLIS_TYPE, &blis_beta);
      bli_setsc(beta.real(), beta.imag(), &blis_beta);

      // nI x nP matrix (column major)
      bli_obj_create_with_attached_buffer(BLIS_TYPE, nI, nP, A, 1, nI, &blis_A);

      // nP x nJ matrix (column major)
      bli_obj_create_with_attached_buffer(BLIS_TYPE, nP, nJ, B, 1, nP, &blis_B);

      // nI x nJ matrix (column major)
      bli_obj_create_with_attached_buffer(BLIS_TYPE, nI, nJ, C, 1, nI, &blis_C);
      bli_setm(&BLIS_ZERO, &blis_C);
      bli_gemm(&blis_alpha, &blis_A, &blis_B, &blis_beta, &blis_C);
    }
  }

  namespace utils
  {
    const auto HPTT_SELECTION_METHOD = hptt::ESTIMATE;

    class IndexBundle
    {
    public:
      void push_back(char label, int idx, int length)
      {
        this->labels.push_back(label);
        this->indices.push_back(idx);
        this->lengths.push_back(length);
      }

      void merge(IndexBundle *other)
      {
        this->labels.insert(this->labels.end(), other->labels.begin(), other->labels.end());
        this->indices.insert(this->indices.end(), other->indices.begin(), other->indices.end());
        this->lengths.insert(this->lengths.end(), other->lengths.begin(), other->lengths.end());
      }

      const int size()
      {
        return std::accumulate(this->lengths.begin(), this->lengths.end(), 1, std::multiplies<int>());
      }

      std::vector<char> labels;
      std::vector<int> indices;
      std::vector<unsigned long> lengths;
    };

    template <typename T>
    void alloc_aligned(T **ptr, size_t n)
    {
      // TODO: Memory Alignment?
      if (posix_memalign((void **)ptr, 32, n * sizeof(T)))
      {
        std::throw_with_nested(std::bad_alloc());
      }
    }

    template <unsigned long NDim>
    void find_index_bundles(
        std::string labelsA, const MArray::short_vector<long, NDim> sizesA,
        std::string labelsB, const MArray::short_vector<long, NDim> sizesB,
        IndexBundle *I, IndexBundle *pA, IndexBundle *pB, IndexBundle *J)
    {
      bool in_I;
      std::unordered_set<char> setB{labelsB.cbegin(), labelsB.cend()};

      for (int i = 0; i < labelsA.length(); i++)
      {
        in_I = false;
        for (int j = 0; j < labelsB.length(); j++)
        {
          if (labelsA.at(i) == labelsB.at(j))
          {
            in_I = true;
            pA->push_back(labelsA.at(i), i, sizesA[i]);
            pB->push_back(labelsA.at(i), j, sizesB[j]);
            setB.erase(labelsA.at(i));
          }
        }

        if (!in_I)
          I->push_back(labelsA.at(i), i, sizesA[i]);
      }

      for (int j = 0; (j < labelsB.length() && setB.count(labelsB.at(j)) > 0); j++)
        J->push_back(labelsB.at(j), j, sizesB[j]);
    }

    template <typename T, unsigned long NDim>
    void transpose(std::complex<T> *M, std::complex<T> *M_, std::vector<int> perm,
                   const MArray::short_vector<long, NDim> size)
    {
      std::vector<int> sizeA(size.begin(), size.end());
      hptt::create_plan(perm.data(), perm.size(),
                        1., M, sizeA.data(), NULL,
                        0., M_, NULL,
                        HPTT_SELECTION_METHOD, 4)
          ->execute();
    }

    void gemm(std::complex<float> alpha, std::complex<float> beta,
              std::complex<float> *A, std::complex<float> *B, std::complex<float> *C,
              int nI, int nJ, int nP)
    {
      internal::gemm<float, BLIS_SCOMPLEX>(alpha, beta, A, B, C, nI, nJ, nP);
    }

    void gemm(std::complex<double> alpha, std::complex<double> beta,
              std::complex<double> *A, std::complex<double> *B, std::complex<double> *C,
              int nI, int nJ, int nP)
    {
      internal::gemm<double, BLIS_DCOMPLEX>(alpha, beta, A, B, C, nI, nJ, nP);
    }

    void find_c_permutation(std::string labels, std::vector<int> *perm, const IndexBundle *I, const IndexBundle *J)
    {
      for (int j = 0; j < labels.length(); j++)
      {
        for (int i = 0; i < I->labels.size(); i++)
          if (I->labels.at(i) == labels.at(j))
            perm->at(j) = i;
        for (int i = 0; i < J->labels.size(); i++)
          if (J->labels.at(i) == labels.at(j))
            perm->at(j) = I->labels.size() + i;
      }
    }
  }
}