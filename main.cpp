#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <random>
// SSE4
#include <immintrin.h>

// Given three arrays containing a, x and b values respectively, compute the equation: y = a * x + b
// For each y:
    // Store 1  if it's positive
    // Store 0  if it's zero
    // Store -1 if it's negative
// into the output array using SIMD instructions

// Number of lanes in the SIMD instruction set that's in use
// 4 for SSE     (128 / 32)
// 8 for AVX     (256 / 32)
// 16 for AVX512 (512 / 32)
static constexpr auto   g_scSIMDWidth = 4u;

// 16-byte alignment for SSE, 32-byte for AVX is required!
static constexpr auto   g_scSIMDDataAlignment = 16u;

// Input size
static auto             g_sNumInput = 1'000'000u;

void RandomizeInputData(int32_t* pAValues, int32_t* pXValues, int32_t* pBValues);

void ComputeResultsSIMD4(const int32_t* pAValues, const int32_t* pXValues, const int32_t* pBValues, int32_t* pResults);
void ComputeResultsScalar(const int32_t* pAValues, const int32_t* pXValues, const int32_t* pBValues, int32_t* pResults);

bool ValidateOutput(int32_t* pResultsSIMD, int32_t* pResultsScalar);

// Allocate memory aligned to g_scSIMDDataAlignment boundary
#ifdef _MSC_VER
    #define ALIGNED_ALLOC(size, type) (reinterpret_cast<type*>(_aligned_malloc(size * sizeof(type), g_scSIMDDataAlignment)))
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #define ALIGNED_ALLOC(size, type) (reinterpret_cast<type*>(aligned_alloc(g_scSIMDDataAlignment, size * sizeof(type))))
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

int main(int argc, char* pArgv[])
{
    // y = a * x + b
    // if y > 0
        // result = 1
    // else if y = 0
        // result = 0
    // else
        // result = -1

    // Assume input size is divisible by SIMD width
    g_sNumInput = (argc > 1) ? atoi(pArgv[1]) : g_sNumInput;

    // Allocate input and output arrays aligned accordingly for SIMD in use

    // Input arrays
    int32_t* pAValues = ALIGNED_ALLOC(g_sNumInput, int32_t);
    int32_t* pXValues = ALIGNED_ALLOC(g_sNumInput, int32_t);
    int32_t* pBValues = ALIGNED_ALLOC(g_sNumInput, int32_t);

    // Output array for SIMD
    int32_t* pOutputValuesSIMD = ALIGNED_ALLOC(g_sNumInput, int32_t);

    // Output array to validate SIMD results with scalar code
    int32_t* pOutputValuesScalar = ALIGNED_ALLOC(g_sNumInput, int32_t);

    RandomizeInputData(pAValues, pXValues, pBValues);

    auto simdBegin = std::chrono::high_resolution_clock::now();

    // Do calculations with SSE4
    ComputeResultsSIMD4(pAValues, pXValues, pBValues, pOutputValuesSIMD);

    auto simdEnd = std::chrono::high_resolution_clock::now();

    // Do calculations with scalar
    ComputeResultsScalar(pAValues, pXValues, pBValues, pOutputValuesScalar);

    auto scalarEnd = std::chrono::high_resolution_clock::now();

    if (ValidateOutput(pOutputValuesSIMD, pOutputValuesScalar))
    {
        auto simdRuntime = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(simdEnd - simdBegin).count();
        auto scalarRuntime = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(scalarEnd - simdEnd).count();

        printf("SUCCESS!\n");
        printf("SIMD:   %.3f ms\n", simdRuntime);
        printf("Scalar: %.3f ms\n", scalarRuntime);
    }
    else
    {
        printf("FAIL: SIMD and scalar results don't match!\n");
    }

    // Clean up
    ALIGNED_FREE(pAValues);
    ALIGNED_FREE(pXValues);
    ALIGNED_FREE(pBValues);
    ALIGNED_FREE(pOutputValuesSIMD);
    ALIGNED_FREE(pOutputValuesScalar);

    return 0;
}

// Fill input data with random values to ensure correctness
void RandomizeInputData(
    int32_t* pAValues,
    int32_t* pXValues,
    int32_t* pBValues)
{
    int32_t minRand = INT32_MIN;
    int32_t maxRand = INT32_MAX;

    std::default_random_engine rndGen;
    std::uniform_int_distribution<int32_t> rndDist(minRand, maxRand);

    for (uint32_t i = 0; i < g_sNumInput; i++)
    {
        pAValues[i] = rndDist(rndGen);
        pXValues[i] = rndDist(rndGen);
        pBValues[i] = rndDist(rndGen);
    }
}

// Calculate results using SSE4
void ComputeResultsSIMD4(
    const int32_t* pAValues,
    const int32_t* pXValues,
    const int32_t* pBValues,
    int32_t* pResults)
{
    // y = a * x + b
    // r = y > 0 ? 1 : (y = 0 ? 0 : -1)

    // Iterate over input range where stride equals SIMD width
    for (uint32_t i = 0; i < g_sNumInput; i += g_scSIMDWidth)
    {
        // Load next four consecutive a values (an+0 an+1 an+2 an+3)
        __m128i sseA4 = _mm_load_si128(reinterpret_cast<const __m128i*>(pAValues + i));

        // Load next four consecutive x values (xn+0 xn+1 xn+2 xn+3)
        __m128i sseX4 = _mm_load_si128(reinterpret_cast<const __m128i*>(pXValues + i));

        // Load next four consecutive b values (bn+0 bn+1 bn+2 bn+3)
        __m128i sseB4 = _mm_load_si128(reinterpret_cast<const __m128i*>(pBValues + i));

        // Notice: _mm_load_* require memory to be 16-byte aligned for SSE, 32-byte for AVX!

        // Compute y = a * x for four integers at a time using 4 SIMD lanes
        __m128i sseY4 = _mm_mullo_epi32(sseA4, sseX4);

        // Compute y += b <==> y = a * x + b
        sseY4 = _mm_add_epi32(sseY4, sseB4);

        // Generate masks based on scalar code branch conditions

        // Mask for result > 0 => for each lane mask[lane] = result > 0
        __m128i sseMaskPositive = _mm_cmpgt_epi32(sseY4, _mm_set1_epi32(0));

        // Mask for result = 0 => for each lane mask[lane] = result = 0
        __m128i sseMaskZero = _mm_cmpeq_epi32(sseY4, _mm_set1_epi32(0));

        // Mask for result < 0 => for each lane mask[lane] = result < 0
        __m128i sseMaskNegative = _mm_cmplt_epi32(sseY4, _mm_setzero_si128()); // _mm_setzero_si128() == _mm_set1_epi32()

#if 1
        // We compose final result by AND'ing masks with the values we'd like to store (1s, 0s or -1s depending on the resultant mask)
        __m128i sseResultPos = _mm_and_si128(_mm_set1_epi32(1), sseMaskPositive);   // Result = for each lane result[lane] = mask[lane] != 0 ? 1 : 0
        __m128i sseResultNeg = _mm_and_si128(_mm_set1_epi32(-1), sseMaskNegative);  // Result = for each lane result[lane] = mask[lane] != 0 ? -1 : 0
        __m128i sseResult = _mm_or_si128(sseResultPos, sseResultNeg);               // Value of 0 will be assigned on lanes where result was neither > 0 nor < 0 (hence result == 0) 
        _mm_store_si128(reinterpret_cast<__m128i*>(pResults + i), sseResult);
#else
        // Alternative would be to apply the generated masks on stores directly
        // where result for each lane will be conditionally written out when the mask for given lane is non-zero
        _mm_maskmoveu_si128(_mm_set1_epi32(1), sseMaskPositive, reinterpret_cast<char*>(pResults + i));
        _mm_maskmoveu_si128(_mm_set1_epi32(0), sseMaskZero, reinterpret_cast<char*>(pResults + i));
        _mm_maskmoveu_si128(_mm_set1_epi32(-1), sseMaskNegative, reinterpret_cast<char*>(pResults + i));
#endif
    }
}

// Calculate results using scalar code
void ComputeResultsScalar(
    const int32_t* pAValues,
    const int32_t* pXValues,
    const int32_t* pBValues,
    int32_t* pResults)
{
    // y = a * x + b
    // r = y > 0 ? 1 : (y = 0 ? 0 : -1)

    // Iterate over input range one by one
    for (uint32_t i = 0; i < g_sNumInput; i++)
    {
        // Load  a, x and b values
        int32_t a = pAValues[i];
        int32_t x = pXValues[i];
        int32_t b = pBValues[i];

        // Compute y = a * x + b
        int32_t y = a * x + b;

        int result;
        if (y > 0)
        {
            result = 1;
        }
        else if (y == 0)
        {
            result = 0;
        }
        else
        {
            result = -1;
        }

        // Store result for single computation
        pResults[i] = result;
    }
}

// Check that output from SIMD and scalar match exactly
bool ValidateOutput(int32_t* pResultsSIMD, int32_t* pResultsScalar)
{
    return memcmp(pResultsSIMD, pResultsScalar, sizeof(int32_t) * g_sNumInput) == 0;
}