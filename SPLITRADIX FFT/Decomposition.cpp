#include "Decomposition.h"
#include <iostream>
#include <cmath>
#include <algorithm>

Decomposition::Decomposition(uint _N) : N(_N) {
    findLDecompose();
}

void Decomposition::print() const {
    std::cout << "N = " << N << " decomposed into "
        << lrows << " x " << lcols << std::endl;
    std::cout << "Transposed matrix: "
        << InRows << " x " << InCols << std::endl;
    std::cout << "M_size = " << M_size << std::endl;
}

bool Decomposition::canbeDecompose(uint n) {
    if (n <= 1) return false;

    for (uint coeff : Coeffs) {
        if (coeff > n) continue;
        if (n % coeff == 0) return true;
    }
    return false;
}

uint Decomposition::findBestDivisior(uint n) {
    if (n <= 1) return 1;

    uint bestDivisor = 1;
    int minRemainder = n - 1;

    for (uint coeff : Coeffs) {
        if (coeff > n) continue;

        uint divisor = coeff;
        uint quotient = n / divisor;
        uint remainder = n % divisor;

        if (remainder == 0) {
            return divisor;
        }

        int currRemainder = std::min(static_cast<int>(remainder),
            static_cast<int>(divisor - remainder));

        if (currRemainder < minRemainder) {
            minRemainder = currRemainder;
            bestDivisor = divisor;
        }
    }

    return bestDivisor;
}

std::optional<Decomposition::DecompRes>
Decomposition::tryDecomp(uint rows, uint cols, uint bDiff) {

    if (rows == 0 || cols == 0) return std::nullopt;

    uint mSize = rows * cols;
    if (mSize < N) return std::nullopt;

    uint diff = mSize - N;
    if (diff > bDiff) return std::nullopt;

    DecompRes result;
    result.inRows = cols;
    result.inCols = rows;
    result.mSize = mSize;

    return result;
}

void Decomposition::findLDecompose() {
    uint sqrtN = static_cast<uint>(std::sqrt(static_cast<double>(N)));

    lrows = 1;
    lcols = N;
    InRows = lcols;
    InCols = lrows;
    M_size = lrows * lcols;

    uint bestDiff = M_size - N;

    for (uint i = 1; i <= sqrtN; ++i) {
        if (N % i == 0) {
            uint row = i;
            uint col = N / i;

            auto result = tryDecomp(row, col, bestDiff);
            if (result.has_value()) {
                lrows = row;
                lcols = col;
                InRows = result->inRows;
                InCols = result->inCols;
                M_size = result->mSize;
                bestDiff = M_size - N;
            }

            result = tryDecomp(col, row, bestDiff);
            if (result.has_value()) {
                lrows = col;
                lcols = row;
                InRows = result->inRows;
                InCols = result->inCols;
                M_size = result->mSize;
                bestDiff = M_size - N;
            }
        }
    }

    if (bestDiff > 0 && canbeDecompose(N)) {
        uint divisor = findBestDivisior(N);

        if (divisor > 1) {
            uint row = divisor;
            uint col = (N + divisor - 1) / divisor;

            auto result = tryDecomp(row, col, bestDiff);
            if (result.has_value() && (result->mSize - N) < bestDiff) {
                lrows = row;
                lcols = col;
                InRows = result->inRows;
                InCols = result->inCols;
                M_size = result->mSize;
            }
        }
    }
}