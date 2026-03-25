#pragma once
#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include <vector>
#include <cstdint>
#include <iostream>
#include <array>

using uint = std::uint32_t;

struct Decomposition {

    uint N{};
    std::vector<uint> factors;

    explicit Decomposition(uint n);

    void print() const;

private:

    void factorize();
    void optimize();

    inline static constexpr std::array<uint, 8> PreferredFactors{
        8, 10, 6, 4, 2, 3, 5, 7
    };
};

#endif