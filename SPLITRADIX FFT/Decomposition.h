#pragma once
#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include <cstdint>
#include <optional>
#include <array>


using uint = std::uint32_t;

extern constexpr std::array<uint, 22> Coeffs{
		2U, 4U, 8U, 16U, 32U, 64U, 128U,
		256U, 3U, 5U, 7U, 11U, 13U, 17U,
		6U, 10U, 12U, 20U, 24U, 40U, 80U, 100U
};

struct Decomposition {

	uint N{};
	uint lrows{};
	uint lcols{};
	uint InRows{};
	uint InCols{};
	uint M_size{};

	explicit Decomposition(uint _N);
	
	void print() const;

private:
	struct DecompRes {

		uint inRows{};
		uint inCols{};
		uint mSize{};
	};

	bool canbeDecompose(uint n);
	std::optional<DecompRes> tryDecomp(uint rows, uint cols, uint bDiff);
	void findLDecompose();
	uint findBestDivisior(uint n);

};

#endif 
