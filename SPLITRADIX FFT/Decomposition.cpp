#include "Decomposition.h"
#include <unordered_map>

Decomposition::Decomposition(uint n) : N(n) {
    factorize();
    optimize();
}

void Decomposition::factorize() {

    uint n = N;
    factors.clear();

    while (n % 2 == 0) {
        factors.push_back(2);
        n /= 2;
    }

    for (uint d = 3; d * d <= n; d += 2) {
        while (n % d == 0) {
            factors.push_back(d);
            n /= d;
        }
    }

    if (n > 1)
        factors.push_back(n);
}



void Decomposition::optimize() {

    std::unordered_map<uint, uint> cnt;

    for (auto f : factors)
        cnt[f]++;

    std::vector<uint> result;

    auto can = [&](uint p, uint k) {
        return cnt[p] >= k;
        };

    auto use = [&](uint p, uint k) {
        cnt[p] -= k;
        };

    while (true) {

        if (can(2, 3)) {
            use(2, 3);
            result.push_back(8);
        }
        else if (can(2, 1) && can(5, 1)) {
            use(2, 1);
            use(5, 1);
            result.push_back(10);
        }
        else if (can(2, 1) && can(3, 1)) {
            use(2, 1);
            use(3, 1);
            result.push_back(6);
        }
        else if (can(2, 2)) {
            use(2, 2);
            result.push_back(4);
        }
        else {
            break;
        }
    }


    for (auto& [p, c] : cnt) {
        for (uint i = 0; i < c; ++i)
            result.push_back(p);
    }


    std::vector<uint> ordered;
    std::vector<bool> used(result.size(), false);

    for (auto pref : PreferredFactors) {
        for (size_t i = 0; i < result.size(); ++i) {
            if (!used[i] && result[i] == pref) {
                ordered.push_back(result[i]);
                used[i] = true;
            }
        }
    }

    for (size_t i = 0; i < result.size(); ++i) {
        if (!used[i])
            ordered.push_back(result[i]);
    }

    factors = ordered;
}



void Decomposition::print() const {

    std::cout << "N = " << N << " = ";

    for (size_t i = 0; i < factors.size(); ++i) {
        std::cout << factors[i];
        if (i != factors.size() - 1)
            std::cout << " * ";
    }

    std::cout << "\n";
}