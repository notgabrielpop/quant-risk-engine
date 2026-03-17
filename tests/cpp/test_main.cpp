/**
 * @file test_main.cpp
 * @brief Lightweight test harness — runs all registered tests.
 */

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

struct TestEntry {
    std::string name;
    std::function<bool()> fn;
};

static std::vector<TestEntry>& registry() {
    static std::vector<TestEntry> r;
    return r;
}

void register_test(const char* name, std::function<bool()> fn) {
    registry().push_back({name, std::move(fn)});
}

int main() {
    int pass = 0, fail = 0;
    for (auto& [name, fn] : registry()) {
        bool ok = fn();
        std::printf("  [%s] %s\n", ok ? " PASS " : "*FAIL*", name.c_str());
        ok ? ++pass : ++fail;
    }
    std::printf("\n  Results: %d passed, %d failed, %d total\n", pass, fail, pass + fail);
    return fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
