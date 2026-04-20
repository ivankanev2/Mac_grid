#pragma once

#include <cstdio>

#ifndef SMOKE_ENABLE_VERBOSE_DIAGNOSTICS
#define SMOKE_ENABLE_VERBOSE_DIAGNOSTICS 0
#endif

#if SMOKE_ENABLE_VERBOSE_DIAGNOSTICS
#define SMOKE_DIAG_LOG(...) std::printf(__VA_ARGS__)
#define SMOKE_DIAG_ERR(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define SMOKE_DIAG_LOG(...) ((void)0)
#define SMOKE_DIAG_ERR(...) ((void)0)
#endif
