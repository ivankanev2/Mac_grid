#pragma once

#include <cstdio>

#ifndef SMOKE_ENABLE_VERBOSE_DIAGNOSTICS
#define SMOKE_ENABLE_VERBOSE_DIAGNOSTICS 0
#endif

#if SMOKE_ENABLE_VERBOSE_DIAGNOSTICS
#define SMOKE_DIAG_PRINTF(...) std::printf(__VA_ARGS__)
#else
#define SMOKE_DIAG_PRINTF(...) ((void)0)
#endif
