#pragma once

// 由 CMake configure_file 生成，包含 Git commit hash。
// 若未使用 CMake 构建，使用手动版本号。
#ifndef ZED_DS_APP_VERSION
#define ZED_DS_APP_VERSION "0.2.0"
#endif

#ifndef ZED_DS_GIT_HASH
#define ZED_DS_GIT_HASH "unknown"
#endif
