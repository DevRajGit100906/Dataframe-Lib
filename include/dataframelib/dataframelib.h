#pragma once
#include "expr.h"
#include "kernel.h"
#include "logical_plan.h"
#include "eager_dataframe.h"
#include "lazy_dataframe.h"
#include "query_optimizer.h"

#ifndef ARROW_THROW_NOT_OK
#include <stdexcept>
#define ARROW_THROW_NOT_OK(expr)                                          \
    do {                                                                  \
        auto _arrow_status = (expr);                                      \
        if (!_arrow_status.ok())                                          \
            throw std::runtime_error(_arrow_status.ToString());           \
    } while (0)
#endif
