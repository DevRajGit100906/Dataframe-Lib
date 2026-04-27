#include "dataframelib/kernel.h"
#include <type_traits>
#include <stdexcept>
namespace dataframelib {
    std::pair<std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Array>> 
    promote_arrays(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        if (!left || !right) {
            throw std::invalid_argument("Cannot promote null arrays");
        };
        if (left->length() != right->length()) {
            throw std::invalid_argument("Arrays must have the same length");
        };
        if (left->type_id() == arrow::Type::NA && right->type_id() != arrow::Type::NA) {
            return {arrow::MakeArrayOfNull(right->type(), left->length()).ValueOrDie(), right};
        };
        if (right->type_id() == arrow::Type::NA && left->type_id() != arrow::Type::NA) {
            return {left, arrow::MakeArrayOfNull(left->type(), right->length()).ValueOrDie()};
        };
        if (left->type_id() == right->type_id()) {
            return {left, right};
        };
        bool left_is_float = arrow::is_floating(left->type_id());
        bool right_is_float = arrow::is_floating(right->type_id());
        std::shared_ptr<arrow::Array> new_left = left;
        std::shared_ptr<arrow::Array> new_right = right;
        if (left_is_float || right_is_float) {
            if (left->type_id() != arrow::Type::DOUBLE) {
                arrow::DoubleBuilder builder;
                DFL_CHECK_OK(builder.Reserve(left->length()));
                dispatch_array(left, [&](const auto& typed_array) {
                    using ValueType = decltype(typed_array->Value(0));
                    if constexpr (std::is_arithmetic_v<ValueType>) {
                        const int64_t n = typed_array->length();
                        for (int64_t i = 0; i < n; ++i) {
                            if (typed_array->IsNull(i)) builder.UnsafeAppendNull();
                            else builder.UnsafeAppend(static_cast<double>(typed_array->Value(i)));
                        };
                    }
                    else {
                        throw std::runtime_error("Attempted to promote a non-numeric array.");
                    };
                });
                auto status = builder.Finish(&new_left);
                if (!status.ok()) throw std::runtime_error("Failed to build Double array for left");
            };
            if (right->type_id() != arrow::Type::DOUBLE) {
                arrow::DoubleBuilder builder;
                DFL_CHECK_OK(builder.Reserve(right->length()));
                dispatch_array(right, [&](const auto& typed_array) {
                    using ValueType = decltype(typed_array->Value(0));
                    if constexpr (std::is_arithmetic_v<ValueType>) {
                        const int64_t n = typed_array->length();
                        for (int64_t i = 0; i < n; ++i) {
                            if (typed_array->IsNull(i)) builder.UnsafeAppendNull();
                            else builder.UnsafeAppend(static_cast<double>(typed_array->Value(i)));
                        };
                    }
                    else {
                        throw std::runtime_error("Attempted to promote a non-numeric array.");
                    };
                });
                auto status = builder.Finish(&new_right);
                if (!status.ok()) throw std::runtime_error("Failed to build Double array for right");
            };
        }
        else {
            if (left->type_id() != arrow::Type::INT64) {
                arrow::Int64Builder builder;
                DFL_CHECK_OK(builder.Reserve(left->length()));
                dispatch_array(left, [&](const auto& typed_array) {
                    using ValueType = decltype(typed_array->Value(0));
                    if constexpr (std::is_arithmetic_v<ValueType>) {
                        const int64_t n = typed_array->length();
                        for (int64_t i = 0; i < n; ++i) {
                            if (typed_array->IsNull(i)) builder.UnsafeAppendNull();
                            else builder.UnsafeAppend(static_cast<int64_t>(typed_array->Value(i)));
                        };
                    }
                    else {
                        throw std::runtime_error("Attempted to promote a non-numeric array.");
                    };
                });
                auto status = builder.Finish(&new_left);
                if (!status.ok()) throw std::runtime_error("Failed to build Int64 array for left");
            };
            if (right->type_id() != arrow::Type::INT64) {
                arrow::Int64Builder builder;
                DFL_CHECK_OK(builder.Reserve(right->length()));
                dispatch_array(right, [&](const auto& typed_array) {
                    using ValueType = decltype(typed_array->Value(0));
                    if constexpr (std::is_arithmetic_v<ValueType>) {
                        const int64_t n = typed_array->length();
                        for (int64_t i = 0; i < n; ++i) {
                            if (typed_array->IsNull(i)) builder.UnsafeAppendNull();
                            else builder.UnsafeAppend(static_cast<int64_t>(typed_array->Value(i)));
                        };
                    }
                    else {
                        throw std::runtime_error("Attempted to promote a non-numeric array.");
                    };
                });
                auto status = builder.Finish(&new_right);
                if (!status.ok()) throw std::runtime_error("Failed to build Int64 array for right");
            };
        };
        return {new_left, new_right};
    };
    std::shared_ptr<arrow::Array> add(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        auto [promoted_left, promoted_right] = promote_arrays(left, right);
        if (promoted_left->type_id() == arrow::Type::DOUBLE) {
            return BinaryNumericKernel<AddOp, arrow::DoubleBuilder>::apply(promoted_left, promoted_right);
        } 
        else {
            return BinaryNumericKernel<AddOp, arrow::Int64Builder>::apply(promoted_left, promoted_right);
        };
    };
    std::shared_ptr<arrow::Array> sub(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        auto [promoted_left, promoted_right] = promote_arrays(left, right);
        if (promoted_left->type_id() == arrow::Type::DOUBLE) {
            return BinaryNumericKernel<SubtractOp, arrow::DoubleBuilder>::apply(promoted_left, promoted_right);
        } 
        else {
            return BinaryNumericKernel<SubtractOp, arrow::Int64Builder>::apply(promoted_left, promoted_right);
        };
    };
    std::shared_ptr<arrow::Array> mul(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        auto [promoted_left, promoted_right] = promote_arrays(left, right);
        if (promoted_left->type_id() == arrow::Type::DOUBLE) {
            return BinaryNumericKernel<MultiplyOp, arrow::DoubleBuilder>::apply(promoted_left, promoted_right);
        } 
        else {
            return BinaryNumericKernel<MultiplyOp, arrow::Int64Builder>::apply(promoted_left, promoted_right);
        };
    };
    std::shared_ptr<arrow::Array> div(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        auto [promoted_left, promoted_right] = promote_arrays(left, right);
        if (promoted_left->type_id() == arrow::Type::DOUBLE) {
            return BinaryNumericKernel<DivideOp, arrow::DoubleBuilder>::apply(promoted_left, promoted_right);
        } 
        else {
            return BinaryNumericKernel<DivideOp, arrow::Int64Builder>::apply(promoted_left, promoted_right);
        };
    };
    std::shared_ptr<arrow::Array> mod(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        auto [promoted_left, promoted_right] = promote_arrays(left, right);
        if (promoted_left->type_id() == arrow::Type::DOUBLE) {
            return BinaryNumericKernel<ModuloOp, arrow::DoubleBuilder>::apply(promoted_left, promoted_right);
        } 
        else {
            return BinaryNumericKernel<ModuloOp, arrow::Int64Builder>::apply(promoted_left, promoted_right);
        };
    };
    std::shared_ptr<arrow::Array> eq(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        if (left->type_id() == arrow::Type::STRING && right->type_id() == arrow::Type::STRING) {
            return BinaryStringComparisonKernel<EqOp>::apply(left, right);
        };
        auto [pl, pr] = promote_arrays(left, right);
        if (pl->type_id() == arrow::Type::DOUBLE) return BinaryComparisonKernel<EqOp, arrow::DoubleArray>::apply(pl, pr);
        return BinaryComparisonKernel<EqOp, arrow::Int64Array>::apply(pl, pr);
    };
    std::shared_ptr<arrow::Array> ne(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        if (left->type_id() == arrow::Type::STRING && right->type_id() == arrow::Type::STRING) {
            return BinaryStringComparisonKernel<NeqOp>::apply(left, right);
        };
        auto [pl, pr] = promote_arrays(left, right);
        if (pl->type_id() == arrow::Type::DOUBLE) return BinaryComparisonKernel<NeqOp, arrow::DoubleArray>::apply(pl, pr);
        return BinaryComparisonKernel<NeqOp, arrow::Int64Array>::apply(pl, pr);
    };
    std::shared_ptr<arrow::Array> lt(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        if (left->type_id() == arrow::Type::STRING && right->type_id() == arrow::Type::STRING) {
            return BinaryStringComparisonKernel<LtOp>::apply(left, right);
        };
        auto [pl, pr] = promote_arrays(left, right);
        if (pl->type_id() == arrow::Type::DOUBLE) return BinaryComparisonKernel<LtOp, arrow::DoubleArray>::apply(pl, pr);
        return BinaryComparisonKernel<LtOp, arrow::Int64Array>::apply(pl, pr);
    };
    std::shared_ptr<arrow::Array> le(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        if (left->type_id() == arrow::Type::STRING && right->type_id() == arrow::Type::STRING) {
            return BinaryStringComparisonKernel<LteOp>::apply(left, right);
        };
        auto [pl, pr] = promote_arrays(left, right);
        if (pl->type_id() == arrow::Type::DOUBLE) return BinaryComparisonKernel<LteOp, arrow::DoubleArray>::apply(pl, pr);
        return BinaryComparisonKernel<LteOp, arrow::Int64Array>::apply(pl, pr);
    };
    std::shared_ptr<arrow::Array> gt(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        if (left->type_id() == arrow::Type::STRING && right->type_id() == arrow::Type::STRING) {
            return BinaryStringComparisonKernel<GtOp>::apply(left, right);
        };
        auto [pl, pr] = promote_arrays(left, right);
        if (pl->type_id() == arrow::Type::DOUBLE) return BinaryComparisonKernel<GtOp, arrow::DoubleArray>::apply(pl, pr);
        return BinaryComparisonKernel<GtOp, arrow::Int64Array>::apply(pl, pr);
    };
    std::shared_ptr<arrow::Array> ge(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right) {
        if (left->type_id() == arrow::Type::STRING && right->type_id() == arrow::Type::STRING) {
            return BinaryStringComparisonKernel<GteOp>::apply(left, right);
        };
        auto [pl, pr] = promote_arrays(left, right);
        if (pl->type_id() == arrow::Type::DOUBLE) return BinaryComparisonKernel<GteOp, arrow::DoubleArray>::apply(pl, pr);
        return BinaryComparisonKernel<GteOp, arrow::Int64Array>::apply(pl, pr);
    };
    std::shared_ptr<arrow::Array> and_kleene(const std::shared_ptr<arrow::Array>& lhs, const std::shared_ptr<arrow::Array>& rhs) {
        return BinaryKleeneKernel<AndKleeneOp>::apply(lhs, rhs);
    };
    std::shared_ptr<arrow::Array> or_kleene(const std::shared_ptr<arrow::Array>& lhs, const std::shared_ptr<arrow::Array>& rhs) {
        return BinaryKleeneKernel<OrKleeneOp>::apply(lhs, rhs);
    };
    std::shared_ptr<arrow::Array> not_kleene(const std::shared_ptr<arrow::Array>& input) {
        if (input->type_id() != arrow::Type::BOOL) {
            throw std::invalid_argument("not_kleene requires a Boolean array");
        };
        return UnaryKleeneKernel<NotKleeneOp, arrow::BooleanArray>::apply(input);
    };
    std::shared_ptr<arrow::Array> abs(const std::shared_ptr<arrow::Array>& input) {
        std::shared_ptr<arrow::Array> result;
        if (input->type_id() == arrow::Type::NA) {
            return input;
        };
        dispatch_array(input, [&](const auto& typed_array) {
            using ValueType = decltype(typed_array->Value(0));
            if constexpr (std::is_same_v<ValueType, double>) {
                result = UnaryNumericKernel<AbsOp, arrow::DoubleBuilder>::apply(input);
            } 
            else if constexpr (std::is_same_v<ValueType, int64_t>) {
                result = UnaryNumericKernel<AbsOp, arrow::Int64Builder>::apply(input);
            } 
            else {
                throw std::runtime_error("Abs requires Int64 or Double array");
            };
        });
        return result;
    };
    std::shared_ptr<arrow::Array> negate(const std::shared_ptr<arrow::Array>& input) {
        std::shared_ptr<arrow::Array> result;
        if (input->type_id() == arrow::Type::NA) {
            return input;
        };
        dispatch_array(input, [&](const auto& typed_array) {
            using ValueType = decltype(typed_array->Value(0));
            if constexpr (std::is_same_v<ValueType, double>) {
                result = UnaryNumericKernel<NegateOp, arrow::DoubleBuilder>::apply(input);
            } 
            else if constexpr (std::is_same_v<ValueType, int64_t>) {
                result = UnaryNumericKernel<NegateOp, arrow::Int64Builder>::apply(input);
            } 
            else {
                throw std::runtime_error("Negate requires Int64 or Double array");
            };
        });
        return result;
    };
    std::shared_ptr<arrow::Array> is_null(const std::shared_ptr<arrow::Array>& input) {
        arrow::BooleanBuilder builder;
        const int64_t n = input->length();
        DFL_CHECK_OK(builder.Reserve(n));
        for (int64_t i = 0; i < n; ++i) {
            builder.UnsafeAppend(input->IsNull(i));
        };
        std::shared_ptr<arrow::Array> out;
        DFL_CHECK_OK(builder.Finish(&out));
        return out;
    };
    std::shared_ptr<arrow::Array> is_not_null(const std::shared_ptr<arrow::Array>& input) {
        arrow::BooleanBuilder builder;
        const int64_t n = input->length();
        DFL_CHECK_OK(builder.Reserve(n));
        for (int64_t i = 0; i < n; ++i) {
            builder.UnsafeAppend(input->IsValid(i));
        };
        std::shared_ptr<arrow::Array> out;
        DFL_CHECK_OK(builder.Finish(&out));
        return out;
    };
    std::shared_ptr<arrow::Array> utf8_length(const std::shared_ptr<arrow::Array>& input) {
        if (input->type_id() != arrow::Type::STRING) {
            throw std::invalid_argument("utf8_length requires a String array");
        };
        return UnaryStringKernel<Utf8LengthOp, arrow::Int64Builder>::apply(input);
    };
    std::shared_ptr<arrow::Array> contains(const std::shared_ptr<arrow::Array>& input, std::string_view substring) {
        if (input->type_id() != arrow::Type::STRING) {
            throw std::invalid_argument("contains requires a String array");
        };
        return StringPredicateKernel<ContainsOp>::apply(input, substring);
    };
    std::shared_ptr<arrow::Array> starts_with(const std::shared_ptr<arrow::Array>& input, std::string_view prefix) {
        if (input->type_id() != arrow::Type::STRING) {
            throw std::invalid_argument("starts_with requires a String array");
        };
        return StringPredicateKernel<StartsWithOp>::apply(input, prefix);
    };
    std::shared_ptr<arrow::Array> ends_with(const std::shared_ptr<arrow::Array>& input, std::string_view suffix) {
        if (input->type_id() != arrow::Type::STRING) {
            throw std::invalid_argument("ends_with requires a String array");
        };
        return StringPredicateKernel<EndsWithOp>::apply(input, suffix);
    };
    std::shared_ptr<arrow::Array> to_lower(const std::shared_ptr<arrow::Array>& input) {
        if (input->type_id() != arrow::Type::STRING) {
            throw std::invalid_argument("to_lower requires a String array");
        };
        return UnaryStringKernel<ToLowerOp, arrow::StringBuilder>::apply(input);
    };
    std::shared_ptr<arrow::Array> to_upper(const std::shared_ptr<arrow::Array>& input) {
        if (input->type_id() != arrow::Type::STRING) {
            throw std::invalid_argument("to_upper requires a String array");
        };
        return UnaryStringKernel<ToUpperOp, arrow::StringBuilder>::apply(input);
    };
    std::shared_ptr<arrow::Array> broadcast_scalar(const std::shared_ptr<arrow::Scalar>& scalar, int64_t length) {
        if (!scalar) {
            throw std::invalid_argument("Cannot broadcast a null scalar");
        };
        std::shared_ptr<arrow::Array> out;
        switch(scalar->type->id()) {
            case arrow::Type::INT64: {
                arrow::Int64Builder builder;
                DFL_CHECK_OK(builder.Reserve(length));
                auto int_scalar = std::static_pointer_cast<arrow::Int64Scalar>(scalar);
                for (int64_t i = 0; i < length; ++i) {
                    if (int_scalar->is_valid) builder.UnsafeAppend(int_scalar->value);
                    else builder.UnsafeAppendNull();
                };
                DFL_CHECK_OK(builder.Finish(&out));
                break;
            };
            case arrow::Type::DOUBLE: {
                arrow::DoubleBuilder builder;
                DFL_CHECK_OK(builder.Reserve(length));
                auto double_scalar = std::static_pointer_cast<arrow::DoubleScalar>(scalar);
                for (int64_t i = 0; i < length; ++i) {
                    if (double_scalar->is_valid) builder.UnsafeAppend(double_scalar->value);
                    else builder.UnsafeAppendNull();
                };
                DFL_CHECK_OK(builder.Finish(&out));
                break;
            };
            case arrow::Type::STRING: {
                arrow::StringBuilder builder;
                DFL_CHECK_OK(builder.Reserve(length));
                auto string_scalar = std::static_pointer_cast<arrow::StringScalar>(scalar);
                for (int64_t i = 0; i < length; ++i) {
                    if (string_scalar->is_valid) DFL_CHECK_OK(builder.Append(string_scalar->value->ToString()));
                    else DFL_CHECK_OK(builder.AppendNull());
                };
                DFL_CHECK_OK(builder.Finish(&out));
                break;
            };
            case arrow::Type::BOOL: {
                arrow::BooleanBuilder builder;
                DFL_CHECK_OK(builder.Reserve(length));
                auto bool_scalar = std::static_pointer_cast<arrow::BooleanScalar>(scalar);
                for (int64_t i = 0; i < length; ++i) {
                    if (bool_scalar->is_valid) builder.UnsafeAppend(bool_scalar->value);
                    else builder.UnsafeAppendNull();
                };
                DFL_CHECK_OK(builder.Finish(&out));
                break;
            };
            default:
                throw std::invalid_argument("Unsupported scalar type for broadcasting");
        };
        return out;
    };
    std::shared_ptr<arrow::Scalar> AggregateKernel(
        AggFn fn, 
        const std::shared_ptr<arrow::Array>& values, 
        const std::vector<int64_t>& indices) {
        if (values->type_id() == arrow::Type::NA) {
            switch (fn) {
                case AggFn::Count:
                    return std::make_shared<arrow::Int64Scalar>(int64_t{0});
                case AggFn::Mean:
                    return arrow::MakeNullScalar(arrow::float64());
                case AggFn::Sum:
                case AggFn::Min:
                case AggFn::Max:
                    return arrow::MakeNullScalar(arrow::null());
                default:
                    throw std::runtime_error("Unsupported aggregation function");
            };
        };
        switch (fn) {
            case AggFn::Sum:
                return dispatch_array(values, [&](auto typed_array) {
                    using ArrayType = typename std::remove_reference<decltype(*typed_array)>::type;
                    return AggregateNumericKernel<SumAggOp, ArrayType>::apply(values, indices);
                });
            case AggFn::Mean:
                return dispatch_array(values, [&](auto typed_array) {
                    using ArrayType = typename std::remove_reference<decltype(*typed_array)>::type;
                    return AggregateNumericKernel<MeanAggOp, ArrayType>::apply(values, indices);
                });
            case AggFn::Count:
                return dispatch_array(values, [&](auto typed_array) {
                    using ArrayType = typename std::remove_reference<decltype(*typed_array)>::type;
                    return AggregateNumericKernel<CountAggOp, ArrayType>::apply(values, indices);
                });
            case AggFn::Min:
                return dispatch_array(values, [&](auto typed_array) {
                    using ArrayType = typename std::remove_reference<decltype(*typed_array)>::type;
                    return AggregateNumericKernel<MinAggOp, ArrayType>::apply(values, indices);
                });
            case AggFn::Max:
                return dispatch_array(values, [&](auto typed_array) {
                    using ArrayType = typename std::remove_reference<decltype(*typed_array)>::type;
                    return AggregateNumericKernel<MaxAggOp, ArrayType>::apply(values, indices);
                });
            default:
                throw std::runtime_error("Unsupported aggregation function");
        };
    };
}