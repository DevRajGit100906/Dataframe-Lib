#pragma once
#include "logical_plan.h"
#include <memory>
#include <vector>

namespace dataframelib {
    class OptimizerRule {
        public:
            virtual ~OptimizerRule() = default;
            virtual std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const = 0;
    };
    class QueryOptimizer {
        public:
            static std::shared_ptr<LogicalPlan> optimize(std::shared_ptr<LogicalPlan> plan);
            static std::shared_ptr<LogicalPlan> rebuild_with_children(
                const std::shared_ptr<LogicalPlan>& node,
                const std::vector<std::shared_ptr<LogicalPlan>>& new_children);
            static std::shared_ptr<LogicalPlan> recurse_children(
                std::shared_ptr<LogicalPlan> node, 
                const OptimizerRule& rule);
    };
    class ConstFoldRule : public OptimizerRule {
        public:
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
        private:
            Expression fold_expr(Expression expr) const;
            ExprPtr fold_node(ExprPtr node) const;
    };
    class ExprSimplificationRule : public OptimizerRule {
        public:
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
        private:
            Expression simplify_expr(Expression expr) const;
            ExprPtr simplify_node(ExprPtr node) const;
        };
    class FilterEliminationRule : public OptimizerRule {
        public:
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
    };
    class ConjSplitRule : public OptimizerRule {
        public:
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
    };
    class PredicatePushdownRule : public OptimizerRule {
        public:
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
    };
    class ProjectionPushdownRule : public OptimizerRule {
        public:
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
    };
    class LimitPushdownRule : public OptimizerRule {
        public:
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
    };
    class FilterMergingRule : public OptimizerRule {
        public: 
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
    };
    class ProjectMergingRule : public OptimizerRule {
        public: 
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
    };
    class SortEliminationRule : public OptimizerRule {
        public: 
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
    };
    class HavingVsWhereRule : public OptimizerRule {
        public: 
            std::shared_ptr<LogicalPlan> apply(std::shared_ptr<LogicalPlan> plan) const override;
    };
}