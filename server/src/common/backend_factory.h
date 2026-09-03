// Backend planning and arch-detecting ModelBackend construction.
//
// BackendArgs is mutable input. prepare_backend() consumes it, resolves model
// and placement facts, normalizes backend policy, and returns an immutable
// BackendPlan. create_backend() accepts only that plan.

#pragma once

#include "backend_args.h"
#include "gguf_inspect.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace dflash::common {

namespace detail {
class BackendPlanBuilder;
}

class BackendPlan;
struct ModelBackend;

// The sole construction entry point. Architecture configs own any path data
// retained by the returned backend, so the plan may have an independent
// lifetime after construction.
std::unique_ptr<ModelBackend> create_backend(const BackendPlan & plan);

class BackendPlan final {
public:
    BackendPlan(BackendPlan &&) noexcept = default;
    BackendPlan & operator=(BackendPlan &&) = delete;
    BackendPlan(const BackendPlan &) = delete;
    BackendPlan & operator=(const BackendPlan &) = delete;

    const BackendArgs & args() const { return args_; }
    const GgufModelInfo & model() const { return model_; }
    const std::string & arch() const { return model_.arch; }
    const std::vector<std::string> & warnings() const { return warnings_; }

private:
    enum class SpeclaEnvironmentAction {
        Preserve,
        Enable,
        Disable,
    };

    BackendPlan() = default;

    BackendArgs args_;
    GgufModelInfo model_;
    std::vector<std::string> warnings_;
    SpeclaEnvironmentAction specla_environment_ =
        SpeclaEnvironmentAction::Preserve;

    friend class detail::BackendPlanBuilder;
    friend std::unique_ptr<ModelBackend> create_backend(
        const BackendPlan & plan);
};

enum class BackendPreparationError {
    InvalidRequest,
    ModelInspection,
    FeatureCompatibility,
};

struct BackendPreparationFailure {
    BackendPreparationError error;
    std::string message;
    std::vector<std::string> warnings;
};

using BackendPreparation =
    std::variant<BackendPlan, BackendPreparationFailure>;

// Consumes the mutable request. Successful preparation performs one GGUF
// inspection and returns the only value accepted by backend construction.
BackendPreparation prepare_backend(
    BackendArgs args,
    BackendAdmissionContext admission = {});

}  // namespace dflash::common
