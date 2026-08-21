#include "common/dflash2_selector_validation.h"
#include "host_check.h"

#include <cstdio>
#include <string>

using namespace dflash::common;

static int g_checks = 0;

static DFlash2SelectorLayout valid_layout() {
    DFlash2SelectorLayout layout;
    layout.rank = 32;
    layout.top_k = 16;
    layout.hproj_rank = 32;
    layout.pred_rank = 32;
    layout.pred_vocab = 151936;
    layout.succ_rank = 32;
    layout.succ_vocab = 151936;
    layout.target_output_vocab = 151936;
    layout.target_declared_vocab = 151936;
    return layout;
}

int main() {
    std::string error;
    DFlash2SelectorLayout layout = valid_layout();
    CHECK(validate_dflash2_selector_layout(layout, error));
    CHECK(error.empty());

    for (int K = 1; K <= 8; ++K) {
        layout = valid_layout();
        layout.top_k = K;
        CHECK(validate_dflash2_selector_layout(layout, error));
    }
    for (int K : {12, 16}) {
        layout = valid_layout();
        layout.top_k = K;
        CHECK(validate_dflash2_selector_layout(layout, error));
    }
    for (int K : {0, 9, 10, 11, 13, 14, 15, 17}) {
        layout = valid_layout();
        layout.top_k = K;
        CHECK(!validate_dflash2_selector_layout(layout, error));
        CHECK(error.find("top_k=") != std::string::npos);
        CHECK(error.find("unsupported") != std::string::npos);
    }

    layout = valid_layout();
    layout.succ_vocab--;
    CHECK(!validate_dflash2_selector_layout(layout, error));
    CHECK(error.find("codebook vocab mismatch") != std::string::npos);

    layout = valid_layout();
    layout.target_output_vocab--;
    CHECK(!validate_dflash2_selector_layout(layout, error));
    CHECK(error.find("target vocab mismatch") != std::string::npos);

    layout = valid_layout();
    layout.target_declared_vocab = 0;
    layout.target_output_vocab--;
    CHECK(!validate_dflash2_selector_layout(layout, error));
    CHECK(error.find("target output/lm_head") != std::string::npos);

    layout = valid_layout();
    layout.target_output_vocab = 0;
    layout.target_declared_vocab--;
    CHECK(!validate_dflash2_selector_layout(layout, error));
    CHECK(error.find("target.n_vocab") != std::string::npos);

    layout = valid_layout();
    layout.pred_vocab = 8;
    layout.succ_vocab = 8;
    layout.target_output_vocab = 0;
    layout.target_declared_vocab = 0;
    CHECK(!validate_dflash2_selector_layout(layout, error));
    CHECK(error.find("exceeds codebook vocab") != std::string::npos);

    layout = valid_layout();
    layout.succ_rank = 31;
    CHECK(!validate_dflash2_selector_layout(layout, error));
    CHECK(error.find("rank mismatch") != std::string::npos);

    std::printf("dflash2 selector validation: %d checks passed\n", g_checks);
    return 0;
}
