// Byte-exact equivalence between lucebox's native ATEM renderer and the
// chat template Muse-Glimmer itself ships.
//
// The fixtures in test/data/muse_atem/ were produced by the MODEL's own
// tokenizer (`AutoTokenizer.apply_chat_template` on
// meta-models/Muse-Glimmer-30B, current_date pinned to 2026-08-12), so this
// is a comparison against ground truth rather than against a second reading
// of the same spec. Regenerate with the command in data/muse_atem/README.md
// if the model ships a new template.
//
// Why byte-exact and not "close enough": prompt drift on this family is
// silent. The model still answers fluently while tool calling degrades, and
// the degradation only shows up on an agentic suite — the geo-quant golden
// gate caught exactly that (TOOL 0/30 with perfect ATEM blocks) when the
// serving path mishandled the control tokens.

#include "chat_template.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using dflash::common::ChatFormat;
using dflash::common::ChatMessage;

static int g_fails = 0;

// Fixture location, in precedence order:
//   1. DFLASH_TEST_DATA_DIR  -- for a relocated / installed tree;
//   2. the source path baked in by CMake -- works from ANY working directory;
//   3. a CWD-relative guess, only if someone builds this without the define.
//
// (2) exists because the CWD-relative form silently made this test pass when
// run by hand from server/ and fail under ctest, which runs from the build
// directory. A fixture-reading test that depends on where it was invoked from
// is a test that reports the wrong answer half the time.
static std::string data_dir() {
    if (const char * d = std::getenv("DFLASH_TEST_DATA_DIR")) return d;
#ifdef DFLASH_MUSE_ATEM_DATA_DIR
    return DFLASH_MUSE_ATEM_DATA_DIR;
#else
    return "test/data/muse_atem";
#endif
}

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Report the first divergence with context — a raw "strings differ" on a
// 1.8 KB prompt is unactionable.
static void expect_equal(const std::string & name,
                         const std::string & got,
                         const std::string & want) {
    if (got == want) {
        std::printf("  %-16s OK (%zu bytes)\n", name.c_str(), got.size());
        return;
    }
    ++g_fails;
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
    const size_t from = i > 60 ? i - 60 : 0;
    std::fprintf(stderr, "FAIL: %s diverges at byte %zu\n", name.c_str(), i);
    std::fprintf(stderr, "  want: ...%s\n",
                 want.substr(from, 140).c_str());
    std::fprintf(stderr, "  got : ...%s\n",
                 got.substr(from, 140).c_str());
}

int main() {
    // The fixtures were rendered with this date; the renderer reads the
    // override so the dateline is deterministic here.
    ::setenv("DFLASH_ATEM_CURRENT_DATE", "2026-08-12", 1);

    const std::string dir = data_dir();
    const std::string kTools =
        R"([{"type":"function","function":{"name":"fs.read_file",)"
        R"("description":"Read a file","parameters":{"type":"object",)"
        R"("properties":{"path":{"type":"string"}},"required":["path"]}}},)"
        R"({"type":"function","function":{"name":"web.search",)"
        R"("description":"Search","parameters":{"type":"object"}}}])";
    const std::string kToolsOne =
        R"([{"type":"function","function":{"name":"fs.read_file",)"
        R"("description":"Read a file","parameters":{"type":"object",)"
        R"("properties":{"path":{"type":"string"}},"required":["path"]}}}])";

    struct Case {
        const char * name;
        std::vector<ChatMessage> messages;
        std::string tools;
    };

    ChatMessage tool_call;
    tool_call.role = "assistant";
    tool_call.content =
        "<atem:function_calls>\n<atem:invoke name=\"fs.read_file\">\n"
        "<atem:parameter name=\"path\">a.txt</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>";
    ChatMessage tool_result;
    tool_result.role = "tool";
    tool_result.content = "hello";
    tool_result.name = "fs.read_file";
    tool_result.tool_call_id = "call_1";

    std::vector<Case> cases = {
        {"user_only", {{"user", "What is 2+2?", "", ""}}, ""},
        {"system_user",
         {{"system", "You are terse.", "", ""}, {"user", "Hi", "", ""}}, ""},
        {"tools", {{"user", "read a.txt", "", ""}}, kTools},
        {"tool_roundtrip",
         {{"user", "read a.txt", "", ""}, tool_call, tool_result}, kToolsOne},
        {"multi_assistant",
         {{"user", "hi", "", ""},
          {"assistant", "thinking out loud", "", ""},
          {"assistant", "the answer", "", ""}}, ""},
    };

    std::printf("ATEM renderer vs the model's own chat template:\n");
    for (const auto & c : cases) {
        std::string want;
        const std::string path = dir + "/" + c.name + ".txt";
        if (!read_file(path, want)) {
            std::fprintf(stderr, "FAIL: cannot read fixture %s\n", path.c_str());
            ++g_fails;
            continue;
        }
        const std::string got = render_chat_template(
            c.messages, ChatFormat::ATEM, /*add_generation_prompt=*/true,
            /*enable_thinking=*/false, c.tools);
        expect_equal(c.name, got, want);
    }

    if (g_fails == 0) {
        std::printf("test_muse_atem_reference_prompts: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_muse_atem_reference_prompts: %d failure(s)\n",
                 g_fails);
    return 1;
}
