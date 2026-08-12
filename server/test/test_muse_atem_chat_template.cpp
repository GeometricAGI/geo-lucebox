// ATEM chat-format renderer (Muse-Glimmer). Pins the turn grammar the
// artifact's embedded GGUF template defines, because the failure mode is
// silent: a prompt that renders "almost right" still generates fluent text
// while tool calling collapses. On the 122-item agentic golden suite the
// muse artifacts are gated on, a serving path that dropped the <|eom|>/
// <|eot|> control tokens scored TOOL 0/30 with otherwise perfect ATEM
// blocks in the output.
//
// Reference grammar (from tokenizer.chat_template in the artifact):
//   <|start|>system<|message|>{sys}\n\nReasoning strength: high.
//     [\n\n{tool defs}]\n\n# Valid recipients: …<|eot|>
//   <|start|>user<|message|>{msg}<|eot|>
//   <|start|>tool {name}<|message|><tool_output name="{name}">\n{r}\n
//     </tool_output><|eot|>
//   <|start|>assistant[ to={recipient}]<|message|>{msg}<|eot|>
//   <|start|>assistant                                  ← generation prompt

#include "chat_template.h"

#include <cstdio>
#include <string>
#include <vector>

using dflash::common::ChatFormat;
using dflash::common::ChatMessage;

static int g_fails = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fails; } \
    } while (0)

static bool has(const std::string & hay, const std::string & needle) {
    return hay.find(needle) != std::string::npos;
}

// Ordered-subsequence check: the pieces appear, in this order, without
// requiring the renderer to be byte-frozen between them.
static bool in_order(const std::string & hay,
                     const std::vector<std::string> & parts) {
    size_t pos = 0;
    for (const auto & p : parts) {
        const size_t at = hay.find(p, pos);
        if (at == std::string::npos) return false;
        pos = at + p.size();
    }
    return true;
}

static void test_arch_dispatch() {
    CHECK(dflash::common::chat_format_for_arch("muse-glimmer") == ChatFormat::ATEM,
          "muse-glimmer must resolve to the ATEM chat format");
    // Neighbouring families must not be perturbed by the new row.
    CHECK(dflash::common::chat_format_for_arch("gemma4") == ChatFormat::GEMMA4,
          "gemma4 dispatch regressed");
    CHECK(dflash::common::chat_format_for_arch("deepseek4") == ChatFormat::DEEPSEEK4,
          "deepseek4 dispatch regressed");
}

static void test_default_system_and_user_turn() {
    std::vector<ChatMessage> msgs;
    msgs.push_back({"user", "What is 2+2?", "", ""});
    const std::string out = render_chat_template(msgs, ChatFormat::ATEM, true);

    // A conversation with no system message still gets the system turn the
    // template synthesises — the model relies on it for the recipient list.
    CHECK(in_order(out, {"<|start|>system<|message|>",
                         "Reasoning strength: high.",
                         "# Valid recipients: \"self\", \"user\".",
                         "<|eot|>",
                         "<|start|>user<|message|>What is 2+2?<|eot|>",
                         "<|start|>assistant"}),
          "default system + user turn + generation prompt");
    CHECK(!has(out, "<|start|>assistant<|message|>"),
          "generation prompt must NOT pre-open <|message|> (the model emits "
          "the recipient itself)");
}

static void test_explicit_system_is_not_duplicated() {
    std::vector<ChatMessage> msgs;
    msgs.push_back({"system", "You are terse.", "", ""});
    msgs.push_back({"user", "Hi", "", ""});
    const std::string out = render_chat_template(msgs, ChatFormat::ATEM, true);

    CHECK(has(out, "<|start|>system<|message|>You are terse."),
          "explicit system content is rendered");
    size_t n = 0;
    for (size_t p = out.find("<|start|>system"); p != std::string::npos;
         p = out.find("<|start|>system", p + 1)) {
        ++n;
    }
    CHECK(n == 1, "exactly one system turn when the caller supplies one");
}

static void test_tool_definitions_and_recipients() {
    const std::string tools =
        R"([{"type":"function","function":{"name":"fs.read_file",)"
        R"("description":"Read a file","parameters":{"type":"object",)"
        R"("properties":{"path":{"type":"string"}},"required":["path"]}}},)"
        R"({"type":"function","function":{"name":"web.search",)"
        R"("description":"Search","parameters":{"type":"object"}}}])";
    std::vector<ChatMessage> msgs;
    msgs.push_back({"user", "read a.txt", "", ""});
    const std::string out =
        render_chat_template(msgs, ChatFormat::ATEM, true, false, tools);

    CHECK(has(out, "<atem:function_calls>"),
          "tool preamble teaches the ATEM invocation block");
    CHECK(has(out, "\"name\": \"fs.read_file\""), "function schema is rendered");
    CHECK(has(out, "\"name\": \"web.search\""), "second function schema");
    // Namespaces are deduped, ordered by first appearance, and the recipient
    // list is what constrains `to=` at generation time.
    CHECK(has(out, "# Valid recipients: \"self\", \"fs.*\", \"web.*\", \"user\"."),
          "recipient list carries one entry per tool namespace");
}

static void test_tool_result_turn_is_addressed_by_name() {
    std::vector<ChatMessage> msgs;
    msgs.push_back({"user", "read a.txt", "", ""});
    ChatMessage call;
    call.role = "assistant";
    call.content =
        "<atem:function_calls>\n<atem:invoke name=\"fs.read_file\">\n"
        "<atem:parameter name=\"path\">a.txt</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>";
    msgs.push_back(call);
    ChatMessage result;
    result.role = "tool";
    result.content = "hello";
    result.name = "fs.read_file";
    result.tool_call_id = "call_1";
    msgs.push_back(result);
    const std::string out = render_chat_template(msgs, ChatFormat::ATEM, true);

    CHECK(in_order(out, {"<|start|>assistant to=fs.read_file<|message|>",
                         "<atem:invoke name=\"fs.read_file\">",
                         // followed by a `tool` turn, so the assistant turn
                         // closes with <|eot|>, not <|eom|> (verified against
                         // the model's own template — see the fixture test)
                         "<|eot|>",
                         "<|start|>tool fs.read_file<|message|>",
                         "<tool_output name=\"fs.read_file\">\nhello\n"
                         "</tool_output><|eot|>",
                         "<|start|>assistant"}),
          "tool-call turn is addressed to the tool and the result turn is "
          "addressed by NAME");
}

static void test_tool_result_falls_back_to_call_id() {
    // OpenAI-style clients may send only tool_call_id. Rendering an empty
    // name would produce `<|start|>tool <|message|>`, which the model reads
    // as an unnamed recipient.
    std::vector<ChatMessage> msgs;
    msgs.push_back({"user", "go", "", ""});
    ChatMessage result;
    result.role = "tool";
    result.content = "42";
    result.tool_call_id = "fs.read_file";
    msgs.push_back(result);
    const std::string out = render_chat_template(msgs, ChatFormat::ATEM, true);
    CHECK(has(out, "<|start|>tool fs.read_file<|message|>"),
          "tool name falls back to tool_call_id when name is absent");
    CHECK(!has(out, "<|start|>tool <|message|>"),
          "never emit an unnamed tool recipient");
}

static void test_assistant_turns_are_addressed_to_user() {
    // Verified against the model's own template (see
    // test_muse_atem_reference_prompts): a plain assistant turn is ALWAYS
    // addressed `to=user` and always closes with <|eot|>. The bare
    // `<|start|>assistant<|message|>` form never appears in a rendered
    // history, and <|eom|> is reserved for turns that continue a channel
    // (the reasoning channel, chained tool calls).
    std::vector<ChatMessage> msgs;
    msgs.push_back({"user", "hi", "", ""});
    msgs.push_back({"assistant", "thinking out loud", "", ""});
    msgs.push_back({"assistant", "the answer", "", ""});
    const std::string out = render_chat_template(msgs, ChatFormat::ATEM, false);
    CHECK(in_order(out, {"<|start|>assistant to=user<|message|>thinking out "
                         "loud<|eot|>",
                         "<|start|>assistant to=user<|message|>the "
                         "answer<|eot|>"}),
          "consecutive assistant turns are each addressed to=user and end "
          "with <|eot|>");
    CHECK(!has(out, "<|start|>assistant<|message|>"),
          "assistant turns in history always carry a recipient");
    CHECK(out.substr(out.size() - 18) != "<|start|>assistant",
          "no generation prompt when add_generation_prompt=false");
}

int main() {
    test_arch_dispatch();
    test_default_system_and_user_turn();
    test_explicit_system_is_not_duplicated();
    test_tool_definitions_and_recipients();
    test_tool_result_turn_is_addressed_by_name();
    test_tool_result_falls_back_to_call_id();
    test_assistant_turns_are_addressed_to_user();

    if (g_fails == 0) {
        std::printf("test_muse_atem_chat_template: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_muse_atem_chat_template: %d failure(s)\n", g_fails);
    return 1;
}
