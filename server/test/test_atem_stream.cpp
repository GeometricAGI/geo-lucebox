// ATEM response segmentation.
//
// The live failure this fixes: serving muse-v4 through dflash_server returned
//   content = " to=selfWhat is 2+2? … Should answer in one word. Probably 4"
// — the recipient marker and the entire reasoning channel leaked into the
// visible answer, because the response side had no counterpart to the chat
// template's segment grammar.
//
// Cases below use the real shape the model emits: the generation prompt ends
// with `<|start|>assistant`, so the first header arrives WITHOUT its opener.

#include "atem_stream.h"
#include "tool_parser.h"

#include <cstdio>
#include <string>

using namespace dflash::common;

static int g_fails = 0;

#define CHECK_EQ_S(got, want, what)                                            \
    do {                                                                       \
        const std::string g_ = (got), w_ = (want);                             \
        if (g_ != w_) {                                                        \
            std::fprintf(stderr, "FAIL: %s\n  got : %s\n  want: %s\n",         \
                         (what), g_.c_str(), w_.c_str());                      \
            ++g_fails;                                                         \
        }                                                                      \
    } while (0)

static void test_reasoning_then_answer() {
    // Exactly the shape observed live.
    const std::string reply =
        " to=self<|message|>Simple arithmetic.<|eom|>"
        "<|start|>assistant to=user<|message|>4<|eot|>";
    const AtemParsed p = atem_parse_response(reply);
    CHECK_EQ_S(p.reasoning, "Simple arithmetic.", "reasoning extracted");
    CHECK_EQ_S(p.content, "4", "content is the user-addressed segment only");
    CHECK_EQ_S(p.tool_text, "", "no tool text");
}

static void test_recipient_marker_never_leaks() {
    const std::string reply = " to=user<|message|>Hello.<|eot|>";
    const AtemParsed p = atem_parse_response(reply);
    CHECK_EQ_S(p.content, "Hello.", "header text is dropped, not emitted");
    CHECK_EQ_S(p.reasoning, "", "no reasoning when the model never opens it");
}

static void test_tool_call_segment_is_routed_by_recipient() {
    const std::string reply =
        " to=self<|message|>Need the file.<|eom|>"
        "<|start|>assistant to=fs.read_file<|message|>"
        "<atem:function_calls>\n<atem:invoke name=\"fs.read_file\">\n"
        "<atem:parameter name=\"path\">a.txt</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls><|eot|>";
    const AtemParsed p = atem_parse_response(reply);
    CHECK_EQ_S(p.reasoning, "Need the file.", "reasoning before the tool turn");
    CHECK_EQ_S(p.content, "", "a tool turn is NOT visible content");
    CHECK_EQ_S(p.tool_recipient, "fs.read_file", "tool recipient captured");
    if (p.tool_text.find("<atem:invoke name=\"fs.read_file\">") == std::string::npos) {
        std::fprintf(stderr, "FAIL: tool body not captured\n");
        ++g_fails;
    }
}

static void test_unterminated_reasoning() {
    // Hitting the token cap mid-reasoning (finish_reason=length) must not
    // dump the partial reasoning into content.
    const std::string reply = " to=self<|message|>Thinking about it";
    const AtemParsed p = atem_parse_response(reply);
    CHECK_EQ_S(p.reasoning, "Thinking about it", "partial reasoning kept as reasoning");
    CHECK_EQ_S(p.content, "", "no content when the turn never reached the user");
}

static void test_incremental_matches_whole_string() {
    // The streaming path feeds tokens; the non-streaming path feeds the
    // string. They must agree, or a client sees a different split depending
    // on whether it streamed.
    const char * toks[] = {
        " to", "=", "self", "<|message|>", "Because ", "2+2", "=4", "<|eom|>",
        "<|start|>", "assistant", " to=user", "<|message|>", "Four", "<|eot|>",
    };
    AtemSegmenter seg(true);
    std::string reasoning, content;
    int opens = 0, closes = 0, ends = 0;
    for (const char * t : toks) {
        const AtemStep s = seg.feed(t);
        switch (s.action) {
        case AtemAction::OpenReasoning:  ++opens;  break;
        case AtemAction::CloseReasoning: ++closes; break;
        case AtemAction::EndTurn:        ++ends;   break;
        case AtemAction::EmitText:
            if (s.channel == AtemChannel::Reasoning) reasoning += s.text;
            else                                     content   += s.text;
            break;
        case AtemAction::Drop: break;
        }
    }
    CHECK_EQ_S(reasoning, "Because 2+2=4", "incremental reasoning");
    CHECK_EQ_S(content, "Four", "incremental content");
    if (opens != 1 || closes != 1 || ends != 1) {
        std::fprintf(stderr, "FAIL: transitions opens=%d closes=%d ends=%d "
                             "(want 1/1/1)\n", opens, closes, ends);
        ++g_fails;
    }

    const AtemParsed whole = atem_parse_response(
        " to=self<|message|>Because 2+2=4<|eom|>"
        "<|start|>assistant to=user<|message|>Four<|eot|>");
    CHECK_EQ_S(whole.reasoning, reasoning, "whole-string reasoning == incremental");
    CHECK_EQ_S(whole.content, content, "whole-string content == incremental");
}

static void test_chained_reasoning_does_not_nest() {
    // Two consecutive to=self turns: one open, one close, not two opens.
    const char * toks[] = {
        " to=self", "<|message|>", "a", "<|eom|>",
        "<|start|>", "assistant to=self", "<|message|>", "b", "<|eom|>",
        "<|start|>", "assistant to=user", "<|message|>", "done", "<|eot|>",
    };
    AtemSegmenter seg(true);
    int opens = 0, closes = 0;
    std::string reasoning, content;
    for (const char * t : toks) {
        const AtemStep s = seg.feed(t);
        if (s.action == AtemAction::OpenReasoning)  ++opens;
        if (s.action == AtemAction::CloseReasoning) ++closes;
        if (s.action == AtemAction::EmitText) {
            if (s.channel == AtemChannel::Reasoning) reasoning += s.text;
            else                                     content   += s.text;
        }
    }
    CHECK_EQ_S(reasoning, "ab", "both reasoning turns accumulate");
    CHECK_EQ_S(content, "done", "final answer is content");
    if (opens != 1 || closes != 1) {
        std::fprintf(stderr, "FAIL: chained to=self gave opens=%d closes=%d "
                             "(want 1/1)\n", opens, closes);
        ++g_fails;
    }
}

// ── ATEM tool-call dialect in tool_parser.cpp ──────────────────────────
static void test_tool_parser_atem_dialect() {
    const json tools = json::parse(
        R"([{"type":"function","function":{"name":"fs.read_file",)"
        R"("parameters":{"type":"object","properties":{)"
        R"("path":{"type":"string"},"lines":{"type":"integer"}}}}}])");
    const std::string text =
        "Reading it now.\n"
        "<atem:function_calls>\n<atem:invoke name=\"fs.read_file\">\n"
        "<atem:parameter name=\"path\">a.txt</atem:parameter>\n"
        "<atem:parameter name=\"lines\">42</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls>";
    const ToolParseResult r = parse_tool_calls(text, tools);
    if (r.tool_calls.size() != 1) {
        std::fprintf(stderr, "FAIL: atem dialect produced %zu calls (want 1)\n",
                     r.tool_calls.size());
        ++g_fails;
        return;
    }
    CHECK_EQ_S(r.tool_calls[0].name, "fs.read_file", "tool name");
    // Typed coercion through the declared schema: path stays a string, lines
    // becomes a number. Emitting both as strings is what a naive text scrape
    // would do and breaks strict clients.
    const json args = json::parse(r.tool_calls[0].arguments);
    CHECK_EQ_S(args.value("path", ""), "a.txt", "string arg");
    if (!args["lines"].is_number_integer() || args["lines"].get<int>() != 42) {
        std::fprintf(stderr, "FAIL: integer arg not coerced: %s\n",
                     args.dump().c_str());
        ++g_fails;
    }
    // The wrapper must not survive in the visible text.
    if (r.cleaned_text.find("<atem:") != std::string::npos) {
        std::fprintf(stderr, "FAIL: atem wrapper left in cleaned_text: %s\n",
                     r.cleaned_text.c_str());
        ++g_fails;
    }
}

static void test_tool_parser_atem_argument_containing_other_dialect() {
    // An argument value may legitimately contain another dialect's opener
    // (agentic prompts pass XML and code as arguments). The ATEM span is
    // claimed first precisely so a later pattern cannot half-consume it.
    const json tools = json::parse(
        R"([{"type":"function","function":{"name":"fs.write",)"
        R"("parameters":{"type":"object","properties":{)"
        R"("body":{"type":"string"}}}}}])");
    const std::string text =
        "<atem:function_calls>\n<atem:invoke name=\"fs.write\">\n"
        "<atem:parameter name=\"body\">see <tool_call>x</tool_call> here"
        "</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
    const ToolParseResult r = parse_tool_calls(text, tools);
    if (r.tool_calls.size() != 1) {
        std::fprintf(stderr, "FAIL: nested-opener case gave %zu calls (want 1)\n",
                     r.tool_calls.size());
        ++g_fails;
        return;
    }
    const json args = json::parse(r.tool_calls[0].arguments);
    CHECK_EQ_S(args.value("body", ""), "see <tool_call>x</tool_call> here",
               "argument text preserved verbatim");
}

static void test_tool_parser_atem_streaming_opener() {
    // The streaming emitter must recognise the opener to start buffering.
    size_t pos = 0;
    const std::string text = "ok\n<atem:function_calls>\n<atem:invoke name=\"a.b\">";
    if (!find_tool_syntax_start(text, json(), pos)) {
        std::fprintf(stderr, "FAIL: streaming opener not recognised\n");
        ++g_fails;
    } else if (text.compare(pos, 21, "<atem:function_calls>") != 0) {
        std::fprintf(stderr, "FAIL: opener position wrong (%zu)\n", pos);
        ++g_fails;
    }
    if (tool_syntax_holdback(json()) < 20) {
        std::fprintf(stderr, "FAIL: holdback too small for the ATEM opener\n");
        ++g_fails;
    }
}

int main() {
    test_tool_parser_atem_dialect();
    test_tool_parser_atem_argument_containing_other_dialect();
    test_tool_parser_atem_streaming_opener();
    test_reasoning_then_answer();
    test_recipient_marker_never_leaks();
    test_tool_call_segment_is_routed_by_recipient();
    test_unterminated_reasoning();
    test_incremental_matches_whole_string();
    test_chained_reasoning_does_not_nest();

    if (g_fails == 0) { std::printf("test_atem_stream: OK\n"); return 0; }
    std::fprintf(stderr, "test_atem_stream: %d failure(s)\n", g_fails);
    return 1;
}
