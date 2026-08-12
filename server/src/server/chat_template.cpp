// Chat template renderer implementation.

#include "chat_template.h"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <stdexcept>

namespace dflash::common {

// Qwen3.5 tool preamble — matches the official Jinja template exactly.
static const char QWEN3_TOOL_PREAMBLE[] =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";

static const char QWEN3_TOOL_SUFFIX[] =
    "\n</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

// Serialise JSON the way the reference ATEM template's `tojson` filter does:
// insertion order preserved, `", "` between items and `": "` after keys.
// nlohmann's dump() sorts keys (json is a std::map) and emits no spaces, so
// a straight dump() produces a prompt that differs from the model's own
// rendering in every tool schema — invisible in a smoke test, and exactly
// the kind of drift that costs tool-calling accuracy.
static void atem_json_dump(const nlohmann::ordered_json & v, std::string & out) {
    if (v.is_object()) {
        out += '{';
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            out += nlohmann::json(it.key()).dump();
            out += ": ";
            atem_json_dump(it.value(), out);
        }
        out += '}';
    } else if (v.is_array()) {
        out += '[';
        bool first = true;
        for (const auto & e : v) {
            if (!first) out += ", ";
            first = false;
            atem_json_dump(e, out);
        }
        out += ']';
    } else {
        out += v.dump();
    }
}

static std::string atem_json_str(const nlohmann::ordered_json & v) {
    std::string out;
    atem_json_dump(v, out);
    return out;
}

// Current UTC date as YYYY-MM-DD — the ATEM system turn's `Current date:`
// line (the template calls strftime_now). Overridable via
// DFLASH_ATEM_CURRENT_DATE so prompt rendering can be pinned in tests and
// in reproducible eval harnesses.
static std::string atem_current_date() {
    if (const char * override_date = std::getenv("DFLASH_ATEM_CURRENT_DATE")) {
        if (*override_date) return override_date;
    }
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &utc);
    return std::string(buf);
}

ChatFormat chat_format_for_arch(const std::string & arch) {
    if (arch == "deepseek4") return ChatFormat::DEEPSEEK4;
    if (arch == "laguna") return ChatFormat::LAGUNA;
    if (arch == "gemma4") return ChatFormat::GEMMA4;
    if (arch == "muse-glimmer") return ChatFormat::ATEM;
    // qwen35, qwen3 use the Qwen3/ChatML format
    return ChatFormat::QWEN3;
}

std::string render_chat_template(
    const std::vector<ChatMessage> & messages,
    ChatFormat format,
    bool add_generation_prompt,
    bool enable_thinking,
    const std::string & tools_json)
{
    std::string result;
    bool has_tools = !tools_json.empty() && tools_json != "[]" && tools_json != "null";

    switch (format) {
    case ChatFormat::QWEN3: {
        // Qwen3/3.5 ChatML format:
        //   <|im_start|>system\n[tool preamble +] content<|im_end|>\n
        //   <|im_start|>user\nHello<|im_end|>\n
        //   <|im_start|>assistant\n...

        // Determine if the first message is a system message.
        size_t start_idx = 0;
        std::string system_content;
        if (!messages.empty() && messages[0].role == "system") {
            system_content = messages[0].content;
            start_idx = 1;
        }

        // Emit system message with tool preamble if tools are present.
        if (has_tools) {
            result += "<|im_start|>system\n";
            result += QWEN3_TOOL_PREAMBLE;
            result += '\n';
            result += tools_json;
            result += QWEN3_TOOL_SUFFIX;
            if (!system_content.empty()) {
                result += "\n\n";
                result += system_content;
            }
            result += "<|im_end|>\n";
        } else if (!system_content.empty()) {
            result += "<|im_start|>system\n";
            result += system_content;
            result += "<|im_end|>\n";
        }

        // Render remaining messages.
        bool in_tool_response = false;
        for (size_t i = start_idx; i < messages.size(); i++) {
            const auto & msg = messages[i];

            if (msg.role == "tool") {
                // Qwen3.5 template: tool responses are grouped inside a user
                // message wrapped in <tool_response> tags.
                if (!in_tool_response) {
                    result += "<|im_start|>user";
                    in_tool_response = true;
                }
                result += "\n<tool_response>\n";
                result += msg.content;
                result += "\n</tool_response>";
                // Close user block if next message is not a tool message.
                bool next_is_tool = (i + 1 < messages.size() &&
                                     messages[i + 1].role == "tool");
                if (!next_is_tool) {
                    result += "<|im_end|>\n";
                    in_tool_response = false;
                }
            } else {
                result += "<|im_start|>";
                result += msg.role;
                result += '\n';
                result += msg.content;
                result += "<|im_end|>\n";
            }
        }

        if (add_generation_prompt) {
            result += "<|im_start|>assistant\n";
            if (!enable_thinking) {
                // Qwen3 thinking disabled: inject closed think block so the
                // model skips reasoning and generates the answer directly.
                result += "<think>\n\n</think>\n\n";
            } else {
                // Qwen3.6 enable_thinking: pre-open the thinking block so the
                // model actually enters reasoning mode. Verified against the
                // official Qwen3.6 chat_template.jinja:
                //   enable_thinking=true  → suffix `assistant\n<think>\n`
                //   enable_thinking=false → suffix `assistant\n<think>\n\n</think>\n\n`
                // Without this prefix, Qwen3.6 stays in non-thinking mode
                // even when the client opts in, defeating the thinking-budget
                // mechanism entirely.
                result += "<think>\n";
            }
        }
        break;
    }

    case ChatFormat::LAGUNA: {
        // Laguna XS.2 format (verified against
        // poolside/Laguna-XS.2/chat_template.jinja, 2026-05-25):
        //
        //   〈|EOS|〉<system>
        //   {system_content_or_default}
        //   </system>
        //   <user>
        //   {user_content}
        //   </user>
        //   <assistant>
        //   <think>      ← if enable_thinking (gen prompt)
        //   </think>     ← if NOT enable_thinking (gen prompt — empty
        //                    think block; model continues with answer)
        //
        // Tokens 18/19/23/24/25/26 are special (<think>, </think>,
        // <assistant>, </assistant>, <tool_call>, </tool_call>);
        // <user> / <system> / </user> / </system> are not added-tokens
        // and tokenize as regular bytes — which is what the upstream
        // template expects.
        //
        // The DeepSeek-style template that used to live here
        // (<｜begin▁of▁sentence｜> / <｜User｜> / <｜Assistant｜>)
        // was a copy-paste error; the tokens don't exist in the laguna
        // vocab, so the model saw replacement-character garbage in its
        // prompt and degenerated into echoing the user message back
        // with `<���Assistant���>` artifacts.
        static const std::string DEFAULT_SYSTEM =
            "You are a helpful, conversationally-fluent assistant "
            "made by Poolside. You are here to be helpful to users "
            "through natural language conversations.";
        result = "〈|EOS|〉";

        const bool has_system =
            !messages.empty() && messages[0].role == "system";
        const std::string system_content = has_system
            ? messages[0].content
            : DEFAULT_SYSTEM;
        // System always emitted (default or supplied) when there's any
        // content or tools — matches the template's
        // `if (system_message and system_message.strip()) or tools`.
        if (!system_content.empty() || has_tools) {
            result += "<system>\n";
            if (!system_content.empty()) result += system_content;
            if (has_tools) {
                // Tools block per the upstream template: each tool schema as
                // raw JSON inside <available_tools>, then the calling
                // instruction with the <tool_call>/<arg_key>/<arg_value>
                // example (thinking and non-thinking variants).
                result += "\n\n### Tools\n\n"
                          "You may call functions to assist with the user query.\n"
                          "All available function signatures are listed below:\n"
                          "<available_tools>\n";
                try {
                    const nlohmann::json tools = nlohmann::json::parse(tools_json);
                    for (const auto & t : tools) {
                        result += t.dump();
                        result += "\n";
                    }
                } catch (const std::exception &) {
                    result += tools_json;
                    result += "\n";
                }
                result += "</available_tools>\n\n";
                if (enable_thinking) {
                    result += "Wrap your thinking in '<think>', '</think>' tags, "
                              "followed by a function call. For each function call, "
                              "return an unescaped XML-like object with function name "
                              "and arguments within '<tool_call>' and '</tool_call>' "
                              "tags, like here:\n"
                              "<think> your thoughts here </think>\n"
                              "<tool_call>function-name\n"
                              "<arg_key>argument-key</arg_key>\n"
                              "<arg_value>value-of-argument-key</arg_value>\n"
                              "</tool_call>";
                } else {
                    result += "For each function call, return an unescaped XML-like "
                              "object with function name and arguments within "
                              "'<tool_call>' and '</tool_call>' tags, like here:\n"
                              "<tool_call>function-name\n"
                              "<arg_key>argument-key</arg_key>\n"
                              "<arg_value>value-of-argument-key</arg_value>\n"
                              "</tool_call>";
                }
            }
            result += "\n</system>\n";
        }

        const size_t start_idx = has_system ? 1 : 0;
        for (size_t i = start_idx; i < messages.size(); i++) {
            const auto & msg = messages[i];
            if (msg.role == "user") {
                result += "<user>\n";
                result += msg.content;
                result += "\n</user>\n";
            } else if (msg.role == "assistant") {
                // Past assistant turns: wrap in <assistant>...</assistant>.
                // Reasoning content is extracted by the upstream template;
                // for our minimal renderer the content is rendered as-is
                // (including any embedded <think>...</think> blocks).
                result += "<assistant>\n";
                result += msg.content;
                result += "\n</assistant>\n";
            } else if (msg.role == "tool") {
                result += "<tool_response>\n";
                result += msg.content;
                result += "\n</tool_response>\n";
            } else if (msg.role == "system") {
                // Additional system messages beyond the first
                result += "<system>\n";
                result += msg.content;
                result += "\n</system>\n";
            }
        }
        if (add_generation_prompt) {
            result += "<assistant>\n";
            if (enable_thinking) {
                result += "<think>";
            } else {
                // Empty think block — model jumps straight to answer.
                result += "</think>";
            }
        }
        break;
    }

    case ChatFormat::GEMMA4: {
        // Gemma4 format (see the chat template embedded in the GGUF
        // metadata of google/gemma-4-26B-A4B-it):
        //
        //   <bos>
        //   <|turn>system
        //   [<|think|>\n      ← if enable_thinking]
        //   {system content}
        //   <turn|>
        //   <|turn>user
        //   {msg}<turn|>
        //   <|turn>model
        //   [<|channel>thought\n<channel|>  ← if NOT enable_thinking]
        //
        // The trailing channel-thought guard is the same trick Qwen3
        // uses (`<think>\n\n</think>\n\n`): when thinking is disabled
        // we pre-fill an empty thought channel so the model SKIPS
        // emitting its own. Without this, Gemma4 self-emits
        // `<|channel>thought\n…<channel|>` which then partially leaks
        // into the visible content because the channel tokens were
        // never opened from the prompt side.
        //
        // The `<|think|>` opener at the start of the system turn is
        // the inverse: it signals "this conversation is in thinking
        // mode" so the model's channel sequence routes to reasoning.
        const bool has_system = !messages.empty() && messages[0].role == "system";
        const bool emit_system_turn = enable_thinking || has_system || has_tools;
        result = "<bos>";

        size_t start_idx = 0;
        std::string system_content;
        if (has_system) {
            system_content = messages[0].content;
            start_idx = 1;
        }

        // System turn — emitted when there's actual system content OR
        // we need somewhere to put the `<|think|>` opener.
        if (emit_system_turn) {
            result += "<|turn>system\n";
            if (enable_thinking) {
                // Per the GGUF chat template: "Inject Thinking token at
                // the very top of the FIRST system turn".
                result += "<|think|>\n";
            }
            if (!system_content.empty()) {
                result += system_content;
            }
            // TODO: tool definitions block (`<|tool>…<tool|>`) goes here
            // when tools_json is non-empty. Out of scope for the
            // budget-signaling fix.
            (void)tools_json;
            result += "<turn|>\n";
        }

        // User/assistant turns. Unlike the previous implementation we
        // don't prepend system content to the first user message — the
        // system turn above already carries it (or there isn't one).
        for (size_t i = start_idx; i < messages.size(); i++) {
            const auto & msg = messages[i];
            std::string role = msg.role;
            if (role == "assistant") role = "model";

            result += "<|turn>";
            result += role;
            result += '\n';
            result += msg.content;
            result += "<turn|>\n";
        }
        if (add_generation_prompt) {
            result += "<|turn>model\n";
            if (!enable_thinking) {
                // Empty thought-channel guard: model will skip its own
                // `<|channel>thought…<channel|>` block since this one
                // already sits in the prompt. Matches the GGUF
                // template's "if not enable_thinking" branch.
                result += "<|channel>thought\n<channel|>";
            }
        }
        break;
    }

    case ChatFormat::DEEPSEEK4: {
        // DeepSeek V4 Flash DSML renderer, matching the ds4 reference server:
        //   <｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜></think>
        // Completed assistant turns are terminated with <｜end▁of▁sentence｜>.
        bool has_system = false;
        std::string system_content;
        for (const auto & msg : messages) {
            if (msg.role != "system") continue;
            if (!system_content.empty()) system_content += "\n\n";
            system_content += msg.content;
            has_system = true;
        }

        result = "<｜begin▁of▁sentence｜>";
        if (has_tools) {
            // Tool schema rendering is not implemented for the native DSML
            // path yet; keep the JSON visible in the system prefix rather than
            // silently dropping it.
            result += tools_json;
            if (has_system) result += "\n\n";
        }
        result += system_content;

        bool pending_assistant = false;
        bool pending_tool_result = false;
        for (const auto & msg : messages) {
            if (msg.role == "system") {
                continue;
            } else if (msg.role == "user") {
                result += "<｜User｜>";
                result += msg.content;
                pending_assistant = true;
                pending_tool_result = false;
            } else if (msg.role == "tool" || msg.role == "function") {
                if (!pending_tool_result) result += "<｜User｜>";
                result += "<tool_result>";
                result += msg.content;
                result += "</tool_result>";
                pending_assistant = true;
                pending_tool_result = true;
            } else if (msg.role == "assistant") {
                if (pending_assistant) {
                    result += "<｜Assistant｜>";
                    result += enable_thinking ? "<think>" : "</think>";
                }
                result += msg.content;
                result += "<｜end▁of▁sentence｜>";
                pending_assistant = false;
                pending_tool_result = false;
            }
        }

        if (add_generation_prompt) {
            result += "<｜Assistant｜>";
            result += enable_thinking ? "<think>" : "</think>";
        }
        break;
    }

    case ChatFormat::ATEM: {
        // ATEM (Muse-Glimmer). Ported from the chat template embedded in the
        // artifact's GGUF metadata; the turn grammar is:
        //
        //   <|start|>system<|message|>{system}\n\nReasoning strength: high.
        //   [\n\n{tool defs}]\n\n# Valid recipients: …<|eot|>
        //   <|start|>user<|message|>{msg}<|eot|>
        //   <|start|>tool {name}<|message|><tool_output name="{name}">
        //   {result}\n</tool_output><|eot|>
        //   <|start|>assistant[ to={recipient}]<|message|>{msg}<|eot|>
        //   <|start|>assistant                      ← generation prompt
        //
        // `<|eom|>` ends a turn that CONTINUES the same role (reasoning
        // channel, chained tool calls); `<|eot|>` ends the turn outright.
        // Both are control tokens the sampler must be able to emit: serving
        // that strips them scores TOOL 0/30 on the muse golden suite even
        // when the model's ATEM blocks are perfect.
        //
        // Assistant tool-call turns are replayed verbatim: normalize_chat_
        // messages() feeds back the model's own raw <atem:function_calls>
        // text, so no re-serialisation happens here (and none should — the
        // arguments-to-XML mapping is the model's, not ours).
        const std::string kReasoning = "Reasoning strength: high.";
        // Template default (`knowledge_cutoff`); the artifact ships no
        // override, so it is a constant of the format, not a policy choice.
        const std::string kKnowledgeCutoff = "2026-01-04";

        result = "<|begin_of_text|>";

        // Recipient list: "self" (reasoning channel), one entry per tool
        // namespace, then "user".
        std::string recipients = "# Valid recipients: \"self\"";
        std::vector<std::string> namespaces;
        if (has_tools) {
            try {
                const auto tools = nlohmann::ordered_json::parse(tools_json);
                for (const auto & t : tools) {
                    const nlohmann::ordered_json & fn =
                        t.contains("function") ? t["function"] : t;
                    const std::string fname = fn.value("name", "");
                    if (fname.empty()) continue;
                    const std::string ns = fname.substr(0, fname.find('.'));
                    if (std::find(namespaces.begin(), namespaces.end(), ns) ==
                        namespaces.end()) {
                        namespaces.push_back(ns);
                    }
                }
            } catch (const std::exception &) {
                // Malformed tools JSON: fall through with no namespaces
                // rather than refusing the request. The tool-definition
                // block below surfaces the raw JSON so it is not silently
                // dropped.
            }
        }
        for (const auto & ns : namespaces) {
            recipients += ", \"" + ns + ".*\"";
        }
        recipients += ", \"user\".";

        std::string tool_defs;
        if (has_tools) {
            tool_defs =
                "In this environment you have access to a set of tools you "
                "can use to answer the user's question.\n\n"
                "You can invoke a function by writing a "
                "\"<atem:function_calls>\" block like the following:\n"
                "<atem:function_calls>\n"
                "<atem:invoke name=\"$FUNCTION_NAME\">\n"
                "<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE"
                "</atem:parameter>\n...\n</atem:invoke>\n"
                "</atem:function_calls>\n\n"
                "String and scalar parameters should be specified as is, "
                "while lists and objects should use JSON format. Note that "
                "spaces for string values are not stripped. The output is "
                "not expected to be valid XML and is parsed with regular "
                "expressions.\n"
                "Here are the functions available in JSONSchema format:\n"
                "// Tool metadata\n";
            for (const auto & ns : namespaces) {
                tool_defs += "{\"name\": \"" + ns + "\", \"description\": \"\"}\n";
            }
            tool_defs += "// Function schemas";
            try {
                const auto tools = nlohmann::ordered_json::parse(tools_json);
                for (const auto & t : tools) {
                    const nlohmann::ordered_json & fn =
                        t.contains("function") ? t["function"] : t;
                    tool_defs += "\n{\"name\": " +
                                 nlohmann::json(fn.value("name", "")).dump() +
                                 ", \"description\": " +
                                 nlohmann::json(fn.value("description", "")).dump() +
                                 ", \"parameters\": " +
                                 (fn.contains("parameters")
                                      ? atem_json_str(fn["parameters"])
                                      : std::string("{}")) +
                                 "}";
                }
            } catch (const std::exception &) {
                tool_defs += "\n" + tools_json;
            }
            tool_defs +=
                "\n\nHere's an example of how to call a function in the tool "
                "set:\n"
                "(If the tool namespace is not specified, invoke the function "
                "directly as `example_function_name` rather than "
                "`example_tool_name.example_function_name`)\n\n"
                "to=example_tool_name.example_function_name\n\n"
                "<atem:function_calls>\n"
                "<atem:invoke name=\"example_tool_name.example_function_name\">\n"
                "<atem:parameter name=\"example_parameter_1\">value_1"
                "</atem:parameter>\n"
                "<atem:parameter name=\"example_parameter_2\">This is the "
                "value for the second parameter\nthat can span\n"
                "\"multiple\" lines\n</atem:parameter>\n"
                "</atem:invoke>\n</atem:function_calls>";
        }

        auto emit_system = [&](const std::string & content, bool synthesized) {
            result += "<|start|>system<|message|>";
            result += content;
            if (synthesized) {
                // Only the SYNTHESIZED system turn carries the dateline; a
                // caller-supplied system message replaces it wholesale.
                result += "\nKnowledge cutoff: " + kKnowledgeCutoff + ".";
                result += "\nCurrent date: " + atem_current_date() + ".";
            }
            result += "\n\n" + kReasoning;
            if (has_tools) result += "\n\n" + tool_defs;
            result += "\n\n" + recipients;
            result += "<|eot|>";
        };

        bool has_system = false;
        for (const auto & m : messages) {
            if (m.role == "system") { has_system = true; break; }
        }
        if (!has_system) {
            emit_system("You are a helpful AI assistant.", /*synthesized=*/true);
        }

        for (size_t i = 0; i < messages.size(); i++) {
            const auto & msg = messages[i];
            // A turn whose successor repeats the role continues the same
            // channel, so it closes with <|eom|> rather than <|eot|>.
            const bool same_role_next =
                (i + 1 < messages.size()) && messages[i + 1].role == msg.role;
            const char * end_token = same_role_next ? "<|eom|>" : "<|eot|>";

            if (msg.role == "system") {
                emit_system(msg.content, /*synthesized=*/false);
            } else if (msg.role == "user") {
                result += "<|start|>user<|message|>" + msg.content + "<|eot|>";
            } else if (msg.role == "tool") {
                const std::string tname =
                    !msg.name.empty() ? msg.name : msg.tool_call_id;
                result += "<|start|>tool " + tname + "<|message|>";
                result += "<tool_output name=\"" + tname + "\">\n";
                result += msg.content;
                result += "\n</tool_output><|eot|>";
            } else if (msg.role == "assistant") {
                // Every assistant turn is addressed: a replayed tool-call
                // turn to the tool it invokes, anything else to the user.
                // The bare `<|start|>assistant<|message|>` form does not
                // occur in the reference rendering.
                const size_t k = msg.content.find("<atem:invoke name=\"");
                std::string recipient = "user";
                if (k != std::string::npos) {
                    const size_t b = k + std::strlen("<atem:invoke name=\"");
                    const size_t e = msg.content.find('"', b);
                    if (e != std::string::npos) {
                        recipient = msg.content.substr(b, e - b);
                    }
                }
                const bool is_tool_call = recipient != "user";
                result += "<|start|>assistant to=" + recipient + "<|message|>";
                result += msg.content;
                // A turn addressed to the user ends it (<|eot|>). A tool-call
                // turn takes the continuation rule: <|eom|> only when the
                // NEXT message repeats the assistant role (chained calls),
                // otherwise <|eot|> because a `tool` turn follows.
                result += is_tool_call ? end_token : "<|eot|>";
            }
        }

        if (add_generation_prompt) {
            result += "<|start|>assistant";
        }
        break;
    }
    }

    return result;
}

// ─── Jinja path ─────────────────────────────────────────────────────────
//
// Render via a Jinja chat template (e.g. froggeric Qwen3.6 template). Each
// thread caches the most-recently-parsed program for its template source,
// so steady-state cost is just the runtime execute (parse happens once per
// process per template).

namespace {

struct JinjaCache {
    std::string                       src;
    std::shared_ptr<jinja::program>   prog;
};

static thread_local JinjaCache tls_jinja_cache;

static std::shared_ptr<jinja::program> get_or_parse(const std::string & template_src) {
    if (tls_jinja_cache.prog && tls_jinja_cache.src == template_src) {
        return tls_jinja_cache.prog;
    }
    jinja::lexer lex;
    jinja::lexer_result lex_res;
    try {
        lex_res = lex.tokenize(template_src);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("jinja lexer: ") + e.what());
    }
    auto prog = std::make_shared<jinja::program>(jinja::parse_from_tokens(lex_res));
    tls_jinja_cache.src  = template_src;
    tls_jinja_cache.prog = prog;
    return prog;
}

}  // namespace

std::string render_chat_template_jinja(
    const std::string & template_src,
    const std::vector<ChatMessage> & messages,
    const std::string & bos_token,
    const std::string & eos_token,
    bool add_generation_prompt,
    bool enable_thinking,
    const std::string & tools_json)
{
    if (template_src.empty()) {
        throw std::runtime_error("render_chat_template_jinja: template_src is empty");
    }

    auto prog = get_or_parse(template_src);

    // Build the JSON input that mirrors llama.cpp's
    // common_chat_template_direct_apply_impl. Field names must match the
    // names the Jinja templates expect (messages, tools, bos_token,
    // eos_token, add_generation_prompt, enable_thinking).
    nlohmann::ordered_json messages_j = nlohmann::ordered_json::array();
    for (const auto & m : messages) {
        nlohmann::ordered_json mj;
        mj["role"]    = m.role;
        mj["content"] = m.content;
        if (!m.tool_call_id.empty()) {
            mj["tool_call_id"] = m.tool_call_id;
        }
        messages_j.push_back(std::move(mj));
    }

    nlohmann::ordered_json inputs;
    inputs["messages"]              = std::move(messages_j);
    inputs["bos_token"]             = bos_token;
    inputs["eos_token"]             = eos_token;
    inputs["add_generation_prompt"] = add_generation_prompt;
    inputs["enable_thinking"]       = enable_thinking;

    bool has_tools = !tools_json.empty() && tools_json != "[]" && tools_json != "null";
    if (has_tools) {
        try {
            inputs["tools"] = nlohmann::ordered_json::parse(tools_json);
        } catch (const std::exception & e) {
            throw std::runtime_error(
                std::string("render_chat_template_jinja: failed to parse tools JSON: ") + e.what());
        }
    }

    jinja::context ctx(template_src);
    try {
        jinja::global_from_json(ctx, inputs, /*mark_input=*/false);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("jinja global_from_json: ") + e.what());
    }

    try {
        jinja::runtime rt(ctx);
        jinja::value results = rt.execute(*prog);
        auto parts = jinja::runtime::gather_string_parts(results);
        return parts->as_string().str();
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("jinja runtime: ") + e.what());
    }
}

}  // namespace dflash::common
