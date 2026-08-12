#include "atem_stream.h"

#include <cctype>
#include <cstddef>

namespace dflash::common {

namespace {

constexpr const char * kStart   = "<|start|>";
constexpr const char * kMessage = "<|message|>";
constexpr const char * kEom     = "<|eom|>";
constexpr const char * kEot     = "<|eot|>";

// Recipient from a header like "assistant to=self" or " to=fs.read_file".
// Absent `to=` means the segment is addressed to the user (the template's
// default when a message carries no recipient).
std::string recipient_of(const std::string & header) {
    const size_t at = header.find("to=");
    if (at == std::string::npos) return "user";
    size_t end = at + 3;
    while (end < header.size() &&
           !std::isspace((unsigned char)header[end])) {
        ++end;
    }
    std::string r = header.substr(at + 3, end - (at + 3));
    return r.empty() ? "user" : r;
}

}  // namespace

AtemStep AtemSegmenter::open_body() {
    recipient_ = recipient_of(header_);
    header_.clear();
    in_header_ = false;

    AtemStep step;
    step.recipient = recipient_;
    if (recipient_ == "self") {
        channel_ = AtemChannel::Reasoning;
        step.channel = channel_;
        // Only announce an OPEN if one is not already outstanding; a model
        // that chains two `to=self` turns must not emit nested <think>.
        if (!reasoning_open_) {
            reasoning_open_ = true;
            step.action = AtemAction::OpenReasoning;
        } else {
            step.action = AtemAction::Drop;
        }
        return step;
    }
    if (recipient_ == "user") {
        channel_ = AtemChannel::Content;
    } else {
        channel_ = AtemChannel::ToolCall;
    }
    step.channel = channel_;
    // Leaving the reasoning channel for ANY other recipient closes it. The
    // close is owed even when the model jumps straight from `to=self` into a
    // tool call without an intervening user turn.
    if (reasoning_open_) {
        reasoning_open_ = false;
        step.action = AtemAction::CloseReasoning;
    } else {
        step.action = AtemAction::Drop;
    }
    return step;
}

AtemStep AtemSegmenter::feed(const std::string & raw) {
    AtemStep step;
    step.channel   = channel_;
    step.recipient = recipient_;

    if (raw == kStart) {
        // A new segment begins; its body channel is unknown until <|message|>.
        in_header_ = true;
        header_.clear();
        channel_ = AtemChannel::None;
        step.action = AtemAction::Drop;
        step.channel = channel_;
        return step;
    }
    if (raw == kMessage) {
        if (!in_header_) {
            // A stray <|message|> outside a header: drop it rather than
            // leaking a control token into the visible answer.
            step.action = AtemAction::Drop;
            return step;
        }
        return open_body();
    }
    if (raw == kEom) {
        // Segment ends and another follows. The channel closes; the pending
        // reasoning close (if any) is emitted when the NEXT segment opens,
        // so a reply that ends mid-reasoning still closes cleanly below.
        channel_ = AtemChannel::None;
        in_header_ = true;      // the next <|start|> may be implicit
        header_.clear();
        step.action = AtemAction::Drop;
        step.channel = channel_;
        return step;
    }
    if (raw == kEot) {
        channel_ = AtemChannel::None;
        step.action = AtemAction::EndTurn;
        step.channel = channel_;
        return step;
    }

    if (in_header_) {
        // Header text (" to=self", "assistant", …) is state, not output.
        header_ += raw;
        step.action = AtemAction::Drop;
        return step;
    }

    step.action = AtemAction::EmitText;
    step.text   = raw;
    return step;
}

AtemParsed atem_parse_response(const std::string & text,
                               bool assistant_header_open) {
    // Walk the string, splitting on the control markers, and feed the
    // segmenter the same way a token stream would.
    AtemSegmenter seg(assistant_header_open);
    AtemParsed out;

    static const char * kMarkers[] = {kStart, kMessage, kEom, kEot};

    size_t i = 0;
    while (i < text.size()) {
        size_t best = std::string::npos;
        const char * best_marker = nullptr;
        for (const char * m : kMarkers) {
            const size_t at = text.find(m, i);
            if (at != std::string::npos && (best == std::string::npos || at < best)) {
                best = at;
                best_marker = m;
            }
        }
        const size_t chunk_end = (best == std::string::npos) ? text.size() : best;
        if (chunk_end > i) {
            const AtemStep s = seg.feed(text.substr(i, chunk_end - i));
            if (s.action == AtemAction::EmitText) {
                switch (s.channel) {
                case AtemChannel::Reasoning: out.reasoning += s.text; break;
                case AtemChannel::ToolCall:
                    out.tool_text += s.text;
                    if (out.tool_recipient.empty()) out.tool_recipient = s.recipient;
                    break;
                default:                     out.content += s.text;   break;
                }
            }
        }
        if (best == std::string::npos) break;
        seg.feed(best_marker);
        i = best + std::string(best_marker).size();
    }
    return out;
}

}  // namespace dflash::common
