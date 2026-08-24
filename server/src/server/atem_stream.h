// ATEM channel segmenter (Muse-Glimmer response side).
//
// The inverse of the ATEM chat template. A reply is a sequence of segments:
//
//   <|start|>assistant to=self<|message|>  …reasoning…  <|eom|>
//   <|start|>assistant to=tool.name<|message|>  …tool call…  <|eom|>
//   <|start|>assistant to=user<|message|>  …answer…  <|eot|>
//
// and the generation prompt already ends with `<|start|>assistant`, so the
// FIRST header arrives without its opener — the model's first emitted tokens
// are ` to=self` (or ` to=user`).
//
// Routing the segments matters for correctness, not tidiness: without it the
// recipient marker and the whole reasoning channel land in the visible
// answer (observed live: content began " to=selfWhat is 2+2?…"). Tool calls
// are addressed by recipient, so a client cannot even tell a tool turn from
// prose without this.
//
// The segmenter is INCREMENTAL and token-driven so the streaming and
// non-streaming paths can share it: feed each generated token's raw text,
// receive what to do with it. Header text is buffered and never emitted —
// the recipient is only known once `<|message|>` closes the header.

#pragma once

#include <string>

namespace dflash::common {

// Where the current segment's body should go.
enum class AtemChannel {
    None,       // no body open yet (still in a header)
    Reasoning,  // to=self
    Content,    // to=user, or an unaddressed segment
    ToolCall,   // to=<namespace>.<fn>
};

// What the caller should do with the token just fed.
enum class AtemAction {
    Drop,            // control token or header text — emit nothing
    EmitText,        // append `text` to the current channel
    OpenReasoning,   // reasoning channel opened (emit a <think> equivalent)
    CloseReasoning,  // reasoning channel closed (emit a </think> equivalent)
    EndTurn,         // <|eot|> — the turn is over
};

struct AtemStep {
    AtemAction  action = AtemAction::Drop;
    std::string text;          // for EmitText
    AtemChannel channel = AtemChannel::None;
    std::string recipient;     // for ToolCall segments: the `to=` value
};

class AtemSegmenter {
public:
    // `assistant_header_open` is true when the prompt ended with the
    // generation prompt `<|start|>assistant` (the normal case), so the model
    // resumes mid-header.
    explicit AtemSegmenter(bool assistant_header_open = true)
        : in_header_(assistant_header_open) {}

    // Feed one token. `raw` is the token's raw text (special tokens included
    // verbatim, e.g. "<|message|>").
    AtemStep feed(const std::string & raw);

    AtemChannel channel() const { return channel_; }
    bool in_header() const { return in_header_; }
    const std::string & recipient() const { return recipient_; }

private:
    AtemStep open_body();

    bool        in_header_ = true;
    std::string header_;        // accumulated header text (never emitted)
    AtemChannel channel_ = AtemChannel::None;
    std::string recipient_;
    bool        reasoning_open_ = false;
};

// Segment a COMPLETE response string (non-incremental convenience used by
// tests and by the non-streaming replay path).
struct AtemParsed {
    std::string reasoning;
    std::string content;
    std::string tool_text;      // concatenated bodies of tool-addressed turns
    std::string tool_recipient; // recipient of the first tool turn
};
AtemParsed atem_parse_response(const std::string & text,
                               bool assistant_header_open = true);

}  // namespace dflash::common
