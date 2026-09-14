"""Small structural checks for displayed model output, not semantic judgments."""

from __future__ import annotations

import re

_SOURCE_QUOTATION_PATTERN = (
    r'["“]([^"”]+)["”]'
    r"|(?<!\w)'(?=\S)(.+?)(?<=\S)'(?!\w)"
    r"|(?<!\w)‘(?=\S)(.+?)(?<=\S)’(?!\w)"
)


def extract_source_quotations(text: str) -> list[str]:
    """Extract paired quotes, treating in-word apostrophes as part of the text."""
    return [
        phrase.strip()
        for match in re.finditer(_SOURCE_QUOTATION_PATTERN, text, re.DOTALL)
        for phrase in match.groups()
        if phrase is not None and phrase.strip()
    ]


def is_single_question(text: str) -> bool:
    """Recognize one English question; this does not establish tone or grounding."""
    question = text.strip()
    if not question.endswith("?") or question.count("?") != 1:
        return False
    question_start = re.compile(
        r"^(?:(?:and|but|so)\s+)?(?:what|which|who|whom|whose|when|where|why|how|"
        r"since\s+when|for\s+(?:what|whom)|in\s+what|to\s+whom|"
        r"is|are|was|were|do|does|did|have|has|had|can|could|would|will|should)\b",
        re.IGNORECASE,
    )
    # Ignore commas inside quoted content, but retain a terminal quoted comma
    # that can punctuate the surrounding clause: 'When you wrote "logs," what…?'
    structure = re.sub(
        _SOURCE_QUOTATION_PATTERN,
        lambda match: (
            "QUOTATION"
            + (
                ","
                if any(part and part.rstrip().endswith(",") for part in match.groups())
                else ""
            )
        ),
        question,
        flags=re.DOTALL,
    )
    # 'When did you buy apples, pears and bananas?' is already interrogative.
    if re.match(
        r"^when\s+(?:is|are|was|were|do|does|did|have|has|had|can|could|"
        r"would|will|should)\b",
        structure,
        re.IGNORECASE,
    ):
        return True
    # A contextual opening can precede the interrogative clause. This accepts
    # ordinary fronted questions without treating any comma-led directive as one.
    has_contextual_clause = False
    for comma in re.finditer(",", structure):
        prefix, suffix = structure[: comma.start()], structure[comma.end() :].strip()
        if re.match(
            r"^(?:if|when|whenever|while|after|before|given|looking back|"
            r"thinking back|in|on|at|from|with)\b",
            prefix,
            re.IGNORECASE,
        ) and not re.search(r"[?!.;]", prefix):
            has_contextual_clause = True
            if question_start.match(suffix):
                return True
    if has_contextual_clause:
        return False
    # Reject statements/directives with a question mark appended. Auxiliary-led
    # questions are structurally valid; whether they are leading needs review.
    return question_start.match(structure) is not None


def openai_response_refusal(response: object) -> str | None:
    """Give refusal precedence even when the response also contains text."""
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            refusal = getattr(content, "refusal", None)
            if refusal or getattr(content, "type", None) == "refusal":
                return str(refusal or "Provider refused the request")
    return None
