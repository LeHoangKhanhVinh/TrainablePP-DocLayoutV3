"""Label-to-id mapping for PP-DocLayoutV3 training.

The default 25-class list mirrors ``models/inference.yml`` ``label_list`` (the
original PaddlePaddle training label order). The shipped
``models/config.json`` has duplicate names in ``id2label`` (e.g. "footer"
maps to both 8 and 9) — this module restores a canonical, deduplicated list.

For fine-tuning on a different label set, build a :class:`LabelMap` from your
own list (or pass it via the YAML config under ``label_list``).
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- default 25-class label list (PP-DocLayoutV3 native) ---------------------------

LABEL_LIST: list[str] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
]

LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {i: name for i, name in enumerate(LABEL_LIST)}

# Aliases used when normalizing labels read from the dataset. The empty string
# value means "skip this shape entirely".
LABEL_ALIASES: dict[str, str] = {
    "formula": "display_formula",
    "reading_order": "",
}


# ---- runtime LabelMap (used when user supplies a custom label_list) -----------------


@dataclass
class LabelMap:
    """Stateful label normalizer.

    Use :func:`LabelMap.build` to construct one from a user-supplied label
    list (and optional aliases). Pass ``label_map.label2id`` to the dataset
    and ``label_map.id2label`` to the model config.
    """

    label_list: list[str]
    label2id: dict[str, int]
    id2label: dict[int, str]
    aliases: dict[str, str]

    @classmethod
    def build(cls, label_list: list[str] | None = None, aliases: dict[str, str] | None = None) -> "LabelMap":
        chosen = list(label_list) if label_list else list(LABEL_LIST)
        if len(chosen) != len(set(chosen)):
            raise ValueError(f"label_list contains duplicates: {chosen}")
        l2i = {name: i for i, name in enumerate(chosen)}
        i2l = {i: name for i, name in enumerate(chosen)}
        merged_aliases = dict(LABEL_ALIASES)
        if aliases:
            merged_aliases.update(aliases)
        return cls(label_list=chosen, label2id=l2i, id2label=i2l, aliases=merged_aliases)

    def normalize(self, name: str) -> str | None:
        """Return canonical label name or ``None`` if the shape should be skipped."""
        if name in self.label2id:
            return name
        aliased = self.aliases.get(name)
        if aliased == "":
            return None
        if aliased and aliased in self.label2id:
            return aliased
        return None

    @property
    def num_classes(self) -> int:
        return len(self.label_list)


# Default global instance — used when callers don't pass a custom map.
DEFAULT_LABEL_MAP = LabelMap.build()


def normalize_label(name: str) -> str | None:
    """Module-level shortcut using the default 25-class map."""
    return DEFAULT_LABEL_MAP.normalize(name)


__all__ = [
    "LABEL_LIST",
    "LABEL2ID",
    "ID2LABEL",
    "LABEL_ALIASES",
    "LabelMap",
    "DEFAULT_LABEL_MAP",
    "normalize_label",
]
