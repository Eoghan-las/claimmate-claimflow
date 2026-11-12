"""Physical text-version contracts, without semantic or approval inference."""
from .base import BlockView, CanonicalTextView, ImageRegion, PageView, SourceSpan, resolve_quote
from .physical import adapt_public_document

__all__ = ["BlockView", "CanonicalTextView", "ImageRegion", "PageView", "SourceSpan",
           "adapt_public_document", "resolve_quote"]
