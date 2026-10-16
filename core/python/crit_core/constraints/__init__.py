"""Structured output: constraint languages, their compiler and preparation, and the tokenizer byte table.

Standard library only (a `crit_core.text.Tokenizer` is passed in, never imported here). The native matchers that
mask tokens during generation are in crit/constraints.h; see core/src/constraints/README.md.
"""
from .compiler import (VERSION, CompileRequest, CompiledConstraint, CompilerCache, UnsupportedConstraint,
                       compile_language, make_compile_request)
from .grammar import ByteGrammar, GrammarError, GrammarState, json_object_matches, matches
from .preparation import Preparation, PreparationFailed, PreparationPending, PreparationService, PreparedMatcher
from .vocabulary import ByteVocabulary, byte_vocabulary, from_pieces

__all__ = ["VERSION", "ByteGrammar", "ByteVocabulary", "CompileRequest", "CompiledConstraint", "CompilerCache",
           "GrammarError", "GrammarState", "Preparation", "PreparationFailed", "PreparationPending",
           "PreparationService", "PreparedMatcher", "UnsupportedConstraint", "byte_vocabulary", "compile_language",
           "from_pieces", "json_object_matches", "make_compile_request", "matches"]
