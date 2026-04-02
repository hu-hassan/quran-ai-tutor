"""
Mistake Detection Step - Tutor-mode MVP.

Detects token-level mistakes between matched Quran ayah text and recognized chunk text.
"""

import re
from collections import defaultdict
from difflib import SequenceMatcher

from app.pipeline.base import PipelineStep, PipelineContext


def _normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text to a common consonantal skeleton for comparison.

    The reference text uses Uthmani Quranic script while Whisper outputs modern
    Arabic script. These differ in alef placement (e.g. ازوجا vs ازواجا) and
    hamza representation (e.g. ءايته vs آياته). The fix is to reduce both sides
    to the same bare consonantal skeleton before comparing, which eliminates all
    script-convention false positives while preserving real word-level mistakes.

    Steps:
      1. Remove all tashkeel / diacritics
      2. Remove Quranic annotation marks
      3. Normalize آ → ا, then all other alef variants → ا
      4. Remove hamza carriers (ء ئ ؤ)
      5. Remove ALL bare alef (ا) — eliminates Uthmani vs modern alef placement diffs
      6. Normalize ى → ي
      7. Normalize ة → ه
      8. Remove tatweel (ـ)
    """
    # 1. Remove diacritics / tashkeel (harakat, tanwin, shadda, sukun, etc.)
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670]', '', text)

    # 2. Remove Quranic annotation marks (U+06D6–U+06ED)
    text = re.sub(r'[\u06D6-\u06ED]', '', text)

    # 3. Normalize alef with madda (آ) → bare alef first
    text = re.sub(r'آ', 'ا', text)

    # 4. Normalize remaining alef variants → bare alef
    #    أ (U+0623), إ (U+0625), ٱ (U+0671 wasla)
    text = re.sub(r'[أإٱ]', 'ا', text)

    # 5. Remove hamza carriers: ء (U+0621), ئ (U+0626), ؤ (U+0624)
    text = re.sub(r'[ءئؤ]', '', text)

    # 6. Remove ALL bare alef (ا)
    #    This handles the core Uthmani vs modern spelling difference where
    #    alef letters appear in different positions (e.g. ازوجا vs ازواجا).
    #    Safe because alef alone never distinguishes Quranic words — the
    #    remaining consonants always differentiate meanings.
    text = re.sub(r'ا', '', text)

    # 7. Normalize dotless ya (ى U+0649) → ya (ي U+064A)
    text = re.sub(r'ى', 'ي', text)

    # 8. Normalize ta marbuta (ة U+0629) → ha (ه U+0647)
    text = re.sub(r'ة', 'ه', text)

    # 9. Remove tatweel / kashida (ـ U+0640)
    text = re.sub(r'ـ', '', text)

    # 10. Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class MistakeDetectionStep(PipelineStep):
    """Analyze matched chunks and annotate tutor-oriented mistake metadata."""

    def __init__(self):
        super().__init__()
        self.MIN_CONFIDENCE_FOR_AUTO_JUDGMENT = 0.75

    def validate_input(self, context: PipelineContext) -> bool:
        if not hasattr(context, 'matched_chunk_verses') or not context.matched_chunk_verses:
            if context.get('no_verse_match', False):
                return True
            self.logger.error("No matched_chunk_verses in context")
            return False
        return True

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.get('no_verse_match', False) and not context.matched_chunk_verses:
            expected_candidate = context.get('expected_ayah_candidate')
            expected_ayah_analysis = None
            if expected_candidate:
                actual_text = context.combined_transcription_normalized or context.final_transcription or ''
                expected_ayah_analysis = self._assess_expected_candidate(expected_candidate, actual_text)

            context.set('tutor_assessments', {})
            context.set('expected_ayah_analysis', expected_ayah_analysis)
            context.add_debug_info(self.name, {
                'verses_assessed': 0,
                'mistake_verses': 0,
                'needs_review_verses': 0,
                'skipped': True,
                'reason': 'no_verse_match',
                'has_expected_ayah_analysis': expected_ayah_analysis is not None
            })
            return context

        grouped = defaultdict(list)
        for chunk_data in context.matched_chunk_verses:
            for ayah in chunk_data.get('matched_ayahs', []):
                key = (ayah.get('surah_number'), ayah.get('ayah_number'))
                grouped[key].append(chunk_data)

        verse_assessments = {}
        for verse_key, verse_chunks in grouped.items():
            assessment = self._assess_verse(verse_chunks)
            verse_assessments[verse_key] = assessment

            # Attach same verse-level assessment to each chunk/ayah entry.
            for chunk in verse_chunks:
                chunk['tutor_assessment'] = assessment
                chunk['tutor_status'] = assessment['status']
                chunk['tutor_mistakes'] = assessment['mistakes']
                for ayah in chunk.get('matched_ayahs', []):
                    ayah['tutor_assessment'] = assessment
                    ayah['tutor_status'] = assessment['status']

        context.set('tutor_assessments', verse_assessments)
        context.add_debug_info(self.name, {
            'verses_assessed': len(verse_assessments),
            'mistake_verses': sum(1 for a in verse_assessments.values() if a['status'] == 'mistake'),
            'needs_review_verses': sum(1 for a in verse_assessments.values() if a['status'] == 'needs_review')
        })

        return context

    def _assess_verse(self, verse_chunks: list) -> dict:
        first_ayah = verse_chunks[0].get('matched_ayahs', [{}])[0]
        expected_text = first_ayah.get('text_normalized', '')
        expected_words = expected_text.split()
        degraded_match = any(
            chunk.get('degraded_match') or
            chunk.get('match_method', '').startswith('degraded') or
            any(ayah.get('degraded_match') for ayah in chunk.get('matched_ayahs', []))
            for chunk in verse_chunks
        )

        ordered_chunks = sorted(verse_chunks, key=lambda c: (c.get('chunk_index', 0), c.get('chunk_start_time', 0.0)))
        actual_words = []
        alignments = []
        for chunk in ordered_chunks:
            actual_words.extend(chunk.get('chunk_normalized_text', '').split())
            alignments.extend(chunk.get('word_alignments', []))

        mistakes, equal_count = self._detect_mistakes(expected_words, actual_words, alignments)

        expected_count = len(expected_words)
        actual_count = len(actual_words)
        mismatch_count = len(mistakes)
        word_accuracy = (equal_count / expected_count) if expected_count else 0.0

        confidences = [a.get('confidence') for a in alignments if isinstance(a, dict) and a.get('confidence') is not None]
        avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0

        if degraded_match:
            status = 'needs_review'
        elif expected_count == 0 or actual_count == 0:
            status = 'needs_review'
        elif avg_confidence < self.MIN_CONFIDENCE_FOR_AUTO_JUDGMENT:
            status = 'needs_review'
        elif mismatch_count == 0:
            status = 'correct'
        else:
            status = 'mistake'

        return {
            'status': status,
            'word_accuracy': round(word_accuracy, 4),
            'expected_word_count': expected_count,
            'recognized_word_count': actual_count,
            'mismatch_count': mismatch_count,
            'avg_word_confidence': round(avg_confidence, 4),
            'degraded_match': degraded_match,
            'mistakes': mistakes
        }

    def _detect_mistakes(self, expected_words: list, actual_words: list, alignments: list) -> tuple:
        # Normalize both sides for comparison only — original words kept for output display
        norm_expected = [_normalize_arabic(w) for w in expected_words]
        norm_actual = [_normalize_arabic(w) for w in actual_words]

        matcher = SequenceMatcher(None, norm_expected, norm_actual)
        mistakes = []
        equal_count = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                equal_count += (i2 - i1)
                continue

            if tag == 'replace':
                # Use original words for output so Arabic displays correctly
                exp_slice = expected_words[i1:i2]
                act_slice = actual_words[j1:j2]
                shared = min(len(exp_slice), len(act_slice))

                for idx in range(shared):
                    span = self._alignment_span(j1 + idx, j1 + idx + 1, alignments)
                    mistakes.append({
                        'type': 'substitution',
                        'expected': exp_slice[idx],
                        'actual': act_slice[idx],
                        'word_index': i1 + idx,
                        'actual_word_index': j1 + idx,
                        'start': span['start'],
                        'end': span['end']
                    })

                if len(exp_slice) > shared:
                    for idx in range(shared, len(exp_slice)):
                        mistakes.append({
                            'type': 'omission',
                            'expected': exp_slice[idx],
                            'actual': None,
                            'word_index': i1 + idx,
                            'actual_word_index': None,
                            'start': None,
                            'end': None
                        })

                if len(act_slice) > shared:
                    for idx in range(shared, len(act_slice)):
                        span = self._alignment_span(j1 + idx, j1 + idx + 1, alignments)
                        mistakes.append({
                            'type': 'insertion',
                            'expected': None,
                            'actual': act_slice[idx],
                            'word_index': None,
                            'actual_word_index': j1 + idx,
                            'start': span['start'],
                            'end': span['end']
                        })

            elif tag == 'delete':
                for idx, word in enumerate(expected_words[i1:i2]):
                    mistakes.append({
                        'type': 'omission',
                        'expected': word,
                        'actual': None,
                        'word_index': i1 + idx,
                        'actual_word_index': None,
                        'start': None,
                        'end': None
                    })

            elif tag == 'insert':
                for idx, word in enumerate(actual_words[j1:j2]):
                    span = self._alignment_span(j1 + idx, j1 + idx + 1, alignments)
                    mistakes.append({
                        'type': 'insertion',
                        'expected': None,
                        'actual': word,
                        'word_index': None,
                        'actual_word_index': j1 + idx,
                        'start': span['start'],
                        'end': span['end']
                    })

        return mistakes, equal_count

    @staticmethod
    def _alignment_span(start_idx: int, end_idx: int, alignments: list) -> dict:
        if not alignments:
            return {'start': None, 'end': None}

        safe_start = max(0, start_idx)
        safe_end = min(len(alignments), end_idx)
        if safe_start >= safe_end or safe_start >= len(alignments):
            return {'start': None, 'end': None}

        selected = alignments[safe_start:safe_end]
        starts = [a.get('start') for a in selected if isinstance(a, dict) and a.get('start') is not None]
        ends = [a.get('end') for a in selected if isinstance(a, dict) and a.get('end') is not None]

        return {
            'start': min(starts) if starts else None,
            'end': max(ends) if ends else None
        }

    def _assess_expected_candidate(self, candidate: dict, actual_text: str) -> dict:
        """Build mistake analysis against top suggested ayah when no strict match exists."""
        expected_text = (candidate or {}).get('text_normalized', '')
        expected_words = expected_text.split()
        actual_words = (actual_text or '').split()
        mistakes, equal_count = self._detect_mistakes(expected_words, actual_words, alignments=[])

        expected_count = len(expected_words)
        actual_count = len(actual_words)
        mismatch_count = len(mistakes)
        word_accuracy = (equal_count / expected_count) if expected_count else 0.0

        status = 'correct' if mismatch_count == 0 and expected_count > 0 else 'mistake'

        return {
            'status': status,
            'word_accuracy': round(word_accuracy, 4),
            'expected_word_count': expected_count,
            'recognized_word_count': actual_count,
            'mismatch_count': mismatch_count,
            'avg_word_confidence': 0.0,
            'degraded_match': True,
            'mistakes': mistakes,
            'surah_number': candidate.get('surah_number'),
            'ayah_number': candidate.get('ayah_number'),
            'similarity': candidate.get('similarity'),
            'source': 'closest_ayah_candidate'
        }
