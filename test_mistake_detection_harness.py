"""Small runnable harness for MistakeDetectionStep MVP."""

from app.pipeline.base import PipelineContext
from app.pipeline.steps.mistake_detection import MistakeDetectionStep


def run_harness() -> None:
    context = PipelineContext(final_transcription="وقاتلوهم حك لتكون فتنة")
    context.matched_chunk_verses = [
        {
            'chunk_index': 0,
            'chunk_start_time': 2.38,
            'chunk_end_time': 6.21,
            'chunk_normalized_text': 'وقاتلوهم حك',
            'word_alignments': [
                {'word': 'وقاتلوهم', 'start': 2.56, 'end': 3.98, 'confidence': 0.85},
                {'word': 'حك', 'start': 4.70, 'end': 5.50, 'confidence': 0.85}
            ],
            'matched_ayahs': [
                {
                    'surah_number': 2,
                    'ayah_number': 193,
                    'text': 'dummy',
                    'text_normalized': 'وقاتلوهم حتى',
                    'is_basmalah': False,
                    'similarity': 94.0
                }
            ]
        }
    ]

    step = MistakeDetectionStep()
    result = step.process(context)

    ayah_assessment = result.matched_chunk_verses[0]['matched_ayahs'][0]['tutor_assessment']
    assert ayah_assessment['status'] == 'mistake', ayah_assessment
    assert ayah_assessment['mismatch_count'] >= 1, ayah_assessment
    assert any(m['type'] == 'substitution' for m in ayah_assessment['mistakes']), ayah_assessment

    print('MistakeDetectionStep harness passed.')
    print(ayah_assessment)


if __name__ == '__main__':
    run_harness()

