import csv
import difflib
import logging
import string
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from mt_metrics_eval.data import EvalSet


@dataclass
class MQMRating:

    START_TAG = "<v>"
    END_TAG = "</v>"

    system: str
    doc: str
    doc_id: int
    seg_id: int
    rater: str
    source: str
    annotated_target: str
    category: str
    severity: str
    original_target: str
    comment: Optional[str] = None

    @property
    def has_target_mismatch(self) -> bool:
        return self.annotated_target_without_annotations != self.original_target

    @property
    def has_target_mismatch_without_punctuation(self) -> bool:
        remove_punctuation = lambda s: s.translate(str.maketrans('', '', string.punctuation + "’‘, "))
        return remove_punctuation(self.annotated_target_without_annotations).lower() != remove_punctuation(self.original_target).lower()

    @property
    def annotated_target_without_annotations(self) -> str:
        return self.annotated_target.replace(self.START_TAG, "").replace(self.END_TAG, "").strip()

    def get_cleaned_annotated_target(self) -> str:
        if not self.has_target_mismatch:
            return self.annotated_target

        original_target = self.original_target
        annotated_target = self.annotated_target
        assert annotated_target.count(self.START_TAG) == annotated_target.count(self.END_TAG) <= 1
        if annotated_target.count(self.START_TAG) == 0:
            return original_target

        # Remove accidental duplications
        annotated_span = annotated_target[annotated_target.index(self.START_TAG) + len(self.START_TAG):annotated_target.index(self.END_TAG)]
        if f"{annotated_span.strip()}{annotated_span.strip()}" not in original_target:
            annotated_target = annotated_target.replace(
                f"{annotated_span.strip()}{self.START_TAG}{annotated_span}{self.END_TAG}",
                f"{self.START_TAG}{annotated_span}{self.END_TAG}"
            )
            annotated_target = annotated_target.replace(
                f"{self.START_TAG}{annotated_span}{self.END_TAG}{annotated_span.strip()}",
                f"{self.START_TAG}{annotated_span}{self.END_TAG}"
            )
        if f"{annotated_span.strip()} {annotated_span.strip()}" not in original_target:
            annotated_target = annotated_target.replace(
                f"{annotated_span.strip()} {self.START_TAG}{annotated_span}{self.END_TAG}",
                f"{self.START_TAG}{annotated_span}{self.END_TAG}"
            )
            annotated_target = annotated_target.replace(
                f"{self.START_TAG}{annotated_span}{self.END_TAG} {annotated_span.strip()}",
                f"{self.START_TAG}{annotated_span}{self.END_TAG}"
            )

        matcher = difflib.SequenceMatcher(a=list(original_target), b=list(annotated_target), autojunk=False)
        opcodes = matcher.get_opcodes()
        start_tag_seen = False
        end_tag_seen = False
        original_start = None
        original_end = None
        annotated_start = None
        annotated_end = None
        for opcode, opcode_original_start, opcode_original_end, opcode_annotated_start, opcode_annotated_end in opcodes:
            annotated_span = annotated_target[opcode_annotated_start:opcode_annotated_end]
            # Check if we are inside or outside of tags
            if self.START_TAG in annotated_span:
                start_tag_seen = True
            if not start_tag_seen or end_tag_seen:
                continue  # Edit outside of error span
            if self.END_TAG in annotated_span:
                end_tag_seen = True

            # Ignore edits outside of tags but within opcode
            if self.START_TAG in annotated_span:
                opcode_original_start += annotated_span.index(self.START_TAG)
                opcode_annotated_start += annotated_span.index(self.START_TAG)
            if self.END_TAG in annotated_span:
                opcode_original_end -= len(annotated_span) - annotated_span.index(self.END_TAG) - len(self.END_TAG)
                opcode_annotated_end -= len(annotated_span) - annotated_span.index(self.END_TAG) - len(self.END_TAG)

            if original_start is None:
                original_start = opcode_original_start
            if annotated_start is None:
                annotated_start = opcode_annotated_start
            original_end = opcode_original_end
            annotated_end = opcode_annotated_end

        error_span_in_original = original_target[original_start:original_end]
        error_span_in_annotated = annotated_target[annotated_start:annotated_end]
        assert error_span_in_annotated.startswith(self.START_TAG) and error_span_in_annotated.endswith(self.END_TAG)
        error_span_in_annotated_without_tags = error_span_in_annotated[len(self.START_TAG):-len(self.END_TAG)]
        if not error_span_in_original.strip():
            # Deletion or no-op
            cleaned_target = original_target
        elif error_span_in_annotated_without_tags.strip() != error_span_in_original.strip() and (
                error_span_in_annotated_without_tags.strip().startswith(error_span_in_original) or
                error_span_in_annotated_without_tags.strip().endswith(error_span_in_original)
        ):
            # Addition
            cleaned_target = original_target
        else:
            cleaned_target = "".join([
                original_target[:original_start],
                self.START_TAG,
                original_target[original_start:original_end],
                self.END_TAG,
                original_target[original_end:]
            ])
        assert cleaned_target.replace(self.START_TAG, "").replace(self.END_TAG, "") == self.original_target
        assert cleaned_target.count(self.START_TAG) == cleaned_target.count(self.END_TAG) <= 1
        return cleaned_target


class MQMData:

    def __init__(self, wmt_data: EvalSet, mqm_path: Path):
        self.wmt_data = wmt_data
        self.mqm_path = mqm_path
        self.mqm_ratings = self.load_ratings()

    def load_ratings(self) -> List[MQMRating]:
        mqm_ratings = []
        with open(self.mqm_path, newline='') as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for line in reader:
                rating = MQMRating(
                    system=line["system"].strip(),
                    doc=line["doc"].strip(),
                    doc_id=int(line["doc_id"]),
                    seg_id=int(line["seg_id"]),
                    rater=line["rater"].strip(),
                    source=line["source"].strip(),
                    annotated_target=line["target"].strip(),
                    category=line["category"].strip(),
                    severity=line["severity"].strip() if line["severity"] else None,
                    original_target="",
                )
                if self.wmt_data._name == "wmt21.news":
                    if self.wmt_data._lp == "en-de":
                        if line["comment"]:
                            rating.comment = line["comment"]
                    if rating.system.startswith("hyp."):
                        rating.system = rating.system[4:]
                    if rating.system.startswith("ref."):
                        rating.system = "ref-" + rating.system[4:]
                rating.original_target = self.wmt_data.sys_outputs[rating.system][rating.seg_id - 1]
                mqm_ratings.append(rating)
        return mqm_ratings

    def save_cleaned_annotations(self, out_path: Path = None):
        out_path = out_path or self.mqm_path.with_suffix(".cleaned.tsv")
        with open(out_path, newline='', mode="w") as f:
            writer = csv.DictWriter(f, delimiter="\t", quotechar='', quoting=csv.QUOTE_NONE,
                fieldnames="system	doc	doc_id	seg_id	rater	source	target	category	severity".split("\t"))
            writer.writeheader()
            for rating in self.mqm_ratings:
                try:
                    cleaned_target = rating.get_cleaned_annotated_target()
                except AssertionError:
                    logging.warning(traceback.format_exc())
                    cleaned_target = rating.original_target
                writer.writerow({
                    "system": rating.system,
                    "doc": rating.doc,
                    "doc_id": rating.doc_id,
                    "seg_id": rating.seg_id,
                    "rater": rating.rater,
                    "source": rating.source,
                    "target": cleaned_target,
                    "category": rating.category,
                    "severity": rating.severity,
                })


if __name__ == '__main__':
    data_dir = Path(__file__).parent / "newstest2020"
    language_pair = "en-de"
    # language_pair = "zh-en"
    original_path = data_dir / language_pair.replace("-", "") / f'mqm_newstest2020_{language_pair.replace("-", "")}.tsv'
    wmt_data = EvalSet("wmt20", language_pair)
    mqm_data = MQMData(wmt_data, original_path)
    mqm_data.save_cleaned_annotations()
