import pandas as pd
import regex as re
import spacy
import string
import json
from copy import deepcopy

from apteryx_transformers.parsers.parser_utils import remove_tables, get_start_stop
from apteryx_transformers.GLOBALS import BLOCKLIST, STRIPLIST

"""
^ : start at beginning of string
\W*? : match non-alphanumeric, but as little as possible
[0-9]+ : match numeric, at least one.
"""
NUM_PATTERN = re.compile("^\W*?([0-9]+[.A-z]*.*?[0-9]*)\W*?")

DEFAULT_CONFIG = {"window": 5, "remove_table_text": True, "severity": 0}


class NPParser:
    def __init__(
        self,
        spacy_model="en_core_web_lg",
        add_to_blocklist: list = [],
        config=DEFAULT_CONFIG,
    ):
        print("Initializing!")
        print(f"Loading spacy model: {spacy_model}")
        self.nlp = spacy.load(spacy_model)
        self.blocklist = BLOCKLIST + add_to_blocklist
        print(f"Blocklist: {self.blocklist}")
        self.config = config

    def clean_np(self, tokens):
        clean = [
            t.text_with_ws.upper()
            for t in tokens
            if not any(
                [
                    t.is_stop,
                    # Allow hyphen exception
                    (t.is_punct and not t.text == "-"),
                    t.text_with_ws.lower() in self.blocklist,
                ]
            )
        ]
        if clean:
            # Join text and remove l/r whitespace, if any.
            clean_txt = "".join(clean).strip()
            for s in STRIPLIST:
                clean_txt = clean_txt.replace(s.upper(), "")
            return clean_txt
        else:
            return None

    def set_config(self, new_config):
        self.config.update(new_config)
        return self.config

    def get_nps(self, s):
        start_stop = get_start_stop(self._get_nps(s, **self.config), s)

        return start_stop

    def _get_nps(self, s, window=5, remove_table_text=True, severity=0):
        """
        Given an input string and a lookahead window, return a dataframe of detected noun phrases and their part numbers.
        """
        s = str(s)
        if remove_table_text:
            s = remove_tables(s, severity=severity)

        doc = self.nlp(s)

        data = []
        for chunk in doc.noun_chunks:
            tok_end = chunk.end
            next_text = "".join(
                [t.text_with_ws for t in doc[tok_end : tok_end + window]]
            )

            # Detect following number.
            num_match = re.match(NUM_PATTERN, next_text)
            if num_match:
                group = num_match.group(1)
                # Iteratively remove punctuation from the match, if found.
                while group[-1] in string.punctuation:
                    group = group[:-1]
                num_match = group

            data.append([chunk, num_match, next_text])

        df = pd.DataFrame.from_records(data)
        df.columns = ["np_raw", "num", "trailing"]

        # Remove stopwords from np
        df["np_clean"] = df.np_raw.apply(self.clean_np)
        # Drop nps that were completely eliminated by the earlier step.
        df = df.dropna(subset=["np_clean", "np_raw"]).reset_index(drop=True)
        # Convert raw nps to text
        df["np_raw"] = df.np_raw.apply(
            lambda tokens: "".join([t.text_with_ws for t in tokens])
        )

        df["in_blocklist"] = df.np_clean.apply(
            lambda noun_phrase: any(
                [
                    i.lower() in self.blocklist
                    for i in re.split("\s", noun_phrase.strip())
                ]
            )
        )

        return (
            df[~df.in_blocklist].dropna(subset=["in_blocklist"]).reset_index(drop=True)
        )

    def prep_report(self, df, group: str, cols: list):
        if len(df.dropna(subset=[group])) > 0:
            new_df = (
                df.groupby(group)
                .apply(lambda g: pd.Series([list(set(g[c].values)) for c in cols]))
                .reset_index()
            )
            new_df.columns = [group] + cols
            new_df["start"] = new_df.start.apply(lambda s: sorted(list(s)))
            new_df["end"] = new_df.end.apply(lambda s: sorted(list(s)))
            return new_df
        else:
            return None

    def report(self, s):
        nps = self.get_nps(s)
        np_groups = self.prep_report(nps, "np_clean", ["num", "np_raw", "start", "end"])
        num_groups = self.prep_report(
            nps, "num", ["np_clean", "np_raw", "start", "end"]
        )

        to_return = {
            "main": nps.to_dict(orient="records"),
            "nps": {
                "all": np_groups.to_dict(orient="records"),
                "multiple": np_groups[
                    np_groups.apply(lambda x: len(x.num) > 1, axis=1)
                ].to_dict(orient="records"),
            }
            if isinstance(np_groups, pd.DataFrame)
            else None,
            "nums": {
                "all": num_groups.to_dict(orient="records"),
                "multiple": num_groups[
                    num_groups.apply(lambda x: len(x.np_clean) > 1, axis=1)
                ].to_dict(orient="records"),
            }
            if isinstance(num_groups, pd.DataFrame)
            else None,
        }

        return to_return

    """
    Broken: handle None type
    """

    def report_json(self, s):
        return self.report(s)
        # return serialize_report(self.report(s))


if __name__ == "__main__":
    p = NPParser(spacy_model="en_core_web_sm")
    txt = "The present disclosure is concerned with a visual apparatus and a method for creation of artificial vision. In particular, the present disclosure provides an interface and method for controlling a visual prosthesis (i.e. device) implanted in an individual patient (i.e. subject) to create artificial vision. FIG. 1 shows a visual prosthesis apparatus"
    print(p.report_json(txt))
