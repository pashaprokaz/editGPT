import difflib
from pathlib import Path
from typing import Dict, List

from edit_gpt.components.file_storage import BaseFileReader


def diff_strings(str1, str2):
    d = difflib.Differ()
    diff = list(d.compare(str1.splitlines(), str2.splitlines()))

    content = []
    action = []
    for i in diff:
        if i[0] == "+":
            content.append(i[2:] + "\n")
            action.append("+")
        elif i[0] == "-":
            content.append(i[2:] + "\n")
            action.append("-")
        else:
            content.append(i[2:] + "\n")
            action.append(None)
    return Diff(content, action)


class Diff:
    def __init__(self, diff_lines: List[str], actions: List[str]):
        self.diff_lines = diff_lines
        self.actions = actions

    @property
    def text_content(self):
        result = ""
        for line, action in zip(self.diff_lines, self.actions):
            if action is None or action == "+":
                result += line
        return result

    def to_list(self):
        result = []
        for line, action in zip(self.diff_lines, self.actions):
            result.append((line, action))
        return result

    def __iter__(self):
        return iter(zip(self.diff_lines, self.actions))


class DiffStorage:
    def __init__(self):
        self._diffs: Dict[str, List[Diff]] = {}
        self.changes_history: List[List[str]] = []

    def add_history_step(self, edited_files: Dict[str, str] = None):
        self.changes_history.append([])
        if edited_files is None:
            return self.changes_history

        for filename, new_content in edited_files.items():
            try:
                file_content = Path(filename).read_text(encoding="utf-8")
            except (FileNotFoundError, PermissionError):
                continue
            diff_content = diff_strings(
                file_content,
                new_content,
            )
            if filename not in self._diffs:
                self._diffs[filename] = []
            self._diffs[filename].append(diff_content)
            self.changes_history[-1].append(filename)
        return self.changes_history

    def undo_history_step(self):
        last_changed_files = self.changes_history.pop()
        for filename in last_changed_files:
            self._diffs[filename].pop()
        self._diffs = {k: v for k, v in self._diffs.items() if v}

    def get_last_diff(self, filename: str) -> Diff | None:
        if filename not in self._diffs:
            return None
        return self._diffs[filename][-1]

    @property
    def edited_files(self):
        return self._diffs.keys()

    @property
    def edited_files_with_diff(self) -> Dict[str, Diff]:
        result = {}
        for filename, diff_content in self._diffs.items():
            result[filename] = diff_content[-1]
        return result

    def get_last_diff_content(self, filename: str) -> str | None:
        if filename not in self._diffs:
            return None
        return "\n".join(self._diffs[filename][-1].text_content)

    def clear(self):
        self._diffs = {}
        self.changes_history = []


class DiffReader(BaseFileReader):
    def __init__(self, storage: DiffStorage):
        self.storage = storage

    def read_from_filename(self, filename: str) -> str:
        if filename in self.storage.edited_files:
            return self.storage.get_last_diff_content(filename)
        else:
            return Path(filename).read_text(encoding="utf-8")
