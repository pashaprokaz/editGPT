import logging
import os
from pathlib import Path
from typing import Any, Iterator, List, Optional

import gradio as gr
import pandas as pd
from gradio.themes.utils.colors import slate
from langchain_core.messages import AIMessage, HumanMessage
from pandas.io.formats.style import Styler

from edit_gpt.components.chat.agent.tools.edit_file import get_text_from_observation
from edit_gpt.components.chat.chat_manager import ChatManager
from edit_gpt.components.chat.data_preprocessor import AdditionalDataPreprocessor
from edit_gpt.components.diff_storage import DiffStorage
from edit_gpt.components.ingest_service import IngestService
from edit_gpt.constants import PROJECT_ROOT_PATH
from edit_gpt.utils.loaders import normalize_to_straight_slash

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)

UI_TAB_TITLE = "My Edit GPT"
SOURCES_SEPARATOR = "\n\n Sources: \n"

LLM_MODE = "LLM Chat"
AGENT_MODE = "Agent Chat"
MODES = [
    LLM_MODE,
    AGENT_MODE,
]
DEFAULT_MODE = MODES[0]

OPTIONS = ["RAG", "Web Search"]
DEFAULT_OPTIONS_VALUES = ["RAG"]


class EditGptUi:
    def __init__(
        self,
        ingest_service: IngestService,
        chat_manager: ChatManager,
        diff_storage: DiffStorage,
        additional_data_manager: AdditionalDataPreprocessor,
        filenames: Optional[List[str]] = None,
    ) -> None:
        self.diff_storage = diff_storage
        self._ingest_service = ingest_service
        self._chat_manager = chat_manager
        self.additional_data_manager = additional_data_manager

        # Cache the UI blocks
        self._ui_block = None

        self._selected_filename = None

        # Initialize system prompt based on default mode
        self.mode = MODES[0]
        self.options = DEFAULT_OPTIONS_VALUES

        self.filenames = filenames
        if self.filenames:
            self._upload_files(self.filenames)

    def _update_history(self, history: list[list[str]]) -> None:
        past_messages = []
        for step in history:
            past_messages.append(HumanMessage(content=step[0]))
            past_messages.append(AIMessage(content=step[1]))
        self._chat_manager.update_history(past_messages)

    def _chat(self, message: str, history: list[list[str]], *_: Any) -> Iterator[str]:
        self._update_history(history)

        if self.mode == AGENT_MODE:
            yield from self._process_agent_chat(message)
        elif self.mode == LLM_MODE:
            yield from self._process_llm_chat(message)

    def _process_agent_chat(self, message: str) -> Iterator[str]:
        additional_data = self.additional_data_manager.prepare_data(message, ["RAG"])
        result = self._chat_manager.agent_gen(message, **additional_data)
        output = result["output"]
        edited_files = {}
        for step in result["intermediate_steps"]:
            step_agent_action = step[0]
            step_agent_action_observation = step[1]
            if step_agent_action.tool == "edit_file_with_llm":
                file_path = Path(step_agent_action.tool_input["path"])
                edited_files[normalize_to_straight_slash(file_path)] = (
                    get_text_from_observation(step_agent_action_observation)
                )
        self.diff_storage.add_history_step(edited_files)
        yield output

    def _process_llm_chat(self, message: str) -> Iterator[str]:
        additional_data = self.additional_data_manager.prepare_data(
            message, self.options
        )
        for accumulated_text in self._chat_manager.chat_gen(message, **additional_data):
            yield accumulated_text
        self.diff_storage.add_history_step(None)

    def _list_ingested_files(self) -> List[str]:
        filenames = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.metadata.get("source", "[FILE NAME MISSING]")

            filenames.add(file_name)

        return list(filenames)

    def _list_ingested_files_styled(self) -> Styler:
        filenames = self._list_ingested_files()
        df = pd.DataFrame({"Ingested files": filenames})

        df = df.style.map(
            lambda x: (
                "background-color: rgb(220, 252, 231); color: black"
                if x in self.diff_storage.edited_files
                else ""
            )
        )

        return df

    def _upload_files(self, files: list[str]) -> None:
        logger.debug("Loading count=%s files", len(files))

        doc_ids_to_delete = []
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.metadata["source"]
                and ingested_document.metadata["source"] in files
            ):
                doc_ids_to_delete.append(ingested_document.metadata["id"])
        if len(doc_ids_to_delete) > 0:
            logger.info(
                "Uploading file(s) which were already ingested: %s document(s) will be replaced.",
                len(doc_ids_to_delete),
            )
            for doc_id in doc_ids_to_delete:
                self._ingest_service.delete(doc_id)

        self._ingest_service.bulk_ingest(files)

    def _delete_all_files(self) -> Any:
        ingested_files = self._ingest_service.list_ingested()
        logger.debug("Deleting count=%s files", len(ingested_files))
        for ingested_document in ingested_files:
            self._ingest_service.delete(ingested_document.metadata["id"])
        return [
            gr.DataFrame(self._list_ingested_files_styled()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _delete_selected_file(self) -> Any:
        logger.debug("Deleting selected %s", self._selected_filename)
        ingested_files = self._ingest_service.list_ingested()
        for ingested_document in ingested_files:
            if (
                ingested_document.metadata
                and ingested_document.metadata["source"] == self._selected_filename
            ):
                self._ingest_service.delete(ingested_document.metadata["id"])
        return [
            gr.DataFrame(self._list_ingested_files_styled()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _deselect_selected_file(self) -> Any:
        self._selected_filename = None
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> Any:
        self._selected_filename = select_data.value
        self._update_if_outdated(self._selected_filename)
        return self._update_file_view()

    def _update_if_outdated(self, filepath: str) -> None:
        selected_filename_path = normalize_to_straight_slash(Path(filepath))
        timestamp = os.path.getmtime(selected_filename_path)
        if (
            timestamp
            > self._ingest_service.ingested_files_last_edit_time[selected_filename_path]
        ):
            self._ingest_service.bulk_ingest([selected_filename_path])

    def _update_file_view(self):
        if self._selected_filename is None:
            return [
                gr.components.Button(),
                gr.components.Button(),
                gr.components.Textbox(),
                [("", "+")],
            ]
        file_content = Path(self._selected_filename).read_text(encoding="utf-8")
        selected_filename_path = normalize_to_straight_slash(
            Path(self._selected_filename)
        )

        selected_filename_last_diff = self.diff_storage.get_last_diff(
            selected_filename_path
        )
        if selected_filename_last_diff:
            content = selected_filename_last_diff.to_list()
        else:
            content = [(file_content, None)]

        return [
            gr.components.Button(interactive=True),
            gr.components.Button(interactive=True),
            gr.components.Textbox(selected_filename_path),
            content,
        ]

    def _set_current_mode(self, mode: str) -> Any:
        self.mode = mode
        if self.mode != AGENT_MODE:
            return [gr.Column(visible=False), gr.CheckboxGroup(visible=True)]
        else:
            return [gr.Column(visible=True), gr.CheckboxGroup(visible=False)]

    def _set_current_options(self, options: list[str]) -> Any:
        self.options = options

    def _save_changes(self) -> Any:
        for (
            file_path,
            diff_content,
        ) in self.diff_storage.edited_files_with_diff.items():
            Path(file_path).write_text(diff_content.text_content, encoding="utf-8")
        self._upload_files(list(self.diff_storage.edited_files))
        self.diff_storage.clear()
        return [*self._update_file_view(), self._list_ingested_files_styled()]

    def _undo(self):
        self.diff_storage.undo_history_step()

    def _clear(self):
        self.diff_storage.clear()

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=slate),
            css=".logo { "
            "display:flex;"
            "background-color: #C7BAFF;"
            "height: 80px;"
            "border-radius: 8px;"
            "align-content: center;"
            "justify-content: center;"
            "align-items: center;"
            "}"
            ".logo img { height: 25% }"
            ".contain { display: flex !important; flex-direction: column !important; }"
            "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
            "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
            "#col { height: calc(100vh - 112px - 16px) !important; }"
            "#file-view {height: calc(100vh - 112px - 16px) !important; overflow: auto !important; }",
        ) as blocks:
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    mode = gr.Radio(
                        MODES,
                        label="Mode",
                        value=DEFAULT_MODE,
                    )
                    self.options_component = gr.CheckboxGroup(
                        OPTIONS,
                        value=DEFAULT_OPTIONS_VALUES,
                        label="Options",
                        interactive=True,
                    )

                    self.options_component.input(
                        self._set_current_options, inputs=self.options_component
                    )

                    ingested_dataset = gr.Dataframe(
                        pd.DataFrame({"Ingested Files": []}),
                        type="array",
                        col_count=1,
                        headers=["File name"],
                        label="Ingested Files",
                        height=400,
                        interactive=False,
                        render=False,  # Rendered under the button
                    )

                    upload_button = gr.components.UploadButton(
                        "Upload File(s)",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                    )
                    upload_button.upload(
                        self._upload_files,
                        inputs=upload_button,
                        outputs=ingested_dataset,
                    )

                    ingested_dataset.change(
                        self._list_ingested_files_styled,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()
                    deselect_file_button = gr.components.Button(
                        "De-select selected file", size="sm", interactive=False
                    )
                    selected_text = gr.components.Textbox(
                        "All files", label="Selected for Query or Deletion", max_lines=1
                    )
                    save_changes_button = gr.components.Button(
                        "Save changes",
                        size="sm",
                        visible=True,
                        interactive=True,
                    )
                    delete_file_button = gr.components.Button(
                        "🗑️ Delete selected file",
                        size="sm",
                        visible=True,
                        interactive=False,
                    )
                    delete_files_button = gr.components.Button(
                        "⚠️ Delete ALL files",
                        size="sm",
                        visible=True,
                    )

                    deselect_file_button.click(
                        self._deselect_selected_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_file_button.click(
                        self._delete_selected_file,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_files_button.click(
                        self._delete_all_files,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )

                with gr.Column(scale=4, elem_id="col"):
                    chatbot = gr.Chatbot(
                        label="LLM: ",
                        show_copy_button=True,
                        elem_id="chatbot",
                        render=False,
                    )
                    chat_interface = gr.ChatInterface(self._chat, chatbot=chatbot)
                    chat_interface.undo_btn.click(self._undo)
                    chat_interface.retry_btn.click(self._undo)
                    chat_interface.clear_btn.click(self._clear)

                with gr.Column(
                    scale=4,
                    visible=False,
                ) as self.column_3:
                    file_content_component = gr.HighlightedText(
                        value=[("", "+")],
                        label="Diff",
                        show_legend=True,
                        color_map={"+": "green", "-": "red"},
                        elem_id="file-view",
                    )

                    ingested_dataset.select(
                        fn=self._selected_a_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                            file_content_component,
                        ],
                    )

                mode.change(
                    self._set_current_mode,
                    inputs=mode,
                    outputs=[self.column_3, self.options_component],
                )

                chatbot.change(
                    self._update_file_view,
                    outputs=[
                        delete_file_button,
                        deselect_file_button,
                        selected_text,
                        file_content_component,
                    ],
                )

                chatbot.change(
                    self._list_ingested_files_styled,
                    outputs=ingested_dataset,
                )
                save_changes_button.click(
                    self._save_changes,
                    outputs=[
                        delete_file_button,
                        deselect_file_button,
                        selected_text,
                        file_content_component,
                        ingested_dataset,
                    ],
                )

        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block
