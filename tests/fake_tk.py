"""A minimal stand-in for tkinter, enough to drive imageflow.ui.main_window without a display.

It records the widget tree and queues ``after`` callbacks, so a test can click buttons and pump the
event loop deterministically. It checks the window's logic, not how it looks.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Iterator


class Widget:
    def __init__(self, parent: "Widget | None" = None, **options: Any) -> None:
        self.parent = parent
        self.options = dict(options)
        self.children: list[Widget] = []
        self.destroyed = False
        self.disabled = False
        if parent is not None:
            parent.children.append(self)

    def pack(self, **kwargs: Any) -> None:
        self.pack_options = kwargs

    def winfo_children(self) -> list["Widget"]:
        return list(self.children)

    def destroy(self) -> None:
        for child in list(self.children):
            child.destroy()
        self.destroyed = True
        if self.parent is not None and self in self.parent.children:
            self.parent.children.remove(self)

    def configure(self, **kwargs: Any) -> None:
        self.options.update(kwargs)

    def state(self, spec: list[str]) -> None:
        if "disabled" in spec:
            self.disabled = True
        if "!disabled" in spec:
            self.disabled = False

    def walk(self) -> Iterator["Widget"]:
        yield self
        for child in list(self.children):
            yield from child.walk()


class Frame(Widget):
    pass


class Label(Widget):
    @property
    def text(self) -> str:
        return str(self.options.get("text", ""))


class Button(Widget):
    def invoke(self) -> None:
        if not self.disabled:
            self.options["command"]()

    @property
    def text(self) -> str:
        return str(self.options.get("text", ""))


class Checkbutton(Widget):
    pass


class Combobox(Widget):
    def __init__(self, parent: Widget | None = None, **options: Any) -> None:
        super().__init__(parent, **options)
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class Progressbar(Widget):
    def start(self, interval: int = 50) -> None:
        self.started = True


class BooleanVar:
    def __init__(self, value: bool = False) -> None:
        self._value = bool(value)

    def get(self) -> bool:
        return self._value

    def set(self, value: bool) -> None:
        self._value = bool(value)


class Tk(Widget):
    def __init__(self) -> None:
        super().__init__(None)
        self._after: list[Any] = []

    def title(self, text: str) -> None:
        self.window_title = text

    def minsize(self, width: int, height: int) -> None:
        pass

    def protocol(self, name: str, callback: Any) -> None:
        pass

    def mainloop(self) -> None:
        pass

    def after(self, ms: int, callback: Any) -> None:
        self._after.append(callback)

    def run_after(self) -> None:
        """Run the callbacks queued so far. Callbacks queued while running wait for the next call."""
        queued, self._after = self._after, []
        for callback in queued:
            callback()


class MessageBox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def showerror(self, title: str, message: str) -> None:
        self.calls.append(("error", message))

    def showwarning(self, title: str, message: str) -> None:
        self.calls.append(("warning", message))

    def showinfo(self, title: str, message: str) -> None:
        self.calls.append(("info", message))


class FileDialog:
    answer = ""

    def askdirectory(self, **kwargs: Any) -> str:
        return self.answer


def install(monkeypatch: Any) -> tuple[types.ModuleType, MessageBox, FileDialog]:
    """Put the fake tkinter in sys.modules and import a fresh imageflow.ui.main_window against it."""
    messagebox, filedialog = MessageBox(), FileDialog()
    ttk = types.SimpleNamespace(Frame=Frame, Label=Label, Button=Button, Checkbutton=Checkbutton, Combobox=Combobox, Progressbar=Progressbar)
    tkinter = types.ModuleType("tkinter")
    tkinter.Tk, tkinter.BooleanVar = Tk, BooleanVar
    tkinter.ttk, tkinter.messagebox, tkinter.filedialog = ttk, messagebox, filedialog
    for name, module in {"tkinter": tkinter, "tkinter.ttk": ttk, "tkinter.messagebox": messagebox, "tkinter.filedialog": filedialog}.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("imageflow.ui.main_window", None)
    import imageflow.ui.main_window as main_window

    return main_window, messagebox, filedialog
